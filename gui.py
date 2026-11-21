#!/usr/bin/env python3
"""NiceGUI frontend for the Power Flow Simulator."""

import asyncio
import io
import os
import socket
import tempfile
import threading
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from nicegui import app, ui

from data_readers import (
    ALIGN_AUTO,
    ALIGN_CALENDAR,
    DEFAULT_TARIFF_GROUP,
    NETWORK_TARIFF_FILE,
    SUPPORTED_EXTENSIONS,
    load_network_tariffs,
)
from power_flow_simulator import (
    BATTERY_SUBSIDY_BASES,
    DEFAULT_REFERENCE_SOLAR_KW,
    MultiScenarioAnalyzer,
    PowerFlowSimulator,
    SubsidyConfig,
)
from price_cache import DEFAULT_CACHE_DIR, cached_price_file, market_price_file, update_price_cache

matplotlib.use("Agg")

# Plot style scoped to GUI only — does not affect power_flow_simulator.py CLI plots.
# Applied via plt.rc_context(_GUI_PLOT_RC) around each figure creation.
_DARK_BG = plt.style.library["dark_background"]
_GUI_PLOT_RC = {
    **_DARK_BG,
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.facecolor": "#1d1d1d",
    "axes.facecolor": "#2a2a2a",
    "savefig.facecolor": "#1d1d1d",
}

_ACCEPT = ",".join(SUPPORTED_EXTENSIONS)

# Persistent temp dir for uploaded files (lives for the process lifetime)
_upload_dir = Path(tempfile.mkdtemp(prefix="solar_sim_"))
# Original filename of each upload, by saved stem (e.g. "production" -> "my_pv.xlsx")
_upload_names: dict[str, str] = {}


def _display_name(path: str) -> str:
    """Short name for a data file: the original upload name, else the path as given."""
    p = Path(path)
    if p.parent == _upload_dir:
        return _upload_names.get(p.stem, p.name)
    return path


def _candidate_names(name: str) -> list[str]:
    """``name`` first, then the same stem with every other supported extension."""
    stem = Path(name).stem
    return [name] + [f"{stem}{ext}" for ext in SUPPORTED_EXTENSIONS if f"{stem}{ext}" != name]


def _find_data_file(name: str) -> str:
    """Locate a data file: uploaded files first, then data/ subdir, then cwd.

    Any supported extension (CSV, XLSX, XLSM, XLS) is accepted for the given stem,
    so ``"consumption.csv"`` also finds ``consumption.xlsx``.
    """
    for directory in (_upload_dir, Path("data"), None):
        for candidate_name in _candidate_names(name):
            candidate = directory / candidate_name if directory else Path(candidate_name)
            if candidate.exists():
                return str(candidate)
    return str(Path("data") / name)


async def _handle_upload(event, target_name: str, label_widget) -> bool:
    """Save an uploaded file under ``target_name``'s stem, keeping its own extension.

    ``event`` is NiceGUI's ``UploadEventArguments`` (file in ``event.file``). Returns
    whether the file was saved (False for an unsupported type).
    """
    name = event.file.name
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        label_widget.text = f"{name}: unsupported type (use {', '.join(SUPPORTED_EXTENSIONS)})"
        return False
    stem = Path(target_name).stem
    # Remove earlier uploads of this input in other formats so they can't shadow it.
    for old_ext in SUPPORTED_EXTENSIONS:
        (_upload_dir / f"{stem}{old_ext}").unlink(missing_ok=True)
    dest = _upload_dir / f"{stem}{ext}"
    dest.write_bytes(await event.file.read())
    _upload_names[stem] = name
    label_widget.text = f"{stem}: found ({name})"
    return True


def _data_files_section(inputs):
    """Create file upload section — collapsed when files are already found.

    Also holds the reference capacity of the production file (registered in ``inputs``),
    since it describes that file rather than the simulated system.
    """

    def found(name: str) -> str | None:
        path = _find_data_file(name)
        return path if Path(path).exists() else None

    def refresh() -> None:
        prod, cons = found("production.csv"), found("consumption.csv")
        if prod and cons:
            summary.text = f"Using: {_display_name(prod)}, {_display_name(cons)}"
            summary.classes(replace="text-caption text-positive")
        else:
            summary.text = "Upload production and consumption files (CSV, XLSX or XLS)"
            summary.classes(replace="text-caption text-warning")
        for label, stem, path in (
            (prod_label, "production", prod),
            (cons_label, "consumption", cons),
        ):
            uploaded = path and Path(path).parent == _upload_dir
            label.text = f"{stem}: " + (
                f"found ({_display_name(path)})" if uploaded else "found" if path else "missing"
            )

    async def on_upload(event, target_name: str, label_widget) -> None:
        if await _handle_upload(event, target_name, label_widget):
            refresh()

    both_found = bool(found("production.csv") and found("consumption.csv"))
    with ui.expansion("Data Files", icon="folder_open", value=not both_found).classes("w-full"):
        summary = ui.label()

        with ui.row().classes("w-full q-gutter-sm items-center"):
            prod_label = ui.label().classes("text-caption")
            ui.upload(
                label="Upload",
                auto_upload=True,
                on_upload=lambda e: on_upload(e, "production.csv", prod_label),
            ).props(f"accept={_ACCEPT} flat dense").classes("max-w-[180px]")
            _number(
                "Reference power", "reference_solar_kw", inputs, step=0.1, min=0.1, suffix="kWp"
            ).classes("w-36").tooltip(
                "Installed capacity of the plant that recorded the production file; "
                "generation is scaled from it to the simulated solar size"
            )

        with ui.row().classes("w-full q-gutter-sm items-center"):
            cons_label = ui.label().classes("text-caption")
            ui.upload(
                label="Upload",
                auto_upload=True,
                on_upload=lambda e: on_upload(e, "consumption.csv", cons_label),
            ).props(f"accept={_ACCEPT} flat dense").classes("max-w-[180px]")

    refresh()


EXPORT_MODE_FIXED = "Fixed price"
EXPORT_MODE_MARKET = "Market price (SI day-ahead, last 365 days)"
EXPORT_PRICES_NAME = "export_prices.csv"


def _export_prices(params: dict) -> tuple[str | None, str]:
    """(price file, alignment) for the selected export pricing mode.

    Fixed price -> (None, ...). Market price -> an uploaded / data/ ``export_prices``
    file (matched by exact time when it covers the consumption dates), otherwise the
    rolling last-year cache, always matched by month/day/time so any consumption year
    is compared against the latest 365 days of prices.
    """
    if params.get("export_price_mode", EXPORT_MODE_FIXED) != EXPORT_MODE_MARKET:
        return None, ALIGN_AUTO
    path = params.get("export_price_file") or _find_data_file(EXPORT_PRICES_NAME)
    if Path(path).exists():
        return path, ALIGN_AUTO
    cached = cached_price_file(DEFAULT_CACHE_DIR)
    if cached is None:  # startup refresh not finished (or offline): try once now
        try:
            cached = market_price_file(DEFAULT_CACHE_DIR)
        except RuntimeError as exc:
            raise FileNotFoundError(f"Market export pricing: {exc}") from exc
    return cached, ALIGN_CALENDAR


# Rolling 1-year price cache: refreshed at startup and then every few hours, so new
# day-ahead prices (published ~13:00 for the next day) are picked up and old ones dropped.
PRICE_CACHE_REFRESH_SECONDS = 6 * 3600
_price_cache_status = {"text": "price cache: not updated yet"}


def _price_cache_status_text(summary: dict) -> str:
    if "path" not in summary:
        return f"price cache: unavailable ({'; '.join(summary['errors'])})"
    text = (
        f"1-year price cache: {summary['start']} \u2192 {summary['end']} ({summary['days']} days)"
    )
    if summary["errors"]:
        text += f" \u2014 {'; '.join(summary['errors'])}"
    return text


async def _price_cache_loop() -> None:
    while True:
        try:
            summary = await asyncio.to_thread(update_price_cache, DEFAULT_CACHE_DIR)
            _price_cache_status["text"] = _price_cache_status_text(summary)
        except Exception as exc:  # never let the refresher take the app down
            _price_cache_status["text"] = f"price cache: update failed ({exc})"
        print(_price_cache_status["text"])
        await asyncio.sleep(PRICE_CACHE_REFRESH_SECONDS)


async def _start_price_cache() -> None:
    if os.environ.get("PRICE_CACHE_AUTO_UPDATE", "1") != "0":
        app.state.price_cache_task = asyncio.create_task(_price_cache_loop())


app.on_startup(_start_price_cache)


def make_transmission_costs(b1, b2, b3, b4, b5):
    return {"block1": b1, "block2": b2, "block3": b3, "block4": b4, "block5": b5}


def make_monthly_power_fees(b1, b2, b3, b4, b5):
    return {"block1": b1, "block2": b2, "block3": b3, "block4": b4, "block5": b5}


TARIFF_MANUAL = "manual"


def _tariff_file() -> Path:
    """Network tariff file: upload / data/ / cwd, else the copy shipped next to the code."""
    found = Path(_find_data_file(NETWORK_TARIFF_FILE))
    if found.exists():
        return found
    return Path(__file__).resolve().parent / NETWORK_TARIFF_FILE


def _load_tariffs() -> tuple[dict, str | None]:
    """(tariffs by user group, error text); empty tariffs when the file is missing/invalid."""
    path = _tariff_file()
    try:
        return load_network_tariffs(path), None
    except (OSError, ValueError) as exc:
        return {}, f"{path.name}: {exc}"


def _tariff_options(tariffs: dict) -> dict[str, str]:
    options = {group: f"User group {group}" for group in tariffs}
    options[TARIFF_MANUAL] = "Manual"
    return options


def _default_tariff_group(tariffs: dict) -> str:
    if DEFAULT_TARIFF_GROUP in tariffs:
        return DEFAULT_TARIFF_GROUP
    return next(iter(tariffs), TARIFF_MANUAL)


def resolve_network_tariffs(params: dict) -> tuple[dict, dict]:
    """(transmission_costs, monthly_power_fees) for the selected source.

    A user group is read from the tariff file at run time; ``"manual"`` (or no
    selection) uses the tb1-5 / pf1-5 inputs.
    """
    group = params.get("tariff_group", TARIFF_MANUAL)
    if group and group != TARIFF_MANUAL:
        tariffs = load_network_tariffs(_tariff_file())
        if group not in tariffs:
            raise ValueError(f"User group {group} not found in {NETWORK_TARIFF_FILE}")
        return (
            dict(tariffs[group]["transmission_costs"]),
            dict(tariffs[group]["monthly_power_fees"]),
        )
    return (
        make_transmission_costs(*(params[f"tb{i}"] for i in range(1, 6))),
        make_monthly_power_fees(*(params[f"pf{i}"] for i in range(1, 6))),
    )


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return buf.read()


async def run_single_simulation(params: dict, results_container, spinner):
    """Run a single scenario simulation in a background thread."""
    spinner.visible = True
    results_container.clear()

    def _run():
        heating_config = None
        if params["add_heating_load"]:
            heating_config = {
                "heating_kwh": params["heating_kwh"],
                "start_hour": params["heating_start_hour"],
                "end_hour": params["heating_end_hour"],
            }

        max_power_by_block = None
        if params["enable_power_smoothing"]:
            max_power_by_block = {
                1: params["max_power_block1"],
                2: params["max_power_block2"],
                3: params["max_power_block3"],
                4: params["max_power_block4"],
                5: params["max_power_block5"],
            }

        export_price_file, export_price_alignment = _export_prices(params)
        transmission_costs, monthly_power_fees = resolve_network_tariffs(params)
        sim = PowerFlowSimulator(
            solar_panel_power_kw=params["solar_power"],
            inverter_power_kw=params["inverter_power"],
            battery_capacity_kwh=params["battery_capacity"],
            battery_c_rate=params["battery_c_rate"],
            battery_efficiency=params["battery_efficiency"],
            peak_price=params["peak_price"],
            off_peak_price=params["off_peak_price"],
            transmission_costs=transmission_costs,
            monthly_power_fees=monthly_power_fees,
            ove_spte_fee=params["ove_spte_fee"],
            enable_power_smoothing=params["enable_power_smoothing"],
            min_soc_reserve=params["min_soc_reserve"],
            max_power_by_block=max_power_by_block,
            heating_config=heating_config,
            production_file=params.get("production_file", _find_data_file("production.csv")),
            consumption_file=params.get("consumption_file", _find_data_file("consumption.csv")),
            export_price_file=export_price_file,
            export_price_alignment=export_price_alignment,
            export_fee_eur_mwh=params.get("export_fee", 0.0),
            reference_solar_kw=params.get("reference_solar_kw") or DEFAULT_REFERENCE_SOLAR_KW,
        )

        sim.load_and_align_data()
        sim.scale_solar_generation()
        sim.simulate_power_flows()

        cost_analysis = sim.calculate_costs_and_savings(
            peak_price=params["peak_price"],
            off_peak_price=params["off_peak_price"],
            export_price=params["export_price"],
        )

        transmission_breakdown = sim.calculate_transmission_cost_breakdown()
        ove_spte = sim.calculate_ove_spte_cost(sim.max_power_by_block_simulated)

        # Build power flow figure
        r = sim.simulation_results
        days = 7
        kw_factor = sim.kw_factor
        intervals_per_hour = int(1 / sim.interval_hours)
        plot_data = r.head(days * 24 * intervals_per_hour).copy()
        idx = range(len(plot_data))

        with plt.rc_context(_GUI_PLOT_RC):
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9))
            ax1.plot(
                idx,
                plot_data["solar_generation_kwh"] * kw_factor,
                label="Solar",
                color="orange",
                lw=1.5,
            )
            ax1.plot(
                idx,
                plot_data["consumption_kwh"] * kw_factor,
                label="Consumption",
                color="blue",
                lw=1.5,
            )
            ax1.fill_between(
                idx, 0, plot_data["solar_generation_kwh"] * kw_factor, alpha=0.2, color="orange"
            )
            ax1.set_ylabel("Power (kW)")
            ax1.set_title(f"Power Flows — First {days} Days")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(idx, plot_data["battery_soc_percent"], color="green", lw=1.5)
            ax2.fill_between(idx, 0, plot_data["battery_soc_percent"], alpha=0.2, color="green")
            ax2.set_ylabel("Battery SOC (%)")
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)

            ax3.plot(
                idx, plot_data["grid_import_kwh"] * kw_factor, label="Import", color="red", lw=1.5
            )
            ax3.plot(
                idx,
                -plot_data["grid_export_kwh"] * kw_factor,
                label="Export",
                color="purple",
                lw=1.5,
            )
            ax3.fill_between(
                idx, 0, plot_data["grid_import_kwh"] * kw_factor, alpha=0.2, color="red"
            )
            ax3.fill_between(
                idx, 0, -plot_data["grid_export_kwh"] * kw_factor, alpha=0.2, color="purple"
            )
            ax3.set_ylabel("Power (kW)")
            ax3.set_xlabel("15-min intervals")
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            fig.tight_layout()
            plot_bytes = fig_to_png_bytes(fig)
        # Summary stats
        total_consumption = r["consumption_kwh"].sum()
        total_solar = r["solar_generation_kwh"].sum()
        total_import = r["grid_import_kwh"].sum()
        total_export = r["grid_export_kwh"].sum()
        self_sufficiency = (
            (1 - total_import / total_consumption) * 100 if total_consumption > 0 else 0
        )
        avg_soc = r["battery_soc_percent"].mean()
        cycles = (
            (r["battery_charge_kwh"].sum() + r["battery_discharge_kwh"].sum())
            / (2 * params["battery_capacity"])
            if params["battery_capacity"] > 0
            else 0
        )

        return {
            "cost_analysis": cost_analysis,
            "transmission_breakdown": transmission_breakdown,
            "ove_spte": ove_spte,
            "plot_bytes": plot_bytes,
            "market_export": sim.export_price_file is not None,
            "total_consumption": total_consumption,
            "total_solar": total_solar,
            "total_import": total_import,
            "total_export": total_export,
            "self_sufficiency": self_sufficiency,
            "avg_soc": avg_soc,
            "cycles": cycles,
        }

    try:
        result = await asyncio.to_thread(_run)
    except Exception as e:
        spinner.visible = False
        with results_container:
            ui.notify(f"Simulation failed: {e}", type="negative")
        return

    spinner.visible = False
    ca = result["cost_analysis"]

    with results_container:
        ui.label("Results").classes("text-h5 q-mt-md")

        # Summary cards
        with ui.row().classes("q-gutter-md q-mt-sm"):
            _metric_card("Baseline Cost", f"\u20ac{ca['baseline_cost']:,.0f}/yr")
            _metric_card("Solar+Battery Cost", f"\u20ac{ca['solar_battery_cost']:,.0f}/yr")
            _metric_card("Annual Savings", f"\u20ac{ca['savings_vs_baseline']:,.0f}/yr")
            _metric_card("Self-Sufficiency", f"{result['self_sufficiency']:.1f}%")

        with ui.row().classes("q-gutter-md q-mt-sm"):
            _metric_card("Solar Generation", f"{result['total_solar']:,.0f} kWh/yr")
            _metric_card("Grid Import", f"{result['total_import']:,.0f} kWh/yr")
            _metric_card("Grid Export", f"{result['total_export']:,.0f} kWh/yr")
            _metric_card("Export Revenue", f"\u20ac{ca['export_revenue']:,.0f}/yr")
            _metric_card(
                "Avg Export Price" + (" (market)" if result["market_export"] else ""),
                f"\u20ac{ca['avg_export_price']:.4f}/kWh",
            )
            _metric_card("Avg SOC", f"{result['avg_soc']:.1f}%")
            _metric_card("Battery Cycles", f"{result['cycles']:.1f}/yr")

        # Power flow plot
        ui.label("Power Flow Visualization").classes("text-h6 q-mt-lg")
        ui.image(f"data:image/png;base64,{_b64(result['plot_bytes'])}").classes("w-full")

        # Transmission breakdown table
        tb = result["transmission_breakdown"]
        if tb:
            ui.label("Transmission Cost Breakdown").classes("text-h6 q-mt-lg")
            rows = []
            for block_name, data in tb.items():
                rows.append(
                    {
                        "Block": block_name.replace("_", " ").title(),
                        "Intervals": data["intervals"],
                        "% of Year": f"{data['percentage_of_year']:.1f}",
                        "Import (kWh)": f"{data['total_import_kwh']:.1f}",
                        "Rate (\u20ac/kWh)": f"{data['transmission_rate_eur_per_kwh']:.5f}",
                        "Trans. Cost (\u20ac)": f"{data['transmission_cost_eur']:.2f}",
                        "Max Power (kW)": f"{data['max_import_power_kw']:.1f}",
                        "Annual Fee (\u20ac)": f"{data['annual_power_fee_eur']:.2f}",
                    }
                )
            columns = [{"name": k, "label": k, "field": k, "sortable": True} for k in rows[0]]
            ui.table(columns=columns, rows=rows).classes("q-mt-sm")


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode()


def _metric_card(label: str, value: str):
    with ui.card().classes("p-4 text-center min-w-[140px]"):
        ui.label(value).classes("text-h6 text-weight-bold")
        ui.label(label).classes("text-caption text-grey-7")


class _BatchStopped(Exception):
    """Raised inside the batch worker when the user pressed Stop Analysis."""


async def run_batch_analysis(
    params: dict, results_container, spinner, stop_event: threading.Event | None = None
):
    """Run batch analysis in a background thread with progress tracking.

    Setting ``stop_event`` ends the run after the scenario currently simulating.
    """
    stop_event = stop_event or threading.Event()
    spinner.visible = True
    results_container.clear()

    # Progress tracking — shared between background thread and UI timer
    progress = {"current": 0, "total": 0, "phase": "Initializing..."}

    with results_container:
        progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full q-mt-sm")
        progress_label = ui.label("Initializing...").classes("text-caption")

    def _update_progress():
        total = progress["total"]
        analyzer = progress.get("analyzer")
        current = len(analyzer.results) if analyzer and hasattr(analyzer, "results") else 0
        if total > 0:
            progress_bar.set_value(current / total)
            progress_label.set_text(f"{progress['phase']}: {current}/{total} scenarios")
        else:
            progress_label.set_text(progress["phase"])

    timer = ui.timer(0.5, _update_progress)

    def _run():
        def parse_range(s):
            return [float(x.strip()) for x in s.split(",") if x.strip()]

        heating_config = None
        if params["add_heating_load"]:
            heating_config = {
                "heating_kwh": params["heating_kwh"],
                "start_hour": params["heating_start_hour"],
                "end_hour": params["heating_end_hour"],
            }

        max_power_by_block = None
        if params["enable_power_smoothing"]:
            max_power_by_block = {
                1: params["max_power_block1"],
                2: params["max_power_block2"],
                3: params["max_power_block3"],
                4: params["max_power_block4"],
                5: params["max_power_block5"],
            }

        export_price_file, export_price_alignment = _export_prices(params)
        transmission_costs, monthly_power_fees = resolve_network_tariffs(params)
        analyzer = MultiScenarioAnalyzer(
            solar_range=parse_range(params["solar_range"]),
            inverter_range=parse_range(params["inverter_range"]),
            battery_range=parse_range(params["battery_range"]),
            battery_c_rate=params["battery_c_rate"],
            battery_efficiency=params["battery_efficiency"],
            peak_price=params["peak_price"],
            off_peak_price=params["off_peak_price"],
            export_price=params["export_price"],
            solar_cost_per_kw=params["solar_cost_per_kw"],
            inverter_cost_per_kw=params["inverter_cost_per_kw"],
            battery_cost_per_kwh=params["battery_cost_per_kwh"],
            maintenance_fee_per_kw=params["maintenance_fee_per_kw"],
            battery_maintenance_fee_per_kwh=params["battery_maintenance_fee_per_kwh"],
            discount_rate=params["discount_rate"],
            loan_rate=params["loan_rate"],
            loan_years=_loan_years(params),
            subsidy=_subsidy_config(params),
            transmission_costs=transmission_costs,
            monthly_power_fees=monthly_power_fees,
            ove_spte_fee=params["ove_spte_fee"],
            enable_power_smoothing=params["enable_power_smoothing"],
            min_soc_reserve=params["min_soc_reserve"],
            max_power_by_block=max_power_by_block,
            heating_config=heating_config,
            production_file=params.get("production_file", _find_data_file("production.csv")),
            consumption_file=params.get("consumption_file", _find_data_file("consumption.csv")),
            export_price_file=export_price_file,
            export_price_alignment=export_price_alignment,
            export_fee_eur_mwh=params.get("export_fee", 0.0),
            reference_solar_kw=params.get("reference_solar_kw") or DEFAULT_REFERENCE_SOLAR_KW,
        )

        progress["phase"] = "Generating scenarios"
        analyzer.generate_scenarios()
        progress["total"] = len(analyzer.scenarios)
        progress["phase"] = "Running simulations"
        # Store analyzer ref so the timer can poll len(analyzer.results)
        progress["analyzer"] = analyzer

        analyzer.run_all_scenarios(should_stop=stop_event.is_set)
        if stop_event.is_set():
            raise _BatchStopped

        progress["current"] = progress["total"]
        progress["phase"] = "Generating plots"
        analyzer.create_comparison_summary(output_path=None)

        # Build heatmap figures in memory instead of saving to files
        plots = {}
        df = pd.DataFrame(analyzer.results)

        import seaborn as sns

        with plt.rc_context(_GUI_PLOT_RC):
            # Savings heatmap
            pivot = df.pivot_table(
                values="savings_vs_baseline",
                index="battery_capacity_kwh",
                columns="solar_panel_power_kw",
                aggfunc="max",
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".0f",
                cmap="viridis",
                cbar_kws={"label": "Annual Savings (\u20ac)"},
                ax=ax,
            )
            ax.set_title("Annual Savings: Solar vs Battery")
            ax.set_xlabel("Solar (kW)")
            ax.set_ylabel("Battery (kWh)")
            fig.tight_layout()
            plots["savings"] = fig_to_png_bytes(fig)

            # ROI heatmap
            df["payback_capped"] = df["payback_years"].apply(lambda x: min(x, 20))
            pivot = df.pivot_table(
                values="payback_capped",
                index="battery_capacity_kwh",
                columns="solar_panel_power_kw",
                aggfunc="min",
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1f",
                cmap="RdYlGn_r",
                vmin=0,
                vmax=20,
                cbar_kws={"label": "Payback (years)"},
                ax=ax,
            )
            ax.set_title("ROI Payback Period: Solar vs Battery")
            ax.set_xlabel("Solar (kW)")
            ax.set_ylabel("Battery (kWh)")
            fig.tight_layout()
            plots["roi"] = fig_to_png_bytes(fig)

            # Self-sufficiency heatmap
            pivot = df.pivot_table(
                values="self_sufficiency_percent",
                index="battery_capacity_kwh",
                columns="solar_panel_power_kw",
                aggfunc="max",
            )
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".1f",
                cmap="YlGn",
                cbar_kws={"label": "Self-Sufficiency (%)"},
                ax=ax,
            )
            ax.set_title("Self-Sufficiency: Solar vs Battery")
            ax.set_xlabel("Solar (kW)")
            ax.set_ylabel("Battery (kWh)")
            fig.tight_layout()
            plots["self_sufficiency"] = fig_to_png_bytes(fig)

            # NPV heatmap
            if "npv_20_years" in df.columns:
                pivot = df.pivot_table(
                    values="npv_20_years",
                    index="battery_capacity_kwh",
                    columns="solar_panel_power_kw",
                    aggfunc="max",
                )
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    pivot,
                    annot=True,
                    fmt=".0f",
                    cmap="RdYlGn",
                    cbar_kws={"label": "NPV 20yr (\u20ac)"},
                    ax=ax,
                )
                ax.set_title("20-Year NPV: Solar vs Battery")
                ax.set_xlabel("Solar (kW)")
                ax.set_ylabel("Battery (kWh)")
                fig.tight_layout()
                plots["npv"] = fig_to_png_bytes(fig)

        comparison_df = comparison_dataframe(analyzer.results)
        return {
            "plots": plots,
            "comparison_df": comparison_df,
            "table_rows": comparison_table_rows(comparison_df),
            "n_scenarios": len(analyzer.results),
        }

    try:
        result = await asyncio.to_thread(_run)
    except _BatchStopped:
        timer.deactivate()
        spinner.visible = False
        results_container.clear()
        with results_container:
            ui.notify("Batch analysis stopped", type="warning")
        return
    except Exception as e:
        timer.deactivate()
        spinner.visible = False
        results_container.clear()
        with results_container:
            ui.notify(f"Batch analysis failed: {e}", type="negative")
        return

    timer.deactivate()
    spinner.visible = False
    results_container.clear()

    with results_container:
        ui.label(f"Batch Results — {result['n_scenarios']} scenarios").classes("text-h5 q-mt-md")

        # Heatmaps
        with ui.tabs().classes("q-mt-md") as tabs:
            savings_tab = ui.tab("Savings")
            roi_tab = ui.tab("ROI")
            ss_tab = ui.tab("Self-Sufficiency")
            if "npv" in result["plots"]:
                npv_tab = ui.tab("NPV")

        def _plot_img(plot_bytes):
            b64 = _b64(plot_bytes)
            ui.html(f'<img src="data:image/png;base64,{b64}" style="width:100%;height:auto">')

        with ui.tab_panels(tabs, value=savings_tab).classes("w-full"):
            with ui.tab_panel(savings_tab):
                _plot_img(result["plots"]["savings"])
            with ui.tab_panel(roi_tab):
                _plot_img(result["plots"]["roi"])
            with ui.tab_panel(ss_tab):
                _plot_img(result["plots"]["self_sufficiency"])
            if "npv" in result["plots"]:
                with ui.tab_panel(npv_tab):
                    _plot_img(result["plots"]["npv"])

        # Comparison table
        ui.label("Scenario Comparison").classes("text-h6 q-mt-lg")
        if result["table_rows"]:
            columns = [
                {"name": k, "label": k, "field": k, "sortable": True}
                for k in result["table_rows"][0]
            ]
            comparison_df = result["comparison_df"]
            with ui.row().classes("q-mt-sm items-center"):
                ui.button(
                    "Download Excel",
                    icon="download",
                    on_click=lambda: ui.download.content(
                        comparison_xlsx_bytes(comparison_df), "scenario_comparison.xlsx"
                    ),
                ).props("outline")
                ui.button(
                    "Download CSV",
                    icon="download",
                    on_click=lambda: ui.download.content(
                        comparison_csv_bytes(comparison_df), "scenario_comparison.csv"
                    ),
                ).props("outline")
            ui.table(columns=columns, rows=result["table_rows"]).classes("q-mt-sm")


# ---------------------------------------------------------------------------
# Scenario comparison table + Slovenian-locale exports
# ---------------------------------------------------------------------------
# (column label, result key, decimals shown in the GUI table / Excel number format)
COMPARISON_COLUMNS: list[tuple[str, str, int]] = [
    ("Solar (kW)", "solar_panel_power_kw", 1),
    ("Inverter (kW)", "inverter_power_kw", 1),
    ("Battery (kWh)", "battery_capacity_kwh", 1),
    ("Savings (\u20ac/yr)", "savings_vs_baseline", 0),
    ("Payback (yr)", "payback_years", 1),
    ("NPV 20yr (\u20ac)", "npv_20_years", 0),
    ("Self-Suff (%)", "self_sufficiency_percent", 1),
    ("Investment (\u20ac)", "total_investment", 0),
    ("Subsidy (\u20ac)", "subsidy", 0),
    ("Solar produced (kWh)", "annual_solar_generation", 0),
    ("Exported (kWh)", "annual_grid_export", 0),
    ("Imported (kWh)", "annual_grid_import", 0),
    ("Consumed (kWh)", "annual_consumption", 0),
    ("Variable cost (\u20ac/yr)", "variable_energy_cost", 0),
    ("Fixed cost (\u20ac/yr)", "fixed_cost", 0),
]


def comparison_dataframe(results: list[dict]) -> pd.DataFrame:
    """Numeric scenario comparison, best savings first.

    Variable cost = imported kWh x (energy price + transmission block rate); fixed cost =
    monthly power fees + OVE-SPTE, both for the solar+battery system.
    """
    rows = sorted(results, key=lambda r: r["savings_vs_baseline"], reverse=True)
    return pd.DataFrame(
        [{label: float(r.get(key, 0.0)) for label, key, _ in COMPARISON_COLUMNS} for r in rows],
        columns=[label for label, _, _ in COMPARISON_COLUMNS],
    )


def comparison_table_rows(df: pd.DataFrame) -> list[dict]:
    """Display rows for ``ui.table`` (thousands separators, fixed decimals)."""
    rows = []
    for record in df.to_dict("records"):
        row = {}
        for label, _, decimals in COMPARISON_COLUMNS:
            value = record[label]
            if label in ("Solar (kW)", "Inverter (kW)", "Battery (kWh)"):
                row[label] = value
            else:
                row[label] = f"{value:,.{decimals}f}"
        rows.append(row)
    return rows


def comparison_csv_bytes(df: pd.DataFrame) -> bytes:
    """CSV in Slovenian locale: ``;`` delimiter, decimal comma, UTF-8 BOM for Excel."""
    rounded = df.round({label: max(d, 2) for label, _, d in COMPARISON_COLUMNS})
    text = rounded.to_csv(sep=";", decimal=",", index=False, lineterminator="\r\n")
    return text.encode("utf-8-sig")


# Slovenian locale tag (LCID 0x424): LibreOffice renders tagged formats as ``1.234,5`` / ``45,3 %``
# whatever the viewer's locale; Excel takes separators from the OS, so Slovenian Excel matches.
XLSX_LOCALE = "[$-424]"
XLSX_PERCENT_COLUMNS = {"Self-Suff (%)"}


def xlsx_number_format(label: str, decimals: int) -> str:
    """Slovenian-tagged Excel number format: ``.`` thousands, ``,`` decimals, ``45,3 %``."""
    frac = "." + "0" * decimals if decimals else ""
    if label in XLSX_PERCENT_COLUMNS:
        return f"{XLSX_LOCALE}0{frac}\\ %"
    return f"{XLSX_LOCALE}#,##0{frac}"


def comparison_xlsx_bytes(df: pd.DataFrame) -> bytes:
    """Excel workbook in Slovenian number formatting, with real numbers in the cells.

    Values stay numeric (sortable, usable in formulas); percentages are stored as fractions
    (0.453) in a ``0,0 %`` cell so they display as ``45,3 %``.
    """
    from openpyxl.utils import get_column_letter

    out = df.copy()
    for label in XLSX_PERCENT_COLUMNS:
        out[label] = out[label] / 100.0

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="Scenarios")
        writer.book.properties.language = "sl-SI"
        ws = writer.sheets["Scenarios"]
        ws.freeze_panes = "A2"
        for idx, (label, _, decimals) in enumerate(COMPARISON_COLUMNS, start=1):
            fmt = xlsx_number_format(label, decimals)
            letter = get_column_letter(idx)
            for (cell,) in ws.iter_rows(min_row=2, min_col=idx, max_col=idx):
                cell.number_format = fmt
            ws.column_dimensions[letter].width = max(12, len(label) + 2)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Shared parameter state with defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "solar_power": 8.5,
    "inverter_power": 8.0,
    "battery_capacity": 10.0,
    "battery_c_rate": 0.5,
    "battery_efficiency": 0.9,
    "peak_price": 0.14683,
    "off_peak_price": 0.10664,
    "export_price": 0.01,
    "export_fee": 0.0,
    "reference_solar_kw": DEFAULT_REFERENCE_SOLAR_KW,
    "export_price_mode": EXPORT_MODE_FIXED,
    "tb1": 0.01282,
    "tb2": 0.01216,
    "tb3": 0.01186,
    "tb4": 0.01164,
    "tb5": 0.01175,
    "pf1": 3.75969,
    "pf2": 1.05262,
    "pf3": 0.12837,
    "pf4": 0.0,
    "pf5": 0.0,
    "ove_spte_fee": 3.44078,
    "enable_power_smoothing": False,
    "min_soc_reserve": 0.2,
    "max_power_block1": 250.0,
    "max_power_block2": 250.0,
    "max_power_block3": 340.0,
    "max_power_block4": 2000.0,
    "max_power_block5": 2000.0,
    "add_heating_load": False,
    "heating_kwh": 3333.0,
    "heating_start_hour": 7,
    "heating_end_hour": 0,
    # Batch-specific
    "solar_range": "5,8.5,10,15,20",
    "inverter_range": "5,8,10,15",
    "battery_range": "0,10,15,20",
    "solar_cost_per_kw": 450.0,
    "inverter_cost_per_kw": 130.0,
    "battery_cost_per_kwh": 250.0,
    "maintenance_fee_per_kw": 10.0,
    "battery_maintenance_fee_per_kwh": 5.0,
    "discount_rate": 0.05,
    "use_loan": True,
    "loan_rate": 0.03,
    "loan_years": 10,
    # One-off investment subsidies (batch)
    "use_subsidies": False,
    "subsidy_solar_enabled": True,
    "subsidy_solar_per_kw": 0.0,
    "subsidy_inverter_enabled": False,
    "subsidy_inverter_per_kw": 0.0,
    "subsidy_battery_enabled": True,
    "subsidy_battery_per_kwh": 0.0,
    "subsidy_battery_min_ratio": 0.5,
    "subsidy_battery_max_ratio": 2.5,
    "subsidy_battery_basis": "solar",
    "subsidy_cap_enabled": False,
    "subsidy_cap_fraction": 0.5,
}

SUBSIDY_BASIS_LABELS = {
    "solar": "Solar kW",
    "inverter": "Inverter kW",
    "sum": "Solar + inverter kW",
}
assert set(SUBSIDY_BASIS_LABELS) == set(BATTERY_SUBSIDY_BASES)


def _subsidy_config(params: dict) -> SubsidyConfig:
    """Subsidies for the analyzer; all off unless the "Apply subsidies" switch is on."""
    if not params.get("use_subsidies"):
        return SubsidyConfig()

    def num(key: str) -> float:
        value = params.get(key, DEFAULTS[key])
        return float(DEFAULTS[key] if value is None else value)

    return SubsidyConfig(
        solar_enabled=bool(params.get("subsidy_solar_enabled")),
        solar_eur_per_kw=num("subsidy_solar_per_kw"),
        inverter_enabled=bool(params.get("subsidy_inverter_enabled")),
        inverter_eur_per_kw=num("subsidy_inverter_per_kw"),
        battery_enabled=bool(params.get("subsidy_battery_enabled")),
        battery_eur_per_kwh=num("subsidy_battery_per_kwh"),
        battery_min_ratio=num("subsidy_battery_min_ratio"),
        battery_max_ratio=num("subsidy_battery_max_ratio"),
        battery_basis=params.get("subsidy_battery_basis") or DEFAULTS["subsidy_battery_basis"],
        cap_enabled=bool(params.get("subsidy_cap_enabled")),
        cap_fraction=num("subsidy_cap_fraction"),
    )


def _loan_years(params: dict) -> int:
    """Loan term for the analyzer; 0 (cash purchase) when the loan is off or the term is < 1."""
    if not params.get("use_loan", True):
        return 0
    return max(int(params.get("loan_years") or 0), 0)


def build_shared_params(inputs: dict) -> dict:
    """Collect current values from all input widgets."""
    params = {}
    for key, widget in inputs.items():
        params[key] = widget.value
    return params


def _number(label, key, inputs, *, step=None, suffix="", **kwargs):
    """Create a number input and register it."""
    n = ui.number(label=label, value=DEFAULTS[key], step=step, suffix=suffix, **kwargs)
    inputs[key] = n
    return n


def _pricing_section(inputs):
    with ui.expansion("Electricity Pricing", icon="payments").classes("w-full"):
        with ui.row().classes("q-gutter-sm"):
            _number("Peak price", "peak_price", inputs, step=0.001, suffix="\u20ac/kWh")
            _number("Off-peak price", "off_peak_price", inputs, step=0.001, suffix="\u20ac/kWh")
        mode = ui.select(
            options=[EXPORT_MODE_FIXED, EXPORT_MODE_MARKET],
            label="Export pricing",
            value=DEFAULTS["export_price_mode"],
        ).classes("w-full")
        inputs["export_price_mode"] = mode
        fixed = _number("Export price", "export_price", inputs, step=0.001, suffix="\u20ac/kWh")
        fixed.bind_visibility_from(mode, "value", value=EXPORT_MODE_FIXED)
        _number("Export fee", "export_fee", inputs, step=0.5, suffix="\u20ac/MWh").tooltip(
            "Deducted from every exported MWh (fixed or market price)"
        )
        with ui.column().classes("w-full gap-1") as market:
            found = _find_data_file(EXPORT_PRICES_NAME)
            ui.label(
                "SI day-ahead prices of the last 365 days (rolling cache, Energy-Charts, "
                "CC BY 4.0), matched to the consumption by month/day/time whatever its "
                "year. An uploaded HUPX download / timestamp,price file overrides it."
            ).classes("text-caption")
            ui.label().classes("text-caption text-positive").bind_text_from(
                _price_cache_status, "text"
            )
            with ui.row().classes("w-full q-gutter-sm items-center"):
                label = ui.label(
                    f"export_prices: {found if Path(found).exists() else 'missing'}"
                ).classes("text-caption")
                ui.upload(
                    label="Upload",
                    auto_upload=True,
                    on_upload=lambda e: _handle_upload(e, EXPORT_PRICES_NAME, label),
                ).props(f"accept={_ACCEPT} flat dense").classes("max-w-[180px]")
        market.bind_visibility_from(mode, "value", value=EXPORT_MODE_MARKET)


def _transmission_section(inputs):
    tariffs, error = _load_tariffs()
    with ui.expansion("Transmission Costs", icon="electrical_services").classes("w-full"):
        source = ui.select(
            options=_tariff_options(tariffs),
            label="Network tariff",
            value=_default_tariff_group(tariffs),
        ).classes("w-full")
        source.tooltip(f"User group rates from {NETWORK_TARIFF_FILE}, or enter them manually")
        inputs["tariff_group"] = source
        status = ui.label(error or f"{NETWORK_TARIFF_FILE}: {len(tariffs)} user groups")
        status.classes("text-caption " + ("text-warning" if error else "text-positive"))

        ui.label("Transmission costs (\u20ac/kWh)").classes("text-caption")
        with ui.row().classes("q-gutter-sm"):
            for i in range(1, 6):
                _number(f"Block {i}", f"tb{i}", inputs, step=0.0001)
        ui.separator()
        ui.label("Monthly power fees (\u20ac/kW/month)").classes("text-caption")
        with ui.row().classes("q-gutter-sm"):
            for i in range(1, 6):
                _number(f"Block {i}", f"pf{i}", inputs, step=0.01)

        rate_inputs = [inputs[f"tb{i}"] for i in range(1, 6)] + [
            inputs[f"pf{i}"] for i in range(1, 6)
        ]

        def _apply(group) -> None:
            manual = group == TARIFF_MANUAL or group not in tariffs
            if not manual:
                rates = tariffs[group]
                for i in range(1, 6):
                    inputs[f"tb{i}"].value = rates["transmission_costs"][f"block{i}"]
                    inputs[f"pf{i}"].value = rates["monthly_power_fees"][f"block{i}"]
            for widget in rate_inputs:
                if manual:
                    widget.props(remove="readonly")
                else:
                    widget.props("readonly")

        async def _on_upload(event) -> None:
            if not await _handle_upload(event, NETWORK_TARIFF_FILE, status):
                return
            new, err = _load_tariffs()
            tariffs.clear()
            tariffs.update(new)
            status.text = err or f"{NETWORK_TARIFF_FILE}: {len(tariffs)} user groups"
            source.set_options(
                _tariff_options(tariffs),
                value=source.value if source.value in tariffs else _default_tariff_group(tariffs),
            )
            _apply(source.value)

        ui.upload(
            label="Replace tariff file",
            auto_upload=True,
            on_upload=_on_upload,
        ).props(f"accept={_ACCEPT} flat dense").classes("max-w-[220px]")
        source.on_value_change(lambda e: _apply(e.value))
        _apply(source.value)

        ui.separator()
        _number("OVE-SPTE fee", "ove_spte_fee", inputs, step=0.01, suffix="\u20ac/kW/mo")


def _smoothing_section(inputs):
    with ui.expansion("Power Smoothing", icon="tune").classes("w-full"):
        sw = ui.switch("Enable power smoothing", value=DEFAULTS["enable_power_smoothing"])
        inputs["enable_power_smoothing"] = sw
        _number("Min SOC reserve", "min_soc_reserve", inputs, step=0.05)
        ui.label("Block power thresholds (kW)").classes("text-caption")
        with ui.row().classes("q-gutter-sm"):
            for i in range(1, 6):
                _number(f"Block {i}", f"max_power_block{i}", inputs, step=10)


def _heating_section(inputs):
    with ui.expansion("Heating Load", icon="whatshot").classes("w-full"):
        sw = ui.switch("Add heating load (Oct\u2013Mar)", value=DEFAULTS["add_heating_load"])
        inputs["add_heating_load"] = sw
        _number("Heating energy", "heating_kwh", inputs, step=100, suffix="kWh/season")
        with ui.row().classes("q-gutter-sm"):
            _number("Start hour", "heating_start_hour", inputs, step=1)
            _number("End hour (0=midnight)", "heating_end_hour", inputs, step=1)


SINGLE_TAB = "Single Scenario"
BATCH_TAB = "Batch Analysis"


def _tab_name(value) -> str | None:
    """Name of the selected tab (``ui.tabs.value`` may be a tab element or its name)."""
    return value.props["name"] if isinstance(value, ui.tab | ui.tab_panel) else value


def _only_on(element, tabs, name: str):
    """Show ``element`` only while the tab called ``name`` is selected."""
    return element.bind_visibility_from(tabs, "value", backward=lambda v: _tab_name(v) == name)


def _system_specs_section(inputs):
    with (
        ui.expansion("System Specs", icon="solar_power", value=True)
        .classes("w-full")
        .props("dense") as section
    ):
        with ui.grid(columns=1).classes("w-full gap-1"):
            _number("Solar", "solar_power", inputs, step=0.5, suffix="kW")
            _number("Inverter", "inverter_power", inputs, step=0.5, suffix="kW")
            _number("Battery", "battery_capacity", inputs, step=1, suffix="kWh")
            _number("C-rate", "battery_c_rate", inputs, step=0.1)
            _number("Efficiency", "battery_efficiency", inputs, step=0.05)
    return section


def _parameter_ranges_section(inputs):
    with (
        ui.expansion("Parameter Ranges", icon="grid_view", value=True)
        .classes("w-full")
        .props("dense") as section
    ):
        inputs["solar_range"] = ui.input(
            label="Solar range (kW)", value=DEFAULTS["solar_range"]
        ).classes("w-full")
        inputs["inverter_range"] = ui.input(
            label="Inverter range (kW)", value=DEFAULTS["inverter_range"]
        ).classes("w-full")
        inputs["battery_range"] = ui.input(
            label="Battery range (kWh)", value=DEFAULTS["battery_range"]
        ).classes("w-full")
    return section


def _battery_section(inputs):
    """Batch copy of the System Specs battery inputs, two-way bound so both show one value."""
    with (
        ui.expansion("Battery", icon="battery_charging_full")
        .classes("w-full")
        .props("dense") as section
    ):
        with ui.grid(columns=2).classes("w-full gap-1"):
            for label, key, step in (
                ("C-rate", "battery_c_rate", 0.1),
                ("Efficiency", "battery_efficiency", 0.05),
            ):
                ui.number(label=label, value=DEFAULTS[key], step=step).bind_value(
                    inputs[key], "value"
                )
    return section


def _equipment_costs_section(inputs):
    with (
        ui.expansion("Equipment Costs", icon="euro", value=True)
        .classes("w-full")
        .props("dense") as section
    ):
        with ui.grid(columns=1).classes("w-full gap-1"):
            _number("Solar", "solar_cost_per_kw", inputs, step=10, suffix="\u20ac/kW")
            _number("Inverter", "inverter_cost_per_kw", inputs, step=10, suffix="\u20ac/kW")
            _number("Battery", "battery_cost_per_kwh", inputs, step=10, suffix="\u20ac/kWh")
            _number("Solar maint.", "maintenance_fee_per_kw", inputs, step=1, suffix="\u20ac/kW/yr")
            _number(
                "Batt. maint.",
                "battery_maintenance_fee_per_kwh",
                inputs,
                step=1,
                suffix="\u20ac/kWh/yr",
            )
    return section


def _financial_section(inputs):
    with (
        ui.expansion("Financial", icon="account_balance")
        .classes("w-full")
        .props("dense") as section
    ):
        use_loan = ui.switch("Use loan", value=DEFAULTS["use_loan"])
        use_loan.tooltip("Off: the investment is paid in cash upfront")
        inputs["use_loan"] = use_loan
        with ui.grid(columns=3).classes("w-full gap-1"):
            _number("Discount", "discount_rate", inputs, step=0.01)
            loan_rate = _number("Loan rate", "loan_rate", inputs, step=0.01)
            loan_years = _number("Loan yrs", "loan_years", inputs, step=1, min=1)
            loan_rate.bind_enabled_from(use_loan, "value")
            loan_years.bind_enabled_from(use_loan, "value")
    return section


def _subsidies_section(inputs):
    with ui.expansion("Subsidies", icon="savings").classes("w-full").props("dense") as section:
        use = ui.switch("Apply subsidies", value=DEFAULTS["use_subsidies"])
        use.tooltip("One-off payments deducted from the initial investment")
        inputs["use_subsidies"] = use
        with ui.column().classes("w-full gap-1") as body:

            def part(label: str, key: str, suffix: str):
                with ui.row().classes("w-full items-center no-wrap gap-1"):
                    sw = ui.switch(value=DEFAULTS[f"{key}_enabled"])
                    inputs[f"{key}_enabled"] = sw
                    rate_key = key + ("_per_kwh" if suffix.endswith("kWh") else "_per_kw")
                    n = _number(label, rate_key, inputs, step=10, min=0, suffix=suffix)
                    n.classes("grow").bind_enabled_from(sw, "value")
                return sw

            part("Solar", "subsidy_solar", "\u20ac/kW")
            part("Inverter", "subsidy_inverter", "\u20ac/kW")
            battery = part("Battery", "subsidy_battery", "\u20ac/kWh")
            ui.label("Battery subsidy only if kWh per kW is within").classes("text-caption")
            with ui.row().classes("w-full items-center no-wrap gap-1") as ratio_row:
                _number("Min", "subsidy_battery_min_ratio", inputs, step=0.1, min=0)
                _number("Max", "subsidy_battery_max_ratio", inputs, step=0.1, min=0)
                inputs["subsidy_battery_basis"] = ui.select(
                    options=SUBSIDY_BASIS_LABELS,
                    label="of",
                    value=DEFAULTS["subsidy_battery_basis"],
                ).classes("grow")
            for el in ratio_row.default_slot.children:
                el.bind_enabled_from(battery, "value")
            with ui.row().classes("w-full items-center no-wrap gap-1"):
                cap = ui.switch("Cap", value=DEFAULTS["subsidy_cap_enabled"])
                cap.tooltip("Limit the total subsidy to a fraction of the investment")
                inputs["subsidy_cap_enabled"] = cap
                frac = _number("Max fraction", "subsidy_cap_fraction", inputs, step=0.05, min=0)
                frac.classes("grow").bind_enabled_from(cap, "value")
        body.bind_visibility_from(use, "value")
    return section


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
@ui.page("/")
def index():
    # One sidebar for both tabs: every input exists once, so the tabs always share their
    # values. Each tab shows the sections it had before: System Specs is single-only;
    # Parameter Ranges, Battery, Equipment Costs, Financial and Subsidies are batch-only.
    inputs: dict = {}
    app.storage.user.setdefault("dark_mode", True)
    ui.dark_mode().bind_value(app.storage.user, "dark_mode")
    # Sidebar sections: outlined boxes with bold titles; hide the uploader's empty file list
    ui.add_css(
        """
        .q-uploader__list:empty { display: none; }
        .q-expansion-item {
            border: 1px solid rgba(128, 128, 128, 0.35);
            border-radius: 6px;
            margin-bottom: 6px;
        }
        .q-expansion-item > .q-expansion-item__container > .q-item { font-weight: 700; }
        .q-expansion-item--expanded > .q-expansion-item__container > .q-item {
            border-bottom: 1px solid rgba(128, 128, 128, 0.35);
        }
        """
    )

    with ui.header().classes("items-center justify-between"):
        ui.label("Solar Battery Simulator").classes("text-h5 text-weight-bold")
        ui.switch("Dark mode").bind_value(app.storage.user, "dark_mode").props(
            "keep-color color=grey-6"
        )

    with ui.element("div").classes("w-full relative"):
        with ui.tabs().classes("w-full") as tabs:
            batch_tab = ui.tab(BATCH_TAB)
            single_tab = ui.tab(SINGLE_TAB)

        # Run button of the selected tab, right-aligned in the tab bar
        with ui.row().classes("absolute-right items-center q-pr-md"):
            run_single = ui.button(
                "Run Simulation",
                icon="play_arrow",
                on_click=lambda: run_single_simulation(
                    build_shared_params(inputs), single_results, single_spinner
                ),
            ).props("color=primary no-wrap")
            _only_on(run_single, tabs, SINGLE_TAB)
            batch_stop = threading.Event()
            batch_running = False

            async def _toggle_batch():
                """Run Analysis, or request a stop while a run is in progress."""
                nonlocal batch_running
                if batch_running:
                    batch_stop.set()
                    batch_button.set_text("Stopping...")
                    batch_button.disable()
                    return
                batch_running = True
                batch_stop.clear()
                batch_button.set_text("Stop Analysis")
                batch_button.props("icon=stop color=negative")
                try:
                    await run_batch_analysis(
                        build_shared_params(inputs),
                        batch_results,
                        batch_spinner,
                        stop_event=batch_stop,
                    )
                finally:
                    batch_running = False
                    batch_button.set_text("Run Analysis")
                    batch_button.props("icon=play_arrow color=primary")
                    batch_button.enable()

            batch_button = ui.button("Run Analysis", icon="play_arrow", on_click=_toggle_batch)
            _only_on(batch_button.props("color=primary no-wrap"), tabs, BATCH_TAB)

    with (
        ui.splitter(value=25).classes("w-full").style("min-height: calc(100vh - 160px)") as splitter
    ):
        with splitter.before:
            with ui.scroll_area().classes("w-full").style("height: calc(100vh - 170px)"):
                with ui.column().classes("q-gutter-xs p-2 w-full"):
                    ui.label("Configuration").classes("text-h6")
                    _data_files_section(inputs)
                    _only_on(_system_specs_section(inputs), tabs, SINGLE_TAB)
                    _only_on(_parameter_ranges_section(inputs), tabs, BATCH_TAB)
                    _only_on(_battery_section(inputs), tabs, BATCH_TAB)
                    _only_on(_equipment_costs_section(inputs), tabs, BATCH_TAB)
                    _only_on(_financial_section(inputs), tabs, BATCH_TAB)
                    _only_on(_subsidies_section(inputs), tabs, BATCH_TAB)
                    _pricing_section(inputs)
                    _transmission_section(inputs)
                    _smoothing_section(inputs)
                    _heating_section(inputs)

        with splitter.after:
            with ui.tab_panels(tabs, value=batch_tab).classes("w-full"):
                with ui.tab_panel(batch_tab):
                    with ui.scroll_area().classes("w-full").style("height: calc(100vh - 170px)"):
                        batch_spinner = ui.spinner("dots", size="xl")
                        batch_spinner.visible = False
                        batch_results = ui.column().classes("w-full")
                with ui.tab_panel(single_tab):
                    with ui.scroll_area().classes("w-full").style("height: calc(100vh - 170px)"):
                        single_spinner = ui.spinner("dots", size="xl")
                        single_spinner.visible = False
                        single_results = ui.column().classes("w-full")


def _pick_port(host: str) -> int:
    """GUI_PORT if set; else 8080, or a free port when 8080 is taken (e.g. a second launch)."""
    if "GUI_PORT" in os.environ:
        return int(os.environ["GUI_PORT"])
    with socket.socket() as sock:
        try:
            sock.bind((host, 8080))
            return 8080
        except OSError:
            pass
    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def main() -> None:
    # Localhost only by default: nobody else on the network can connect. The Docker image
    # sets GUI_HOST=0.0.0.0 inside the container; compose publishes it on 127.0.0.1 only.
    host = os.environ.get("GUI_HOST", "127.0.0.1")
    ui.run(
        title="Solar Battery Simulator",
        host=host,
        port=_pick_port(host),
        show=os.environ.get("GUI_OPEN_BROWSER", "1") != "0",
        reload=False,
        storage_secret=os.environ.get("NICEGUI_STORAGE_SECRET", "solar-sim-dev-only"),
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
