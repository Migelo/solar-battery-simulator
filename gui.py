#!/usr/bin/env python3
"""NiceGUI frontend for the Power Flow Simulator."""

import asyncio
import base64
import io
import os
import socket
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import matplotlib
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

# Headless backend: the simulator may draw figures from worker threads
matplotlib.use("Agg")

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

        # Full-year power flow chart (zoomable ECharts, built in the worker thread)
        r = sim.simulation_results
        chart_options = power_flow_chart_options(r, sim.kw_factor)
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
            "chart_options": chart_options,
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

        # Power flow plot: whole year, zoomable, with a date window selector above it
        ui.label("Power Flow Visualization").classes("text-h6 q-mt-lg")
        _power_flow_chart(result["chart_options"])

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


# Days shown when the power flow chart first opens (the user picks start/end dates after)
POWER_FLOW_INITIAL_DAYS = 7
_CHART_TIME_FMT = "%Y-%m-%d %H:%M"
# ECharts cascading time-axis labels (picked by tick granularity)
_TIME_LABELS = {
    "year": "{yyyy}",
    "month": "{MMM} {yyyy}",
    "day": "{d} {MMM}",
    "hour": "{HH}:{mm}",
    "minute": "{HH}:{mm}",
}
_CHART_AXIS = {
    "axisLabel": {"color": "#9e9e9e"},
    "axisLine": {"lineStyle": {"color": "#757575"}},
    "splitLine": {"lineStyle": {"color": "rgba(128,128,128,0.25)"}},
    "nameTextStyle": {"color": "#9e9e9e"},
}


def power_flow_days(first: pd.Timestamp, last: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last calendar day covered by interval-end labels ``first``..``last``.

    A label at midnight closes the previous day's last interval, so it belongs to that day.
    """
    return first.normalize(), (last - pd.Timedelta(minutes=1)).normalize()


def power_flow_range(
    start: str, end: str, first: pd.Timestamp, last: pd.Timestamp
) -> tuple[str, str]:
    """Zoom range (chart time strings) showing the days ``start``..``end``, both inclusive.

    Swapped dates are reordered and the range is clamped to the data ``first``..``last``.
    """
    day = pd.Timedelta(days=1)
    s, e = sorted((pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()))
    begin = min(max(s, first), last - day)
    stop = min(e + day, last)
    if stop <= begin:
        stop = min(begin + day, last)
    return begin.strftime(_CHART_TIME_FMT), stop.strftime(_CHART_TIME_FMT)


def power_flow_chart_options(
    results: pd.DataFrame, kw_factor: float, initial_days: int = POWER_FLOW_INITIAL_DAYS
) -> dict:
    """ECharts options for the full-period power flow plot (power, SOC, grid; linked zoom)."""
    times = pd.to_datetime(results["datetime"])
    values = pd.DataFrame(
        {
            "solar": results["solar_generation_kwh"] * kw_factor,
            "consumption": results["consumption_kwh"] * kw_factor,
            "soc": results["battery_soc_percent"],
            "import": results["grid_import_kwh"] * kw_factor,
            "export": -results["grid_export_kwh"] * kw_factor,
        }
    ).round(2)
    values.insert(0, "time", times.dt.strftime(_CHART_TIME_FMT))
    first, last = times.min(), times.max()
    first_day = power_flow_days(first, last)[0]
    zoom = power_flow_range(
        str(first_day.date()),
        str((first_day + pd.Timedelta(days=initial_days - 1)).date()),
        first,
        last,
    )

    def series(name, dim, color, grid, area=True):
        return {
            "name": name,
            "type": "line",
            "encode": {"x": "time", "y": dim},
            "xAxisIndex": grid,
            "yAxisIndex": grid,
            "showSymbol": False,
            "sampling": "lttb",
            "lineStyle": {"width": 1.2},
            "color": color,
            **({"areaStyle": {"opacity": 0.2}} if area else {}),
        }

    grids = [("8%", "28%"), ("44%", "18%"), ("70%", "18%")]
    y_names = ["Power (kW)", "Battery SOC (%)", "Grid (kW)"]
    zoom_range = {"startValue": zoom[0], "endValue": zoom[1]}
    return {
        "animation": False,
        "backgroundColor": "transparent",
        "textStyle": {"color": "#bdbdbd"},
        "dataset": {
            "dimensions": list(values.columns),
            "source": values.values.tolist(),
        },
        "legend": {"top": 0, "textStyle": {"color": "#bdbdbd"}},
        "tooltip": {"trigger": "axis", "axisPointer": {"type": "line"}},
        "axisPointer": {"link": [{"xAxisIndex": "all"}]},
        "toolbox": {
            "right": 10,
            "feature": {"dataZoom": {"yAxisIndex": "none"}, "restore": {}},
            "iconStyle": {"borderColor": "#9e9e9e"},
        },
        "grid": [{"left": 70, "right": 30, "top": top, "height": height} for top, height in grids],
        "xAxis": [
            {
                "type": "time",
                "gridIndex": i,
                **_CHART_AXIS,
                "axisLabel": {
                    "color": "#9e9e9e",
                    "show": i == len(grids) - 1,
                    "hideOverlap": True,
                    "formatter": _TIME_LABELS,
                },
            }
            for i in range(len(grids))
        ],
        "yAxis": [
            {
                "type": "value",
                "gridIndex": i,
                "name": name,
                "nameLocation": "middle",
                "nameGap": 50,
                **_CHART_AXIS,
                **({"min": 0, "max": 100} if i == 1 else {}),
            }
            for i, name in enumerate(y_names)
        ],
        "dataZoom": [
            {"type": "inside", "xAxisIndex": [0, 1, 2], **zoom_range},
            {
                "type": "slider",
                "xAxisIndex": [0, 1, 2],
                "bottom": 10,
                "height": 28,
                "textStyle": {"color": "#9e9e9e"},
                "labelFormatter": "{yyyy}-{MM}-{dd}",
                **zoom_range,
            },
        ],
        "series": [
            series("Solar", "solar", "#ff9800", 0),
            series("Consumption", "consumption", "#42a5f5", 0, area=False),
            series("Battery SOC", "soc", "#66bb6a", 1),
            series("Grid import", "import", "#ef5350", 2),
            series("Grid export", "export", "#ab47bc", 2),
        ],
    }


def zoom_dates(event: dict, first: pd.Timestamp, last: pd.Timestamp) -> tuple[str, str] | None:
    """Start and end dates (``YYYY-MM-DD``, inclusive) of a manual ``datazoom`` event.

    Slider events carry ``start``/``end`` (percent of the data range), inside zoom/pan a
    ``batch`` of them. Zooms dispatched by the date fields themselves carry ``startValue``
    and are ignored (``None``), as are events without percentages.
    """
    zoom = (event.get("batch") or [event])[0]
    if "startValue" in zoom:
        return None
    start, end = zoom.get("start"), zoom.get("end")
    if not isinstance(start, int | float) or not isinstance(end, int | float):
        return None
    span = last - first
    t0 = first + span * (start / 100)
    t1 = first + span * (end / 100) - pd.Timedelta(minutes=1)  # midnight end = previous day
    d0, d1 = t0.normalize(), max(t1.normalize(), t0.normalize())
    return d0.strftime("%Y-%m-%d"), d1.strftime("%Y-%m-%d")


def _date_field(label: str, value: str, min_d: str, max_d: str) -> ui.input:
    """Date input (``YYYY-MM-DD``) with a calendar popup limited to ``min_d``..``max_d``."""
    with ui.input(label, value=value).props("dense outlined").classes("w-40") as field:
        with ui.menu().props("no-parent-event") as menu:
            ui.date(mask="YYYY-MM-DD").bind_value(field).props(
                f":options=\"d => d >= '{min_d}' && d <= '{max_d}'\""
                f' default-year-month="{min_d[:7]}"'
            ).on_value_change(menu.close)
        with field.add_slot("append"):
            ui.icon("edit_calendar").on("click", menu.open).classes("cursor-pointer")
    return field


def _power_flow_chart(options: dict) -> None:
    """Start/end date selector above a zoomable full-period power flow chart."""
    source = options["dataset"]["source"]
    first, last = pd.Timestamp(source[0][0]), pd.Timestamp(source[-1][0])
    first_day, last_day = power_flow_days(first, last)
    zoom = options["dataZoom"][0]
    init_start = pd.Timestamp(zoom["startValue"]).strftime("%Y-%m-%d")
    init_end = (pd.Timestamp(zoom["endValue"]) - pd.Timedelta(minutes=1)).strftime("%Y-%m-%d")

    syncing = False  # fields updated programmatically: don't re-zoom from their change events

    def set_dates(start: str, end: str) -> None:
        nonlocal syncing
        syncing = True
        try:
            start_input.value, end_input.value = start, end
        finally:
            syncing = False

    def apply(_=None) -> None:
        if syncing:
            return
        try:
            rng = power_flow_range(start_input.value, end_input.value, first, last)
        except ValueError:  # half-typed date
            return
        chart.run_chart_method(
            "dispatchAction", {"type": "dataZoom", "startValue": rng[0], "endValue": rng[1]}
        )

    def shift(direction: int) -> None:
        try:
            s, e = sorted((pd.Timestamp(start_input.value), pd.Timestamp(end_input.value)))
        except ValueError:
            return
        span = e - s + pd.Timedelta(days=1)
        s, e = s + direction * span, e + direction * span
        if s < first_day:
            s, e = first_day, first_day + span - pd.Timedelta(days=1)
        if e > last_day:
            s, e = max(first_day, last_day - span + pd.Timedelta(days=1)), last_day
        set_dates(str(s.date()), str(e.date()))
        apply()

    def whole_period() -> None:
        set_dates(str(first_day.date()), str(last_day.date()))
        apply()

    min_d, max_d = first_day.strftime("%Y/%m/%d"), last_day.strftime("%Y/%m/%d")
    with ui.row().classes("items-center q-gutter-sm q-mt-sm"):
        ui.button(icon="chevron_left", on_click=lambda: shift(-1)).props(
            "flat round dense"
        ).tooltip("Previous period of the same length")
        start_input = _date_field("Start date", init_start, min_d, max_d)
        end_input = _date_field("End date", init_end, min_d, max_d)
        ui.button(icon="chevron_right", on_click=lambda: shift(1)).props(
            "flat round dense"
        ).tooltip("Next period of the same length")
        ui.button("Whole year", icon="zoom_out_map", on_click=whole_period).props(
            "flat dense no-caps"
        )
        ui.label("Scroll / drag on the chart or use the slider to zoom").classes(
            "text-caption text-grey"
        )
    chart = ui.echart(options).classes("w-full").style("height: 680px")
    start_input.on_value_change(apply)
    end_input.on_value_change(apply)

    def follow_zoom(e) -> None:
        dates = zoom_dates(e.args or {}, first, last)
        if dates and dates != (start_input.value, end_input.value):
            set_dates(*dates)

    chart.on(
        "chart:datazoom",
        follow_zoom,
        ["start", "end", "batch", "startValue"],
        throttle=0.2,
        leading_events=False,
    )


def _metric_card(label: str, value: str):
    with ui.card().classes("p-4 text-center min-w-[140px]"):
        ui.label(value).classes("text-h6 text-weight-bold")
        ui.label(label).classes("text-caption text-grey-7")


@dataclass(frozen=True)
class HeatmapSpec:
    """One batch heatmap: a result column over the solar × battery grid."""

    key: str  # tab / file name
    tab: str
    column: str  # column in MultiScenarioAnalyzer.results
    best: str  # "max" or "min": which inverter size wins a (solar, battery) cell
    title: str
    unit: str
    decimals: int
    colors: tuple[str, ...]  # low -> high
    cap: float | None = None  # values above are shown as "<cap>+" (e.g. never paid back)


_VIRIDIS = ("#440154", "#3b528b", "#21918c", "#5ec962", "#fde725")
_RDYLGN = ("#d73027", "#fc8d59", "#fee08b", "#d9ef8b", "#91cf60", "#1a9850")
_YLGN = ("#ffffe5", "#d9f0a3", "#78c679", "#238443", "#004529")
PAYBACK_CAP_YEARS = 20

BATCH_HEATMAPS = (
    HeatmapSpec(
        "savings", "Savings", "savings_vs_baseline", "max", "Annual savings", "€/yr", 0, _VIRIDIS
    ),
    HeatmapSpec(
        "payback",
        "ROI",
        "payback_years",
        "min",
        "Payback period",
        "years",
        1,
        _RDYLGN[::-1],
        cap=PAYBACK_CAP_YEARS,
    ),
    HeatmapSpec(
        "self_sufficiency",
        "Self-Sufficiency",
        "self_sufficiency_percent",
        "max",
        "Self-sufficiency",
        "%",
        1,
        _YLGN,
    ),
    HeatmapSpec("npv", "NPV", "npv_20_years", "max", "20-year NPV", "€", 0, _RDYLGN),
)
_PLOT_BG = "#1d1d1d"  # background of downloaded PNGs (chart text is light grey)


def heatmap_grid(results: pd.DataFrame, spec: HeatmapSpec) -> pd.DataFrame:
    """Best value per (solar, battery) cell over all inverter sizes, with that inverter.

    Columns: ``solar_kw``, ``battery_kwh``, ``inverter_kw``, ``value`` (capped at
    ``spec.cap``), ``capped`` (raw value was above the cap, e.g. never paid back).
    """
    df = results[["solar_panel_power_kw", "battery_capacity_kwh", "inverter_power_kw"]].copy()
    df["raw"] = pd.to_numeric(results[spec.column], errors="coerce")
    df = df[df["raw"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["solar_kw", "battery_kwh", "inverter_kw", "value", "capped"])
    # Pick the winner on raw values (so 25 beats 30 years even though both show as "20+")
    keys = ["solar_panel_power_kw", "battery_capacity_kwh"]
    values = df.groupby(keys)["raw"]
    best = df.loc[values.idxmax() if spec.best == "max" else values.idxmin()].copy()
    best["capped"] = best["raw"] > spec.cap if spec.cap is not None else False
    best["value"] = best["raw"].clip(upper=spec.cap) if spec.cap is not None else best["raw"]
    best = best.drop(columns="raw")
    return (
        best.rename(
            columns={
                "solar_panel_power_kw": "solar_kw",
                "battery_capacity_kwh": "battery_kwh",
                "inverter_power_kw": "inverter_kw",
            }
        )
        .sort_values(["battery_kwh", "solar_kw"])
        .reset_index(drop=True)
    )


def _fmt_kw(x: float) -> str:
    return f"{x:g}"


def _fmt_value(spec: HeatmapSpec, value: float, capped: bool) -> str:
    text = f"{value:,.{spec.decimals}f}"
    return f"{text}+" if capped else text


def heatmap_chart_options(grid: pd.DataFrame, spec: HeatmapSpec) -> dict:
    """ECharts heatmap: solar on x, battery on y, colour + label = best value, rich tooltip."""
    solars = sorted(grid["solar_kw"].unique())
    batteries = sorted(grid["battery_kwh"].unique())
    data = []
    for row in grid.itertuples():
        label = _fmt_value(spec, row.value, row.capped)
        data.append(
            {
                "value": [
                    solars.index(row.solar_kw),
                    batteries.index(row.battery_kwh),
                    round(float(row.value), spec.decimals),
                ],
                "name": (
                    f"Solar {_fmt_kw(row.solar_kw)} kW · Battery {_fmt_kw(row.battery_kwh)} kWh"
                    f"<br>Best inverter: {_fmt_kw(row.inverter_kw)} kW"
                    f"<br><b>{spec.title}: {label} {spec.unit}</b>"
                ),
                "label": {"formatter": label},
            }
        )
    lo = 0 if spec.cap is not None else float(grid["value"].min()) if len(grid) else 0
    hi = spec.cap if spec.cap is not None else float(grid["value"].max()) if len(grid) else 1
    axis_style = {
        "axisLabel": {"color": "#9e9e9e"},
        "axisLine": {"lineStyle": {"color": "#757575"}},
        "nameTextStyle": {"color": "#9e9e9e"},
        "nameLocation": "middle",
        "splitArea": {"show": False},
    }
    return {
        "animation": False,
        "backgroundColor": "transparent",
        "title": {
            "text": f"{spec.title} ({spec.unit}) — best inverter per cell",
            "left": "center",
            "textStyle": {"color": "#e0e0e0", "fontSize": 15},
        },
        "tooltip": {"position": "top", ":formatter": "p => p.name"},
        "toolbox": {
            "right": 10,
            "feature": {
                "saveAsImage": {
                    "name": f"batch_{spec.key}",
                    "pixelRatio": 2,
                    "backgroundColor": _PLOT_BG,
                    "title": "Save PNG",
                }
            },
            "iconStyle": {"borderColor": "#9e9e9e"},
        },
        "grid": {"left": 80, "right": 110, "top": 50, "bottom": 60, "containLabel": False},
        "xAxis": {
            "type": "category",
            "data": [_fmt_kw(s) for s in solars],
            "name": "Solar (kW)",
            "nameGap": 32,
            **axis_style,
        },
        "yAxis": {
            "type": "category",
            "data": [_fmt_kw(b) for b in batteries],
            "name": "Battery (kWh)",
            "nameGap": 50,
            **axis_style,
        },
        "visualMap": {
            "min": lo,
            "max": hi if hi > lo else lo + 1,
            "calculable": True,
            "orient": "vertical",
            "right": 10,
            "top": "middle",
            "precision": spec.decimals,
            "inRange": {"color": list(spec.colors)},
            "textStyle": {"color": "#bdbdbd"},
            "text": [spec.unit, ""],
        },
        "series": [
            {
                "type": "heatmap",
                "data": data,
                "label": {
                    "show": len(solars) * len(batteries) <= 400,
                    "color": "#fff",
                    "textBorderColor": "rgba(0,0,0,0.7)",
                    "textBorderWidth": 2,
                    "fontSize": 11,
                },
                "itemStyle": {"borderColor": _PLOT_BG, "borderWidth": 1},
                "emphasis": {"itemStyle": {"borderColor": "#fff", "borderWidth": 2}},
            }
        ],
    }


def heatmap_csv_bytes(grid: pd.DataFrame, spec: HeatmapSpec) -> bytes:
    """Heatmap grid as CSV (battery rows × solar columns), Slovenian locale like the table export."""
    pivot = grid.pivot(index="battery_kwh", columns="solar_kw", values="value").round(
        max(spec.decimals, 2)
    )
    pivot.columns = [f"Solar {_fmt_kw(c)} kW" for c in pivot.columns]
    pivot.index = [_fmt_kw(i) for i in pivot.index]
    pivot.index.name = f"Battery (kWh) \\ {spec.title} ({spec.unit})"
    text = pivot.to_csv(sep=";", decimal=",", lineterminator="\r\n")
    return text.encode("utf-8-sig")


def _heatmap_height(grid: pd.DataFrame) -> int:
    rows = max(grid["battery_kwh"].nunique(), 1)
    return max(360, min(900, 34 * rows + 140))


def _batch_heatmap(grid: pd.DataFrame, spec: HeatmapSpec) -> None:
    """Interactive heatmap with PNG / CSV download buttons."""
    if grid.empty:
        ui.label("No valid scenarios for this metric.").classes("text-grey q-pa-md")
        return

    async def download_png() -> None:
        url = await chart.run_chart_method(
            "getDataURL",
            {
                "type": "png",
                "pixelRatio": 2,
                "backgroundColor": _PLOT_BG,
                "excludeComponents": ["toolbox"],
            },
        )
        if isinstance(url, str) and "," in url:
            ui.download.content(base64.b64decode(url.split(",", 1)[1]), f"batch_{spec.key}.png")
        else:
            ui.notify("Could not export the chart image", type="warning")

    with ui.row().classes("items-center q-gutter-sm"):
        ui.button("PNG", icon="image", on_click=download_png).props("outline dense")
        ui.button(
            "CSV",
            icon="download",
            on_click=lambda: ui.download.content(
                heatmap_csv_bytes(grid, spec), f"batch_{spec.key}.csv"
            ),
        ).props("outline dense")
        ui.label("Hover a cell for details; drag the colour bar to filter.").classes(
            "text-caption text-grey"
        )
    chart = (
        ui.echart(heatmap_chart_options(grid, spec))
        .classes("w-full")
        .style(f"height: {_heatmap_height(grid)}px")
    )


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

        # Heatmap grids (rendered client-side as interactive ECharts)
        df = pd.DataFrame(analyzer.results)
        heatmaps = [
            (spec, heatmap_grid(df, spec)) for spec in BATCH_HEATMAPS if spec.column in df.columns
        ]

        comparison_df = comparison_dataframe(analyzer.results)
        return {
            "heatmaps": heatmaps,
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

        # Heatmaps: interactive, each with PNG / CSV download
        with ui.tabs().classes("q-mt-md") as tabs:
            heatmap_tabs = [ui.tab(spec.tab) for spec, _ in result["heatmaps"]]
        with ui.tab_panels(tabs, value=heatmap_tabs[0] if heatmap_tabs else None).classes("w-full"):
            for tab, (spec, grid) in zip(heatmap_tabs, result["heatmaps"], strict=True):
                with ui.tab_panel(tab):
                    _batch_heatmap(grid, spec)

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
    # Middle of the batch grid below, sized for the bundled ~100-300 kW site
    "solar_power": 400.0,
    "inverter_power": 300.0,
    "battery_capacity": 250.0,
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
    # Sized for the bundled data (a site drawing ~100-300 kW): smaller PV never has surplus,
    # so the battery would never charge. 6 solar/inverter pairs x 3 batteries = 18 scenarios.
    "solar_range": "200,400,600",
    "inverter_range": "150,300,450",
    "battery_range": "0,250,500",
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
        /* Results panels: keep content at the viewport width so charts fit and wide tables
           scroll inside themselves instead of stretching the whole panel */
        .results-scroll .q-scrollarea__content { width: 100%; }
        .results-scroll .q-table__container { max-width: 100%; }
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
                    with (
                        ui.scroll_area()
                        .classes("w-full results-scroll")
                        .style("height: calc(100vh - 170px)")
                    ):
                        batch_spinner = ui.spinner("dots", size="xl")
                        batch_spinner.visible = False
                        batch_results = ui.column().classes("w-full")
                with ui.tab_panel(single_tab):
                    with (
                        ui.scroll_area()
                        .classes("w-full results-scroll")
                        .style("height: calc(100vh - 170px)")
                    ):
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
