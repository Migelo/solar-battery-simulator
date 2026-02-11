#!/usr/bin/env python3
"""NiceGUI frontend for the Power Flow Simulator."""

import asyncio
import io
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from nicegui import app, ui

from power_flow_simulator import MultiScenarioAnalyzer, PowerFlowSimulator

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

# Persistent temp dir for uploaded files (lives for the process lifetime)
_upload_dir = Path(tempfile.mkdtemp(prefix="solar_sim_"))


def _find_data_file(name: str) -> str:
    """Locate a data file: uploaded files first, then data/ subdir, then cwd."""
    for candidate in [_upload_dir / name, Path("data") / name, Path(name)]:
        if candidate.exists():
            return str(candidate)
    return str(Path("data") / name)


def _handle_upload(event, target_name: str, label_widget) -> None:
    """Save an uploaded file and update the label."""
    dest = _upload_dir / target_name
    dest.write_bytes(event.content.read())
    label_widget.text = f"{target_name}: {event.name}"


def _data_files_section(inputs: dict):
    """Create file upload section — collapsed when files are already found."""
    prod_found = _find_data_file("production.csv")
    cons_found = _find_data_file("consumption.csv")
    prod_exists = Path(prod_found).exists()
    cons_exists = Path(cons_found).exists()
    both_found = prod_exists and cons_exists

    with ui.expansion(
        "Data Files", icon="folder_open", value=not both_found
    ).classes("w-full"):
        if both_found:
            ui.label(f"Using: {prod_found}, {cons_found}").classes("text-caption text-positive")
        else:
            ui.label("Upload production.csv and consumption.csv").classes(
                "text-caption text-warning"
            )

        with ui.row().classes("w-full q-gutter-sm items-center"):
            prod_label = ui.label(
                f"production.csv: {'found' if prod_exists else 'missing'}"
            ).classes("text-caption")
            ui.upload(
                label="Upload",
                auto_upload=True,
                on_upload=lambda e: _handle_upload(e, "production.csv", prod_label),
            ).props("accept=.csv flat dense").classes("max-w-[180px]")

        with ui.row().classes("w-full q-gutter-sm items-center"):
            cons_label = ui.label(
                f"consumption.csv: {'found' if cons_exists else 'missing'}"
            ).classes("text-caption")
            ui.upload(
                label="Upload",
                auto_upload=True,
                on_upload=lambda e: _handle_upload(e, "consumption.csv", cons_label),
            ).props("accept=.csv flat dense").classes("max-w-[180px]")


def make_transmission_costs(b1, b2, b3, b4, b5):
    return {"block1": b1, "block2": b2, "block3": b3, "block4": b4, "block5": b5}


def make_monthly_power_fees(b1, b2, b3, b4, b5):
    return {"block1": b1, "block2": b2, "block3": b3, "block4": b4, "block5": b5}


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor="none")
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

        sim = PowerFlowSimulator(
            solar_panel_power_kw=params["solar_power"],
            inverter_power_kw=params["inverter_power"],
            battery_capacity_kwh=params["battery_capacity"],
            battery_c_rate=params["battery_c_rate"],
            battery_efficiency=params["battery_efficiency"],
            peak_price=params["peak_price"],
            off_peak_price=params["off_peak_price"],
            transmission_costs=make_transmission_costs(
                params["tb1"], params["tb2"], params["tb3"], params["tb4"], params["tb5"]
            ),
            monthly_power_fees=make_monthly_power_fees(
                params["pf1"], params["pf2"], params["pf3"], params["pf4"], params["pf5"]
            ),
            ove_spte_fee=params["ove_spte_fee"],
            enable_power_smoothing=params["enable_power_smoothing"],
            min_soc_reserve=params["min_soc_reserve"],
            max_power_by_block=max_power_by_block,
            heating_config=heating_config,
            production_file=params.get("production_file", _find_data_file("production.csv")),
            consumption_file=params.get("consumption_file", _find_data_file("consumption.csv")),
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
        plot_data = r.head(days * 24 * 4).copy()
        idx = range(len(plot_data))

        with plt.rc_context(_GUI_PLOT_RC):
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9))
            ax1.plot(idx, plot_data["solar_generation_kwh"] * 4, label="Solar", color="orange", lw=1.5)
            ax1.plot(idx, plot_data["consumption_kwh"] * 4, label="Consumption", color="blue", lw=1.5)
            ax1.fill_between(idx, 0, plot_data["solar_generation_kwh"] * 4, alpha=0.2, color="orange")
            ax1.set_ylabel("Power (kW)")
            ax1.set_title(f"Power Flows — First {days} Days")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2.plot(idx, plot_data["battery_soc_percent"], color="green", lw=1.5)
            ax2.fill_between(idx, 0, plot_data["battery_soc_percent"], alpha=0.2, color="green")
            ax2.set_ylabel("Battery SOC (%)")
            ax2.set_ylim(0, 100)
            ax2.grid(True, alpha=0.3)

            ax3.plot(idx, plot_data["grid_import_kwh"] * 4, label="Import", color="red", lw=1.5)
            ax3.plot(idx, -plot_data["grid_export_kwh"] * 4, label="Export", color="purple", lw=1.5)
            ax3.fill_between(idx, 0, plot_data["grid_import_kwh"] * 4, alpha=0.2, color="red")
            ax3.fill_between(idx, 0, -plot_data["grid_export_kwh"] * 4, alpha=0.2, color="purple")
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
        self_sufficiency = (1 - total_import / total_consumption) * 100 if total_consumption > 0 else 0
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
                rows.append({
                    "Block": block_name.replace("_", " ").title(),
                    "Intervals": data["intervals"],
                    "% of Year": f"{data['percentage_of_year']:.1f}",
                    "Import (kWh)": f"{data['total_import_kwh']:.1f}",
                    "Rate (\u20ac/kWh)": f"{data['transmission_rate_eur_per_kwh']:.5f}",
                    "Trans. Cost (\u20ac)": f"{data['transmission_cost_eur']:.2f}",
                    "Max Power (kW)": f"{data['max_import_power_kw']:.1f}",
                    "Annual Fee (\u20ac)": f"{data['annual_power_fee_eur']:.2f}",
                })
            columns = [{"name": k, "label": k, "field": k, "sortable": True} for k in rows[0]]
            ui.table(columns=columns, rows=rows).classes("q-mt-sm")


def _b64(data: bytes) -> str:
    import base64
    return base64.b64encode(data).decode()


def _metric_card(label: str, value: str):
    with ui.card().classes("p-4 text-center min-w-[140px]"):
        ui.label(value).classes("text-h6 text-weight-bold")
        ui.label(label).classes("text-caption text-grey-7")


async def run_batch_analysis(params: dict, results_container, spinner):
    """Run batch analysis in a background thread with progress tracking."""
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
            progress_label.set_text(
                f"{progress['phase']}: {current}/{total} scenarios"
            )
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
            loan_years=params["loan_years"],
            transmission_costs=make_transmission_costs(
                params["tb1"], params["tb2"], params["tb3"], params["tb4"], params["tb5"]
            ),
            ove_spte_fee=params["ove_spte_fee"],
            enable_power_smoothing=params["enable_power_smoothing"],
            min_soc_reserve=params["min_soc_reserve"],
            max_power_by_block=max_power_by_block,
            heating_config=heating_config,
            production_file=params.get("production_file", _find_data_file("production.csv")),
            consumption_file=params.get("consumption_file", _find_data_file("consumption.csv")),
        )

        progress["phase"] = "Generating scenarios"
        analyzer.generate_scenarios()
        progress["total"] = len(analyzer.scenarios)
        progress["phase"] = "Running simulations"
        # Store analyzer ref so the timer can poll len(analyzer.results)
        progress["analyzer"] = analyzer

        analyzer.run_all_scenarios()

        progress["current"] = progress["total"]
        progress["phase"] = "Generating plots"
        analyzer.create_comparison_summary()

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
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis",
                         cbar_kws={"label": "Annual Savings (\u20ac)"}, ax=ax)
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
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn_r", vmin=0, vmax=20,
                         cbar_kws={"label": "Payback (years)"}, ax=ax)
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
            sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGn",
                         cbar_kws={"label": "Self-Sufficiency (%)"}, ax=ax)
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
                sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn",
                             cbar_kws={"label": "NPV 20yr (\u20ac)"}, ax=ax)
                ax.set_title("20-Year NPV: Solar vs Battery")
                ax.set_xlabel("Solar (kW)")
                ax.set_ylabel("Battery (kWh)")
                fig.tight_layout()
                plots["npv"] = fig_to_png_bytes(fig)

        # Comparison table data
        table_rows = []
        for r in sorted(analyzer.results, key=lambda x: x["savings_vs_baseline"], reverse=True):
            table_rows.append({
                "Solar (kW)": r["solar_panel_power_kw"],
                "Inverter (kW)": r["inverter_power_kw"],
                "Battery (kWh)": r["battery_capacity_kwh"],
                "Savings (\u20ac/yr)": f"{r['savings_vs_baseline']:,.0f}",
                "Payback (yr)": f"{r['payback_years']:.1f}",
                "NPV 20yr (\u20ac)": f"{r.get('npv_20_years', 0):,.0f}",
                "Self-Suff (%)": f"{r.get('self_sufficiency_percent', 0):.1f}",
                "Investment (\u20ac)": f"{r.get('total_investment', 0):,.0f}",
            })

        return {"plots": plots, "table_rows": table_rows, "n_scenarios": len(analyzer.results)}

    try:
        result = await asyncio.to_thread(_run)
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
            ui.table(columns=columns, rows=result["table_rows"]).classes("q-mt-sm")


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
    "loan_rate": 0.03,
    "loan_years": 10,
}


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
            _number("Export price", "export_price", inputs, step=0.001, suffix="\u20ac/kWh")


def _transmission_section(inputs):
    with ui.expansion("Transmission Costs", icon="electrical_services").classes("w-full"):
        ui.label("Transmission costs (\u20ac/kWh)").classes("text-caption")
        with ui.row().classes("q-gutter-sm"):
            for i in range(1, 6):
                _number(f"Block {i}", f"tb{i}", inputs, step=0.0001)
        ui.separator()
        ui.label("Monthly power fees (\u20ac/kW/month)").classes("text-caption")
        with ui.row().classes("q-gutter-sm"):
            for i in range(1, 6):
                _number(f"Block {i}", f"pf{i}", inputs, step=0.01)
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


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
@ui.page("/")
def index():
    app.storage.user.setdefault("dark_mode", True)
    ui.dark_mode().bind_value(app.storage.user, "dark_mode")

    with ui.header().classes("items-center justify-between"):
        ui.label("Solar Battery Simulator").classes("text-h5 text-weight-bold")
        ui.switch("Dark mode").bind_value(app.storage.user, "dark_mode").props(
            "keep-color color=grey-6"
        )

    with ui.tabs().classes("w-full") as tabs:
        single_tab = ui.tab("Single Scenario")
        batch_tab = ui.tab("Batch Analysis")

    with ui.tab_panels(tabs, value=single_tab).classes("w-full").style("min-height: calc(100vh - 130px)"):
        # ==================================================================
        # SINGLE SCENARIO TAB
        # ==================================================================
        with ui.tab_panel(single_tab):
            single_inputs = {}
            single_results = ui.column().classes("w-full")
            single_spinner = ui.spinner("dots", size="xl")
            single_spinner.visible = False

            with ui.splitter(value=25).classes("w-full").style("min-height: calc(100vh - 160px)") as splitter:
                with splitter.before:
                    with ui.scroll_area().classes("w-full").style("height: calc(100vh - 170px)"):
                        with ui.column().classes("q-gutter-xs p-2 w-full"):
                            with ui.row().classes("w-full items-center justify-between"):
                                ui.label("Configuration").classes("text-h6")
                                ui.button(
                                    "Run Simulation",
                                    icon="play_arrow",
                                    on_click=lambda: run_single_simulation(
                                        build_shared_params(single_inputs),
                                        single_results,
                                        single_spinner,
                                    ),
                                ).props("color=primary dense")

                            _data_files_section(single_inputs)

                            with ui.expansion("System Specs", icon="solar_power", value=True).classes("w-full").props("dense"):
                                with ui.grid(columns=3).classes("w-full gap-1"):
                                    _number("Solar", "solar_power", single_inputs, step=0.5, suffix="kW")
                                    _number("Inverter", "inverter_power", single_inputs, step=0.5, suffix="kW")
                                    _number("Battery", "battery_capacity", single_inputs, step=1, suffix="kWh")
                                with ui.grid(columns=2).classes("w-full gap-1"):
                                    _number("C-rate", "battery_c_rate", single_inputs, step=0.1)
                                    _number("Efficiency", "battery_efficiency", single_inputs, step=0.05)

                            _pricing_section(single_inputs)
                            _transmission_section(single_inputs)
                            _smoothing_section(single_inputs)
                            _heating_section(single_inputs)

                with splitter.after:
                    with ui.column().classes("q-pa-md w-full"):
                        single_spinner
                        single_results

        # ==================================================================
        # BATCH ANALYSIS TAB
        # ==================================================================
        with ui.tab_panel(batch_tab):
            batch_inputs = {}
            batch_results = ui.column().classes("w-full")
            batch_spinner = ui.spinner("dots", size="xl")
            batch_spinner.visible = False

            with ui.splitter(value=25).classes("w-full").style("min-height: calc(100vh - 160px)") as splitter:
                with splitter.before:
                    with ui.scroll_area().classes("w-full").style("height: calc(100vh - 170px)"):
                        with ui.column().classes("q-gutter-xs p-2 w-full"):
                            with ui.row().classes("w-full items-center justify-between"):
                                ui.label("Configuration").classes("text-h6")
                                ui.button(
                                    "Run Analysis",
                                    icon="play_arrow",
                                    on_click=lambda: run_batch_analysis(
                                        build_shared_params(batch_inputs),
                                        batch_results,
                                        batch_spinner,
                                    ),
                                ).props("color=primary dense no-wrap")

                            _data_files_section(batch_inputs)

                            with ui.expansion("Parameter Ranges", icon="grid_view", value=True).classes("w-full").props("dense"):
                                batch_inputs["solar_range"] = ui.input(
                                    label="Solar range (kW)", value=DEFAULTS["solar_range"]
                                ).classes("w-full")
                                batch_inputs["inverter_range"] = ui.input(
                                    label="Inverter range (kW)", value=DEFAULTS["inverter_range"]
                                ).classes("w-full")
                                batch_inputs["battery_range"] = ui.input(
                                    label="Battery range (kWh)", value=DEFAULTS["battery_range"]
                                ).classes("w-full")

                            with ui.expansion("Battery", icon="battery_charging_full").classes("w-full").props("dense"):
                                with ui.grid(columns=2).classes("w-full gap-1"):
                                    _number("C-rate", "battery_c_rate", batch_inputs, step=0.1)
                                    _number("Efficiency", "battery_efficiency", batch_inputs, step=0.05)

                            with ui.expansion("Equipment Costs", icon="euro", value=True).classes("w-full").props("dense"):
                                with ui.grid(columns=2).classes("w-full gap-1"):
                                    _number("Solar", "solar_cost_per_kw", batch_inputs, step=10, suffix="\u20ac/kW")
                                    _number("Inverter", "inverter_cost_per_kw", batch_inputs, step=10, suffix="\u20ac/kW")
                                    _number("Battery", "battery_cost_per_kwh", batch_inputs, step=10, suffix="\u20ac/kWh")
                                    _number("Solar maint.", "maintenance_fee_per_kw", batch_inputs, step=1, suffix="\u20ac/kW/yr")
                                    _number("Batt. maint.", "battery_maintenance_fee_per_kwh", batch_inputs, step=1, suffix="\u20ac/kWh/yr")

                            with ui.expansion("Financial", icon="account_balance").classes("w-full").props("dense"):
                                with ui.grid(columns=3).classes("w-full gap-1"):
                                    _number("Discount", "discount_rate", batch_inputs, step=0.01)
                                    _number("Loan rate", "loan_rate", batch_inputs, step=0.01)
                                    _number("Loan yrs", "loan_years", batch_inputs, step=1)

                            _pricing_section(batch_inputs)
                            _transmission_section(batch_inputs)
                            _smoothing_section(batch_inputs)
                            _heating_section(batch_inputs)

                with splitter.after:
                    with ui.column().classes("q-pa-md w-full"):
                        batch_spinner
                        batch_results


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="Solar Battery Simulator", port=8080, reload=False, storage_secret="solar-sim")
