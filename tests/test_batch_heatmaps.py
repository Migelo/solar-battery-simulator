"""Interactive batch heatmaps: best-inverter grids, chart options, CSV export."""

import pandas as pd
import pytest

from solar_battery_simulator.gui import (
    BATCH_HEATMAPS,
    PAYBACK_CAP_YEARS,
    heatmap_cell,
    heatmap_chart_options,
    heatmap_csv_bytes,
    heatmap_grid,
    scenario_text,
)
from solar_battery_simulator.simulator import MultiScenarioAnalyzer

SPECS = {s.key: s for s in BATCH_HEATMAPS}


def _results():
    # 2 solar × 2 battery cells, two inverters for solar 10
    return pd.DataFrame(
        {
            "solar_panel_power_kw": [5, 5, 10, 10, 10, 10],
            "battery_capacity_kwh": [0, 10, 0, 0, 10, 10],
            "inverter_power_kw": [5, 5, 5, 8, 5, 8],
            "savings_vs_baseline": [100.0, 150.0, 200.0, 250.0, 300.0, 280.0],
            "payback_years": [5.0, float("inf"), 7.0, 6.0, 30.0, 25.0],
            "self_sufficiency_percent": [10.0, 12.0, 20.0, 21.0, 30.0, 29.0],
            "npv_20_years": [1000.0, -50.0, 2000.0, 2100.0, 3000.0, 2900.0],
        }
    )


def test_grid_picks_best_inverter_per_cell():
    grid = heatmap_grid(_results(), SPECS["savings"])
    assert len(grid) == 4
    cell = grid[(grid.solar_kw == 10) & (grid.battery_kwh == 0)].iloc[0]
    assert (cell.value, cell.inverter_kw) == (250.0, 8)
    cell = grid[(grid.solar_kw == 10) & (grid.battery_kwh == 10)].iloc[0]
    assert (cell.value, cell.inverter_kw) == (300.0, 5)


def test_payback_takes_minimum_and_caps():
    grid = heatmap_grid(_results(), SPECS["payback"])
    cell = grid[(grid.solar_kw == 10) & (grid.battery_kwh == 0)].iloc[0]
    assert (cell.value, cell.inverter_kw, cell.capped) == (6.0, 8, False)
    never = grid[(grid.solar_kw == 5) & (grid.battery_kwh == 10)].iloc[0]
    assert never.value == PAYBACK_CAP_YEARS and never.capped
    late = grid[(grid.solar_kw == 10) & (grid.battery_kwh == 10)].iloc[0]
    assert (late.value, late.inverter_kw, late.capped) == (PAYBACK_CAP_YEARS, 8, True)


def test_chart_options_cells_labels_and_tooltip():
    spec = SPECS["payback"]
    opts = heatmap_chart_options(heatmap_grid(_results(), spec), spec)
    assert opts["xAxis"]["data"] == ["5", "10"]
    assert opts["yAxis"]["data"] == ["0", "10"]
    data = opts["series"][0]["data"]
    assert len(data) == 4
    never = next(d for d in data if d["value"][:2] == [0, 1])
    assert never["label"]["formatter"] == "20.0+"
    assert "Best inverter: 5 kW" in never["name"]
    assert (opts["visualMap"]["min"], opts["visualMap"]["max"]) == (0, PAYBACK_CAP_YEARS)
    assert opts["toolbox"]["feature"]["saveAsImage"]["name"] == "batch_payback"


def test_value_range_follows_data_without_cap():
    spec = SPECS["npv"]
    opts = heatmap_chart_options(heatmap_grid(_results(), spec), spec)
    assert (opts["visualMap"]["min"], opts["visualMap"]["max"]) == (-50.0, 3000.0)


def test_csv_is_slovenian_grid():
    spec = SPECS["self_sufficiency"]
    text = heatmap_csv_bytes(heatmap_grid(_results(), spec), spec).decode("utf-8-sig")
    lines = text.splitlines()
    assert lines[0].endswith(";Solar 5 kW;Solar 10 kW")
    assert lines[1] == "0;10,0;21,0"
    assert lines[2] == "10;12,0;30,0"


def test_missing_values_dropped():
    df = _results()
    df["npv_20_years"] = float("nan")
    assert heatmap_grid(df, SPECS["npv"]).empty


@pytest.fixture
def batch_results(minimal_production_csv, minimal_consumption_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep the joblib .cache out of the repo
    analyzer = MultiScenarioAnalyzer(
        solar_range=[5, 10],
        inverter_range=[5],
        battery_range=[0, 10],
        production_file=str(minimal_production_csv),
        consumption_file=str(minimal_consumption_csv),
    )
    analyzer.generate_scenarios()
    analyzer.run_all_scenarios()
    return pd.DataFrame(analyzer.results)


def test_real_batch_results_build_every_heatmap(batch_results):
    for spec in BATCH_HEATMAPS:
        assert spec.column in batch_results.columns
        grid = heatmap_grid(batch_results, spec)
        assert len(grid) == 4
        heatmap_chart_options(grid, spec)
        heatmap_csv_bytes(grid, spec)


def test_heatmap_cell_maps_data_index_to_scenario():
    grid = heatmap_grid(_results(), SPECS["savings"])
    # grid rows sorted by (battery, solar): (5,0), (10,0), (5,10), (10,10)
    assert heatmap_cell(grid, 1) == (10.0, 8.0, 0.0)
    assert heatmap_cell(grid, 3) == (10.0, 5.0, 10.0)
    assert heatmap_cell(grid, 4) is None
    assert heatmap_cell(grid, None) is None
    assert scenario_text((10.0, 8.0, 0.0)) == "Solar 10 kW · Inverter 8 kW · Battery 0 kWh"


async def test_heatmap_click_and_context_menu_adopt_scenario(
    user, minimal_production_csv, minimal_consumption_csv, tmp_path, monkeypatch
):
    """Left click sets System Specs; the right-click menu sets them or runs the scenario."""
    import asyncio

    from nicegui import ui

    from solar_battery_simulator import gui

    (tmp_path / "production.csv").write_bytes(minimal_production_csv.read_bytes())
    (tmp_path / "consumption.csv").write_bytes(minimal_consumption_csv.read_bytes())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gui, "_find_data_file", lambda name: str(tmp_path / name))

    await user.open("/")

    def widget(kind, label):
        return next(
            e
            for e in user.client.layout.descendants()
            if isinstance(e, kind) and e.props.get("label") == label
        )

    widget(ui.input, "Solar range (kW)").value = "5, 10"
    widget(ui.input, "Inverter range (kW)").value = "5"
    widget(ui.input, "Battery range (kWh)").value = "0, 10"
    user.find("Run Analysis").click()

    def charts_on_page():
        return [e for e in user.client.layout.descendants() if isinstance(e, ui.echart)]

    for _ in range(300):
        if charts_on_page():
            break
        await asyncio.sleep(0.05)
    chart = min(charts_on_page(), key=lambda e: e.id)  # Savings tab
    specs = {n: widget(ui.number, n) for n in ("Solar", "Inverter", "Battery")}

    # Left click on cell 3 = solar 10, battery 10 (inverter 5)
    charts = user.find(kind=ui.echart)
    charts.elements = {chart}
    charts.trigger(
        "componentClick",
        {
            "componentType": "series",
            "seriesType": "heatmap",
            "seriesIndex": 0,
            "seriesName": "",
            "name": "",
            "dataIndex": 3,
            "data": {},
            "value": [1, 1, 0],
        },
    )
    await asyncio.sleep(0.1)
    assert (specs["Solar"].value, specs["Inverter"].value, specs["Battery"].value) == (10, 5, 10)
    await user.should_see("Single Scenario set to Solar 10 kW")

    # Right click on cell 0 = solar 5, battery 0, then "Run this scenario now"
    charts.trigger("chart:contextmenu", {"dataIndex": 0})
    await user.should_see("Solar 5 kW · Inverter 5 kW · Battery 0 kWh")
    section = min(user.find("Run this scenario now").elements, key=lambda e: e.id)
    item = user.find(kind=ui.menu_item)
    item.elements = {section.parent_slot.parent}  # the Savings chart's menu item
    item.click()
    for _ in range(200):
        if specs["Solar"].value == 5 and any(
            isinstance(e, ui.label) and e.text == "Annual Savings"
            for e in user.client.layout.descendants()
        ):
            break
        await asyncio.sleep(0.05)
    assert (specs["Solar"].value, specs["Inverter"].value, specs["Battery"].value) == (5, 5, 0)
    tabs = next(e for e in user.client.layout.descendants() if isinstance(e, ui.tabs))
    assert gui._tab_name(tabs.value) == gui.SINGLE_TAB
