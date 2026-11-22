"""Interactive batch heatmaps: best-inverter grids, chart options, CSV export."""

import pandas as pd
import pytest

from gui import (
    BATCH_HEATMAPS,
    PAYBACK_CAP_YEARS,
    heatmap_chart_options,
    heatmap_csv_bytes,
    heatmap_grid,
)
from power_flow_simulator import MultiScenarioAnalyzer

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
