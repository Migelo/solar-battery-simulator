"""CLI output directory: writers stay under --out-dir, guard blocks clobbering."""

import os
import re

import pytest

from solar_battery_simulator.simulator import (
    MultiScenarioAnalyzer,
    PowerFlowSimulator,
    default_out_dir,
    prepare_out_dir,
)


def _single_simulator(minimal_production_csv, minimal_consumption_csv):
    sim = PowerFlowSimulator(
        solar_panel_power_kw=10.0,
        inverter_power_kw=10.0,
        battery_capacity_kwh=10.0,
        production_file=str(minimal_production_csv),
        consumption_file=str(minimal_consumption_csv),
    )
    sim.load_and_align_data()
    sim.scale_solar_generation()
    sim.simulate_power_flows()
    return sim


def _tiny_analyzer(minimal_production_csv, minimal_consumption_csv):
    analyzer = MultiScenarioAnalyzer(
        solar_range=[5, 10],
        inverter_range=[5],
        battery_range=[0, 10],
        production_file=str(minimal_production_csv),
        consumption_file=str(minimal_consumption_csv),
    )
    analyzer.generate_scenarios()
    analyzer.run_all_scenarios()
    return analyzer


def test_single_run_writes_only_under_out_dir(
    minimal_production_csv, minimal_consumption_csv, tmp_path, monkeypatch
):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    out = tmp_path / "out"

    sim = _single_simulator(minimal_production_csv, minimal_consumption_csv)
    written = sim.export_results(output_dir=prepare_out_dir(out))

    assert sorted(p.name for p in out.iterdir()) == [
        "power_flow_simulation_results.csv",
        "system_summary.csv",
    ]
    assert written == [
        str(out / "power_flow_simulation_results.csv"),
        str(out / "system_summary.csv"),
    ]
    assert list(cwd.glob("*.csv")) == []
    assert list(cwd.glob("*.png")) == []


def test_batch_summary_writes_only_under_out_dir(
    minimal_production_csv, minimal_consumption_csv, tmp_path, monkeypatch
):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    out = tmp_path / "out"

    analyzer = _tiny_analyzer(minimal_production_csv, minimal_consumption_csv)
    analyzer.create_comparison_summary(output_dir=prepare_out_dir(out))
    # GUI/server path still writes nothing.
    analyzer.create_comparison_summary(output_path=None)

    assert (out / "multi_scenario_comparison.csv").exists()
    assert list(cwd.glob("*.csv")) == []
    assert list(cwd.glob("*.png")) == []


def test_prepare_out_dir_guard(tmp_path):
    out = tmp_path / "out"
    prepare_out_dir(out)
    (out / "system_summary.csv").write_text("stale")
    with pytest.raises(SystemExit, match="--overwrite"):
        prepare_out_dir(out)
    prepare_out_dir(out, overwrite=True)  # reusing with the flag is fine
    assert (out / "system_summary.csv").exists()


def test_default_out_dir_is_timestamped_run():
    parent, stamp = default_out_dir().split(os.sep)
    assert parent == "outputs"
    assert re.fullmatch(r"run-\d{8}-\d{6}", stamp) is not None
