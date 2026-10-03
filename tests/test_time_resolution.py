"""Tests for time resolution (15-min vs hourly bins)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from power_flow_simulator import MultiScenarioAnalyzer, PowerFlowSimulator


def _run(sim: PowerFlowSimulator):
    sim.load_and_align_data()
    sim.scale_solar_generation()
    sim.simulate_power_flows()
    return sim.calculate_costs_and_savings(peak_price=0.15, off_peak_price=0.10, export_price=0.05)


class TestTimeResolution:
    def test_default_is_15min(self, simulator_minimal):
        assert simulator_minimal.time_resolution == "15min"
        assert simulator_minimal.interval_hours == 0.25
        assert simulator_minimal.kw_factor == 4.0

    def test_invalid_resolution_rejected(self, minimal_production_csv, minimal_consumption_csv):
        with pytest.raises(ValueError, match="Unsupported time_resolution"):
            PowerFlowSimulator(
                production_file=str(minimal_production_csv),
                consumption_file=str(minimal_consumption_csv),
                time_resolution="30min",
            )

    def test_hourly_resamples_row_count(self, minimal_production_csv, minimal_consumption_csv):
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(minimal_production_csv),
            consumption_file=str(minimal_consumption_csv),
            time_resolution="1h",
        )
        sim.load_and_align_data()
        assert sim.df is not None
        # 96 fifteen-min intervals -> 24 hourly bins
        assert len(sim.df) == 24
        assert sim.interval_hours == 1.0
        assert sim.kw_factor == 1.0

    def test_hourly_conserves_energy(self, minimal_production_csv, minimal_consumption_csv):
        sim15 = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(minimal_production_csv),
            consumption_file=str(minimal_consumption_csv),
        )
        sim15.load_and_align_data()
        assert sim15.df is not None
        total_15 = sim15.df["consumption_kwh"].sum()

        sim1h = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(minimal_production_csv),
            consumption_file=str(minimal_consumption_csv),
            time_resolution="1h",
        )
        sim1h.load_and_align_data()
        assert sim1h.df is not None
        total_1h = sim1h.df["consumption_kwh"].sum()

        assert np.isclose(total_15, total_1h, rtol=1e-9)

    def test_hourly_pipeline_energy_conservation(
        self, minimal_production_csv, minimal_consumption_csv
    ):
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(minimal_production_csv),
            consumption_file=str(minimal_consumption_csv),
            time_resolution="1h",
        )
        _run(sim)
        r = sim.simulation_results
        assert r is not None
        assert len(r) == 24

        energy_in = (
            r["solar_generation_kwh"].sum()
            + r["grid_import_kwh"].sum()
            + r["battery_discharge_kwh"].sum()
        )
        energy_out = (
            r["consumption_kwh"].sum() + r["grid_export_kwh"].sum() + r["battery_charge_kwh"].sum()
        )
        assert np.isclose(energy_in, energy_out, rtol=1e-6)
        assert r["battery_soc_percent"].min() >= 0
        assert r["battery_soc_percent"].max() <= 100

    def test_hourly_max_power_matches_average(
        self, minimal_production_csv, minimal_consumption_csv
    ):
        """At 1h resolution, max block power = max interval energy (kW == kWh)."""
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(minimal_production_csv),
            consumption_file=str(minimal_consumption_csv),
            time_resolution="1h",
        )
        _run(sim)
        block_power = sim.calculate_block_power("grid_import_kwh")
        r = sim.simulation_results
        assert r is not None
        expected = r[r["transmission_block"] == 3]["grid_import_kwh"].max()
        assert np.isclose(block_power["block_3"], expected, rtol=1e-9)

    def test_hourly_row_count_requires_multiple_of_4(self, minimal_production_csv, tmp_path):
        import pandas as pd

        # 95 intervals: not divisible into hourly bins
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-15 00:15:00", periods=95, freq="15min").strftime(
                    "%d. %m. %Y %H:%M:%S"
                ),
                "energy_kwh": [0.5] * 95,
                "power_kw": [2.0] * 95,
                "transmission_block": [3] * 95,
                "extra": [""] * 95,
            }
        )
        odd_path = tmp_path / "consumption_odd.csv"
        df.to_csv(odd_path, index=False)
        sim = PowerFlowSimulator(
            production_file=str(minimal_production_csv),
            consumption_file=str(odd_path),
            time_resolution="1h",
        )
        with pytest.raises(ValueError, match="multiple of 4"):
            sim.load_and_align_data()

    def test_batch_analyzer_threads_resolution(
        self, minimal_production_csv, minimal_consumption_csv
    ):
        analyzer = MultiScenarioAnalyzer(
            solar_range=[10.0],
            inverter_range=[10.0],
            battery_range=[10.0],
            production_file=str(minimal_production_csv),
            consumption_file=str(minimal_consumption_csv),
            time_resolution="1h",
        )
        analyzer.generate_scenarios()
        results = analyzer.run_all_scenarios()
        assert len(results) == 1
        assert results[0]["time_resolution"] == "1h"
        assert results[0]["battery_log"] is not None
        assert len(results[0]["battery_log"]) == 24
