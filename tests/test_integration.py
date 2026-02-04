"""Integration tests for end-to-end simulation pipeline."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from power_flow_simulator import PowerFlowSimulator, MultiScenarioAnalyzer


class TestFullPipeline:
    """Test complete simulation pipeline."""

    def test_full_pipeline_runs(self, minimal_production_csv, minimal_consumption_csv):
        """Full pipeline completes without error."""
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
        costs = sim.calculate_costs_and_savings(
            peak_price=0.15,
            off_peak_price=0.10,
            export_price=0.05
        )
        
        assert 'baseline_cost' in costs
        assert 'savings_vs_baseline' in costs
        assert sim.simulation_results is not None
        assert len(sim.simulation_results) > 0

    def test_pipeline_with_heating(self, winter_production_csv, winter_consumption_csv):
        """Pipeline with heating load completes."""
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(winter_production_csv),
            consumption_file=str(winter_consumption_csv),
            heating_config={'heating_kwh': 1000.0, 'start_hour': 7, 'end_hour': 0},
        )
        
        sim.load_and_align_data()
        sim.scale_solar_generation()
        sim.simulate_power_flows()
        costs = sim.calculate_costs_and_savings(
            peak_price=0.15,
            off_peak_price=0.10,
            export_price=0.05
        )
        
        assert costs['baseline_cost'] > 0

    def test_results_dataframe_columns(self, simulator_minimal):
        """Simulation results have expected columns."""
        simulator_minimal.simulate_power_flows()
        r = simulator_minimal.simulation_results
        
        expected_columns = [
            'datetime',
            'solar_generation_kwh',
            'consumption_kwh',
            'battery_soc_kwh',
            'battery_soc_percent',
            'battery_charge_kwh',
            'battery_discharge_kwh',
            'grid_import_kwh',
            'grid_export_kwh',
            'transmission_block',
        ]
        
        for col in expected_columns:
            assert col in r.columns, f"Missing column: {col}"


class TestDeterminism:
    """Test that simulations are deterministic."""

    def test_same_inputs_same_outputs(self, minimal_production_csv, minimal_consumption_csv):
        """Same inputs produce identical outputs."""
        results = []
        
        for _ in range(2):
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
            results.append(sim.simulation_results.copy())
        
        # Results should be identical
        pd.testing.assert_frame_equal(results[0], results[1])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_solar(self, minimal_production_csv, minimal_consumption_csv, tmp_path):
        """System works with zero solar production."""
        # Create zero solar production file
        df = pd.DataFrame({
            'timestamp_id': range(1, 97),
            'solar_power_kw': [0.0] * 96
        })
        zero_solar_path = tmp_path / "zero_solar.csv"
        df.to_csv(zero_solar_path, index=False)
        
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(zero_solar_path),
            consumption_file=str(minimal_consumption_csv),
        )
        sim.load_and_align_data()
        sim.scale_solar_generation()
        sim.simulate_power_flows()
        
        r = sim.simulation_results
        # All consumption should come from grid
        assert r['solar_generation_kwh'].sum() == 0
        assert r['grid_import_kwh'].sum() > 0
        assert r['grid_export_kwh'].sum() == 0

    def test_zero_consumption(self, minimal_production_csv, tmp_path):
        """System works with zero consumption."""
        # Create zero consumption file
        df = pd.DataFrame({
            'datetime': pd.date_range('2024-01-15 00:15:00', periods=96, freq='15min').strftime('%d. %m. %Y %H:%M:%S'),
            'energy_kwh': [0.0] * 96,
            'power_kw': [0.0] * 96,
            'transmission_block': [3] * 96,
            'extra': [''] * 96
        })
        zero_consumption_path = tmp_path / "zero_consumption.csv"
        df.to_csv(zero_consumption_path, index=False)
        
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(minimal_production_csv),
            consumption_file=str(zero_consumption_path),
        )
        sim.load_and_align_data()
        sim.scale_solar_generation()
        sim.simulate_power_flows()
        
        r = sim.simulation_results
        # All solar should go to battery or export
        assert r['grid_import_kwh'].sum() == 0

    def test_very_large_battery(self, minimal_production_csv, minimal_consumption_csv):
        """System handles very large battery capacity."""
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10000.0,  # 10 MWh
            production_file=str(minimal_production_csv),
            consumption_file=str(minimal_consumption_csv),
        )
        sim.load_and_align_data()
        sim.scale_solar_generation()
        sim.simulate_power_flows()
        
        # Should complete without error
        assert sim.simulation_results is not None
        # SOC should stay in bounds
        assert sim.simulation_results['battery_soc_kwh'].max() <= 10000.0


class TestMultiScenarioAnalyzer:
    """Test batch analysis functionality."""

    def test_generates_scenarios(self, minimal_production_csv, minimal_consumption_csv):
        """Analyzer generates correct number of scenarios."""
        analyzer = MultiScenarioAnalyzer(
            solar_range=[5, 10],
            inverter_range=[5, 10],
            battery_range=[0, 10],
            production_file=str(minimal_production_csv),
            consumption_file=str(minimal_consumption_csv),
        )
        analyzer.generate_scenarios()
        
        # 2 solar * 2 inverter * 2 battery = 8, but some may be filtered
        # (inverter > solar is invalid)
        assert len(analyzer.scenarios) > 0
        assert len(analyzer.scenarios) <= 8

    def test_scenarios_have_required_keys(self, minimal_production_csv, minimal_consumption_csv):
        """Each scenario has required configuration keys."""
        analyzer = MultiScenarioAnalyzer(
            solar_range=[10],
            inverter_range=[10],
            battery_range=[10],
            production_file=str(minimal_production_csv),
            consumption_file=str(minimal_consumption_csv),
        )
        analyzer.generate_scenarios()
        
        required_keys = [
            'solar_panel_power_kw',
            'inverter_power_kw', 
            'battery_capacity_kwh',
        ]
        
        for scenario in analyzer.scenarios:
            for key in required_keys:
                assert key in scenario
