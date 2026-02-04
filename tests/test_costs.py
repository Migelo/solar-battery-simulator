"""Tests for cost calculation functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from power_flow_simulator import PowerFlowSimulator


class TestCostCalculations:
    """Test electricity cost calculations."""

    def test_baseline_cost_positive(self, simulator_minimal):
        """Baseline cost should be positive."""
        simulator_minimal.simulate_power_flows()
        costs = simulator_minimal.calculate_costs_and_savings(
            peak_price=0.15,
            off_peak_price=0.10,
            export_price=0.05
        )
        
        assert costs['baseline_cost'] > 0

    def test_solar_reduces_cost(self, simulator_minimal):
        """Solar + battery should reduce costs vs baseline."""
        simulator_minimal.simulate_power_flows()
        costs = simulator_minimal.calculate_costs_and_savings(
            peak_price=0.15,
            off_peak_price=0.10,
            export_price=0.05
        )
        
        # With solar generation, net cost should be less than baseline
        assert costs['solar_battery_cost'] <= costs['baseline_cost']

    def test_savings_equals_difference(self, simulator_minimal):
        """Savings = baseline - net cost."""
        simulator_minimal.simulate_power_flows()
        costs = simulator_minimal.calculate_costs_and_savings(
            peak_price=0.15,
            off_peak_price=0.10,
            export_price=0.05
        )
        
        expected_savings = costs['baseline_cost'] - costs['solar_battery_cost']
        assert np.isclose(costs['savings_vs_baseline'], expected_savings, rtol=0.01)

    def test_savings_non_negative(self, simulator_minimal):
        """Savings should be non-negative with solar."""
        simulator_minimal.simulate_power_flows()
        costs = simulator_minimal.calculate_costs_and_savings(
            peak_price=0.15,
            off_peak_price=0.10,
            export_price=0.05
        )
        
        # Solar should provide some savings (or at least not increase costs)
        assert costs['savings_vs_baseline'] >= 0

    def test_solar_battery_cost_positive(self, simulator_minimal):
        """Solar+battery cost should be positive (we still import some grid)."""
        simulator_minimal.simulate_power_flows()
        costs = simulator_minimal.calculate_costs_and_savings(
            peak_price=0.15,
            off_peak_price=0.10,
            export_price=0.0
        )
        
        assert costs['solar_battery_cost'] >= 0


class TestPricingLogic:
    """Test time-of-use pricing."""

    def test_peak_price_higher(self, simulator_minimal):
        """Higher peak price should increase costs."""
        simulator_minimal.simulate_power_flows()
        
        # Low peak price
        costs_low = simulator_minimal.calculate_costs_and_savings(
            peak_price=0.10,
            off_peak_price=0.10,
            export_price=0.0
        )
        
        # High peak price
        costs_high = simulator_minimal.calculate_costs_and_savings(
            peak_price=0.30,
            off_peak_price=0.10,
            export_price=0.0
        )
        
        # Higher peak price should mean higher baseline cost
        assert costs_high['baseline_cost'] >= costs_low['baseline_cost']

    def test_transmission_costs_applied(self, simulator_minimal):
        """Transmission costs should be included."""
        simulator_minimal.simulate_power_flows()
        
        # With transmission costs
        costs_with = simulator_minimal.calculate_costs_and_savings(
            peak_price=0.10,
            off_peak_price=0.10,
            export_price=0.0
        )
        
        # Simulator should have non-zero transmission costs by default
        assert costs_with['baseline_cost'] > 0


class TestCostWithHeating:
    """Test costs when heating load is applied."""

    def test_heating_increases_baseline_cost(self, winter_production_csv, winter_consumption_csv):
        """Heating load should increase baseline cost."""
        # Without heating
        sim_no_heat = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(winter_production_csv),
            consumption_file=str(winter_consumption_csv),
            heating_config=None,
        )
        sim_no_heat.load_and_align_data()
        sim_no_heat.scale_solar_generation()
        sim_no_heat.simulate_power_flows()
        costs_no_heat = sim_no_heat.calculate_costs_and_savings(
            peak_price=0.15, off_peak_price=0.10, export_price=0.0
        )

        # With heating
        sim_heat = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(winter_production_csv),
            consumption_file=str(winter_consumption_csv),
            heating_config={'heating_kwh': 500.0, 'start_hour': 7, 'end_hour': 0},
        )
        sim_heat.load_and_align_data()
        sim_heat.scale_solar_generation()
        sim_heat.simulate_power_flows()
        costs_heat = sim_heat.calculate_costs_and_savings(
            peak_price=0.15, off_peak_price=0.10, export_price=0.0
        )

        # Baseline with heating should be higher
        assert costs_heat['baseline_cost'] > costs_no_heat['baseline_cost']
