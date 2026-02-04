"""Tests for core power flow simulation logic."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from power_flow_simulator import PowerFlowSimulator


class TestEnergyBalance:
    """Test energy conservation in simulation."""

    def test_energy_conservation(self, simulator_minimal):
        """Total energy in equals total energy out."""
        simulator_minimal.simulate_power_flows()
        r = simulator_minimal.simulation_results
        
        energy_in = (
            r['solar_generation_kwh'].sum() +
            r['grid_import_kwh'].sum() +
            r['battery_discharge_kwh'].sum()
        )
        energy_out = (
            r['consumption_kwh'].sum() +
            r['grid_export_kwh'].sum() +
            r['battery_charge_kwh'].sum()
        )
        
        assert np.isclose(energy_in, energy_out, rtol=1e-6)

    def test_energy_conservation_no_battery(self, simulator_no_battery):
        """Energy conserved with zero battery capacity."""
        simulator_no_battery.simulate_power_flows()
        r = simulator_no_battery.simulation_results
        
        # With no battery: solar + import = consumption + export
        energy_in = r['solar_generation_kwh'].sum() + r['grid_import_kwh'].sum()
        energy_out = r['consumption_kwh'].sum() + r['grid_export_kwh'].sum()
        
        assert np.isclose(energy_in, energy_out, rtol=1e-6)
        # Battery activity should be zero
        assert r['battery_charge_kwh'].sum() == 0
        assert r['battery_discharge_kwh'].sum() == 0

    def test_consumption_equals_sources(self, simulator_minimal):
        """Consumption met by solar + battery + grid."""
        simulator_minimal.simulate_power_flows()
        r = simulator_minimal.simulation_results
        
        # For each interval: consumption = solar_used + battery_discharge + grid_import
        # (where solar_used = solar_generation - battery_charge - grid_export)
        total_consumption = r['consumption_kwh'].sum()
        total_solar = r['solar_generation_kwh'].sum()
        total_export = r['grid_export_kwh'].sum()
        total_charge = r['battery_charge_kwh'].sum()
        total_discharge = r['battery_discharge_kwh'].sum()
        total_import = r['grid_import_kwh'].sum()
        
        # solar_used_directly = solar - charge - export
        solar_used = total_solar - total_charge - total_export
        supplied = solar_used + total_discharge + total_import
        
        assert np.isclose(supplied, total_consumption, rtol=1e-6)


class TestBatteryBehavior:
    """Test battery charge/discharge logic."""

    def test_soc_within_bounds(self, simulator_minimal):
        """Battery SOC stays within 0-100%."""
        simulator_minimal.simulate_power_flows()
        r = simulator_minimal.simulation_results
        
        assert r['battery_soc_percent'].min() >= 0
        assert r['battery_soc_percent'].max() <= 100

    def test_soc_within_capacity(self, simulator_minimal):
        """Battery SOC kWh stays within capacity."""
        simulator_minimal.simulate_power_flows()
        r = simulator_minimal.simulation_results
        
        assert r['battery_soc_kwh'].min() >= 0
        assert r['battery_soc_kwh'].max() <= simulator_minimal.battery_capacity_kwh

    @pytest.mark.parametrize("capacity", [0, 5, 10, 20, 50])
    def test_soc_bounds_various_capacities(self, minimal_production_csv, minimal_consumption_csv, capacity):
        """SOC stays in bounds for various battery capacities."""
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=capacity,
            production_file=str(minimal_production_csv),
            consumption_file=str(minimal_consumption_csv),
        )
        sim.load_and_align_data()
        sim.scale_solar_generation()
        sim.simulate_power_flows()
        
        r = sim.simulation_results
        assert r['battery_soc_kwh'].min() >= 0
        assert r['battery_soc_kwh'].max() <= capacity + 1e-6  # Small tolerance

    def test_battery_efficiency_losses(self, simulator_minimal):
        """Charging loses energy due to efficiency."""
        simulator_minimal.simulate_power_flows()
        r = simulator_minimal.simulation_results
        
        total_charge_input = r['battery_charge_kwh'].sum()
        total_discharge = r['battery_discharge_kwh'].sum()
        initial_soc = 0  # Simulation starts at 0
        final_soc = r['battery_soc_kwh'].iloc[-1]
        
        # Energy stored = charge_input * efficiency
        efficiency = simulator_minimal.battery_efficiency
        expected_stored = total_charge_input * efficiency
        
        # stored - discharged = final_soc - initial_soc
        delta_soc = final_soc - initial_soc
        actual_net = expected_stored - total_discharge
        
        assert np.isclose(actual_net, delta_soc, atol=1.0)  # 1 kWh tolerance for rounding

    def test_no_discharge_when_empty(self, minimal_production_csv, minimal_consumption_csv):
        """Battery cannot discharge below 0."""
        # High consumption, low solar, small battery
        sim = PowerFlowSimulator(
            solar_panel_power_kw=1.0,  # Very small solar
            inverter_power_kw=1.0,
            battery_capacity_kwh=1.0,  # Small battery
            production_file=str(minimal_production_csv),
            consumption_file=str(minimal_consumption_csv),
        )
        sim.load_and_align_data()
        sim.scale_solar_generation()
        sim.simulate_power_flows()
        
        # SOC should never go negative
        assert sim.simulation_results['battery_soc_kwh'].min() >= 0


class TestSolarBehavior:
    """Test solar generation and usage."""

    def test_solar_charges_battery_when_surplus(self, simulator_minimal):
        """Solar surplus charges battery before export."""
        simulator_minimal.simulate_power_flows()
        r = simulator_minimal.simulation_results
        
        # Find intervals with solar surplus (solar > consumption)
        surplus_mask = r['solar_generation_kwh'] > r['consumption_kwh']
        surplus_intervals = r[surplus_mask]
        
        if len(surplus_intervals) > 0:
            # Should have some battery charging during surplus
            total_charge = surplus_intervals['battery_charge_kwh'].sum()
            assert total_charge > 0 or surplus_intervals['grid_export_kwh'].sum() > 0

    def test_grid_export_only_surplus(self, simulator_minimal):
        """Grid export only when solar exceeds consumption + charge capacity."""
        simulator_minimal.simulate_power_flows()
        r = simulator_minimal.simulation_results
        
        # Grid export should only happen when there's surplus
        for _, row in r.iterrows():
            if row['grid_export_kwh'] > 0:
                # Must have solar surplus
                assert row['solar_generation_kwh'] >= row['consumption_kwh']


class TestGridBehavior:
    """Test grid import/export logic."""

    def test_grid_import_when_deficit(self, simulator_minimal):
        """Grid import covers deficit after battery."""
        simulator_minimal.simulate_power_flows()
        r = simulator_minimal.simulation_results
        
        # Check intervals with deficit
        for _, row in r.iterrows():
            net_demand = row['consumption_kwh'] - row['solar_generation_kwh']
            if net_demand > 0:
                # Deficit should be covered by battery + grid
                supply = row['battery_discharge_kwh'] + row['grid_import_kwh']
                assert np.isclose(supply, net_demand, rtol=0.01)

    def test_no_simultaneous_import_export(self, simulator_minimal):
        """Cannot import and export in same interval."""
        simulator_minimal.simulate_power_flows()
        r = simulator_minimal.simulation_results
        
        # Should never have both import and export > 0
        simultaneous = (r['grid_import_kwh'] > 0) & (r['grid_export_kwh'] > 0)
        assert not simultaneous.any()

    def test_no_battery_activity_zero_capacity(self, simulator_no_battery):
        """Zero capacity battery has no activity."""
        simulator_no_battery.simulate_power_flows()
        r = simulator_no_battery.simulation_results
        
        assert r['battery_charge_kwh'].sum() == 0
        assert r['battery_discharge_kwh'].sum() == 0
        assert (r['battery_soc_kwh'] == 0).all()
