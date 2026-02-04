"""Tests for heating load functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from power_flow_simulator import PowerFlowSimulator


class TestHeatingLoadDistribution:
    """Test heating load energy distribution."""

    def test_heating_not_applied_when_config_none(self, simulator_minimal):
        """No heating load when heating_config is None."""
        # simulator_minimal has no heating config
        assert simulator_minimal.heating_config is None
        # Consumption should be unchanged (0.5 kWh * 96 = 48 kWh)
        total = simulator_minimal.df['consumption_kwh'].sum()
        assert np.isclose(total, 48.0, rtol=0.01)

    def test_heating_increases_consumption(self, winter_production_csv, winter_consumption_csv):
        """Heating load increases total consumption."""
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
        baseline_consumption = sim_no_heat.df['consumption_kwh'].sum()

        # With heating
        heating_kwh = 500.0
        sim_heat = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(winter_production_csv),
            consumption_file=str(winter_consumption_csv),
            heating_config={'heating_kwh': heating_kwh, 'start_hour': 7, 'end_hour': 0},
        )
        sim_heat.load_and_align_data()
        heated_consumption = sim_heat.df['consumption_kwh'].sum()

        # Heating should add energy (not full amount since we only have Jan data)
        assert heated_consumption > baseline_consumption

    def test_heating_only_during_specified_hours(self, winter_production_csv, winter_consumption_csv):
        """Heating only applied during start_hour to end_hour."""
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(winter_production_csv),
            consumption_file=str(winter_consumption_csv),
            heating_config={'heating_kwh': 1000.0, 'start_hour': 10, 'end_hour': 14},
        )
        sim.load_and_align_data()

        # Check that consumption outside 10-14 is unchanged (0.5 kWh)
        outside_hours = sim.df[(sim.df['hour'] < 10) | (sim.df['hour'] >= 14)]
        # All should be base consumption (0.5 kWh)
        assert np.allclose(outside_hours['consumption_kwh'], 0.5, rtol=0.01)

    def test_heating_monthly_weights(self, tmp_path):
        """January gets higher weight than March."""
        # Create data spanning Jan and March
        n_jan = 96 * 2  # 2 days Jan
        n_mar = 96 * 2  # 2 days Mar
        
        jan_dates = pd.date_range('2024-01-15 00:15:00', periods=n_jan, freq='15min')
        mar_dates = pd.date_range('2024-03-15 00:15:00', periods=n_mar, freq='15min')
        all_dates = pd.concat([pd.Series(jan_dates), pd.Series(mar_dates)])
        
        consumption_df = pd.DataFrame({
            'datetime': all_dates.dt.strftime('%d. %m. %Y %H:%M:%S'),
            'energy_kwh': [0.5] * (n_jan + n_mar),
            'power_kw': [2.0] * (n_jan + n_mar),
            'transmission_block': [3] * (n_jan + n_mar),
            'extra': [''] * (n_jan + n_mar)
        })
        consumption_path = tmp_path / "consumption.csv"
        consumption_df.to_csv(consumption_path, index=False)
        
        production_df = pd.DataFrame({
            'timestamp_id': range(1, n_jan + n_mar + 1),
            'solar_power_kw': [5.0] * (n_jan + n_mar)
        })
        production_path = tmp_path / "production.csv"
        production_df.to_csv(production_path, index=False)
        
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(production_path),
            consumption_file=str(consumption_path),
            heating_config={'heating_kwh': 6600.0, 'start_hour': 0, 'end_hour': 0},  # All day
        )
        sim.load_and_align_data()
        
        # January factor = 1.5, March factor = 0.8
        # January should have higher average consumption increase
        jan_data = sim.df[sim.df['month'] == 1]
        mar_data = sim.df[sim.df['month'] == 3]
        
        jan_avg = jan_data['consumption_kwh'].mean()
        mar_avg = mar_data['consumption_kwh'].mean()
        
        # Jan should have ~1.875x the heating of March (1.5/0.8)
        assert jan_avg > mar_avg


class TestHeatingLoadEdgeCases:
    """Test edge cases for heating load."""

    def test_heating_with_zero_kwh(self, winter_production_csv, winter_consumption_csv):
        """Zero heating_kwh should not change consumption."""
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(winter_production_csv),
            consumption_file=str(winter_consumption_csv),
            heating_config={'heating_kwh': 0.0, 'start_hour': 7, 'end_hour': 0},
        )
        sim.load_and_align_data()
        
        # All consumption should be base (0.5 kWh)
        assert np.allclose(sim.df['consumption_kwh'], 0.5, rtol=0.01)

    def test_heating_end_hour_midnight(self, winter_production_csv, winter_consumption_csv):
        """end_hour=0 means heating until midnight (hour 23)."""
        sim = PowerFlowSimulator(
            solar_panel_power_kw=10.0,
            inverter_power_kw=10.0,
            battery_capacity_kwh=10.0,
            production_file=str(winter_production_csv),
            consumption_file=str(winter_consumption_csv),
            heating_config={'heating_kwh': 1000.0, 'start_hour': 20, 'end_hour': 0},
        )
        sim.load_and_align_data()
        
        # Hours 20-23 should have increased consumption
        heated_hours = sim.df[sim.df['hour'] >= 20]
        unheated_hours = sim.df[sim.df['hour'] < 20]
        
        assert heated_hours['consumption_kwh'].mean() > 0.5
        assert np.allclose(unheated_hours['consumption_kwh'], 0.5, rtol=0.01)
