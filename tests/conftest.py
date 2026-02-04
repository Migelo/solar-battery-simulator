"""Shared test fixtures for power flow simulator tests."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from power_flow_simulator import PowerFlowSimulator


@pytest.fixture
def minimal_production_csv(tmp_path):
    """Create minimal production CSV with 96 intervals (1 day)."""
    # Simulate a day of solar production (peak at noon)
    hours = np.arange(96) / 4  # 0, 0.25, 0.5, ... 23.75
    # Simple solar curve: 0 at night, peak at noon
    solar_kw = np.maximum(0, 10 * np.sin((hours - 6) * np.pi / 12))
    solar_kw[hours < 6] = 0
    solar_kw[hours >= 18] = 0
    
    df = pd.DataFrame({
        'timestamp_id': range(1, 97),
        'solar_power_kw': solar_kw
    })
    
    path = tmp_path / "production.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def minimal_consumption_csv(tmp_path):
    """Create minimal consumption CSV with 96 intervals (1 day)."""
    # Constant 2 kW load = 0.5 kWh per 15-min interval
    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-15 00:15:00', periods=96, freq='15min').strftime('%d. %m. %Y %H:%M:%S'),
        'energy_kwh': [0.5] * 96,
        'power_kw': [2.0] * 96,
        'transmission_block': [3] * 96,  # Block 3 for simplicity
        'extra': [''] * 96
    })
    
    path = tmp_path / "consumption.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def winter_consumption_csv(tmp_path):
    """Create consumption CSV spanning heating months (Jan) for heating load tests."""
    # 4 days in January
    n_intervals = 96 * 4
    df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01 00:15:00', periods=n_intervals, freq='15min').strftime('%d. %m. %Y %H:%M:%S'),
        'energy_kwh': [0.5] * n_intervals,
        'power_kw': [2.0] * n_intervals,
        'transmission_block': [3] * n_intervals,
        'extra': [''] * n_intervals
    })
    
    path = tmp_path / "consumption.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def winter_production_csv(tmp_path):
    """Create matching production CSV for winter tests."""
    n_intervals = 96 * 4
    hours = np.tile(np.arange(96) / 4, 4)
    solar_kw = np.maximum(0, 10 * np.sin((hours - 6) * np.pi / 12))
    solar_kw[hours < 6] = 0
    solar_kw[hours >= 18] = 0
    
    df = pd.DataFrame({
        'timestamp_id': range(1, n_intervals + 1),
        'solar_power_kw': solar_kw
    })
    
    path = tmp_path / "production.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def simulator_minimal(minimal_production_csv, minimal_consumption_csv):
    """Create a PowerFlowSimulator with minimal test data."""
    sim = PowerFlowSimulator(
        solar_panel_power_kw=10.0,
        inverter_power_kw=10.0,
        battery_capacity_kwh=10.0,
        battery_efficiency=0.9,
        production_file=str(minimal_production_csv),
        consumption_file=str(minimal_consumption_csv),
    )
    sim.load_and_align_data()
    sim.scale_solar_generation()
    return sim


@pytest.fixture
def simulator_no_battery(minimal_production_csv, minimal_consumption_csv):
    """Create a PowerFlowSimulator with no battery."""
    sim = PowerFlowSimulator(
        solar_panel_power_kw=10.0,
        inverter_power_kw=10.0,
        battery_capacity_kwh=0.0,
        production_file=str(minimal_production_csv),
        consumption_file=str(minimal_consumption_csv),
    )
    sim.load_and_align_data()
    sim.scale_solar_generation()
    return sim


@pytest.fixture
def simulator_with_heating(winter_production_csv, winter_consumption_csv):
    """Create a PowerFlowSimulator with heating load configured."""
    sim = PowerFlowSimulator(
        solar_panel_power_kw=10.0,
        inverter_power_kw=10.0,
        battery_capacity_kwh=10.0,
        production_file=str(winter_production_csv),
        consumption_file=str(winter_consumption_csv),
        heating_config={
            'heating_kwh': 1000.0,
            'start_hour': 7,
            'end_hour': 0,  # midnight
        }
    )
    sim.load_and_align_data()
    sim.scale_solar_generation()
    return sim
