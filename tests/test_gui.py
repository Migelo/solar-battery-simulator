"""Tests for the NiceGUI frontend."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from gui import (
    DEFAULTS,
    _find_data_file,
    build_shared_params,
    make_monthly_power_fees,
    make_transmission_costs,
)


# ---------------------------------------------------------------------------
# Unit tests — pure functions, no NiceGUI server needed
# ---------------------------------------------------------------------------


class TestMakeTransmissionCosts:
    def test_returns_correct_keys(self):
        result = make_transmission_costs(1, 2, 3, 4, 5)
        assert set(result.keys()) == {"block1", "block2", "block3", "block4", "block5"}

    def test_returns_correct_values(self):
        result = make_transmission_costs(0.01, 0.02, 0.03, 0.04, 0.05)
        assert result["block1"] == 0.01
        assert result["block5"] == 0.05


class TestMakeMonthlyPowerFees:
    def test_returns_correct_keys(self):
        result = make_monthly_power_fees(1, 2, 3, 4, 5)
        assert set(result.keys()) == {"block1", "block2", "block3", "block4", "block5"}

    def test_returns_correct_values(self):
        result = make_monthly_power_fees(3.76, 1.05, 0.13, 0.0, 0.0)
        assert result["block1"] == 3.76
        assert result["block4"] == 0.0


class TestFindDataFile:
    def test_prefers_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "production.csv").write_text("header\n")
        (tmp_path / "production.csv").write_text("header\n")

        result = _find_data_file("production.csv")
        assert result == str(Path("data") / "production.csv")

    def test_falls_back_to_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "production.csv").write_text("header\n")

        result = _find_data_file("production.csv")
        assert result == "production.csv"

    def test_defaults_to_data_dir_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _find_data_file("nonexistent.csv")
        assert result == str(Path("data") / "nonexistent.csv")


class TestDefaults:
    def test_single_scenario_keys_present(self):
        required = [
            "solar_power", "inverter_power", "battery_capacity",
            "battery_c_rate", "battery_efficiency",
            "peak_price", "off_peak_price", "export_price",
        ]
        for key in required:
            assert key in DEFAULTS, f"Missing default: {key}"

    def test_batch_keys_present(self):
        required = [
            "solar_range", "inverter_range", "battery_range",
            "solar_cost_per_kw", "battery_cost_per_kwh",
            "discount_rate", "loan_rate", "loan_years",
        ]
        for key in required:
            assert key in DEFAULTS, f"Missing default: {key}"

    def test_transmission_block_keys(self):
        for i in range(1, 6):
            assert f"tb{i}" in DEFAULTS
            assert f"pf{i}" in DEFAULTS

    def test_default_values_reasonable(self):
        assert 0 < DEFAULTS["solar_power"] < 100
        assert 0 < DEFAULTS["battery_efficiency"] <= 1.0
        assert 0 < DEFAULTS["peak_price"] < 1.0
        assert DEFAULTS["peak_price"] > DEFAULTS["off_peak_price"]


# ---------------------------------------------------------------------------
# NiceGUI User fixture tests — headless, no browser
# ---------------------------------------------------------------------------


@pytest.fixture
def gui_csv_files(tmp_path):
    """Create minimal CSV files for GUI simulation tests."""
    hours = np.arange(96) / 4
    solar_kw = np.maximum(0, 10 * np.sin((hours - 6) * np.pi / 12))
    solar_kw[hours < 6] = 0
    solar_kw[hours >= 18] = 0

    prod = pd.DataFrame({"timestamp_id": range(1, 97), "solar_power_kw": solar_kw})
    prod_path = tmp_path / "production.csv"
    prod.to_csv(prod_path, index=False)

    cons = pd.DataFrame({
        "datetime": pd.date_range("2024-01-15 00:15:00", periods=96, freq="15min")
        .strftime("%d. %m. %Y %H:%M:%S"),
        "energy_kwh": [0.5] * 96,
        "power_kw": [2.0] * 96,
        "transmission_block": [3] * 96,
        "extra": [""] * 96,
    })
    cons_path = tmp_path / "consumption.csv"
    cons.to_csv(cons_path, index=False)

    return str(prod_path), str(cons_path)


class TestPageStructure:
    """Test that the GUI pages load and contain expected elements."""

    async def test_index_loads(self, user):
        await user.open("/")
        await user.should_see("Solar Battery Simulator")

    async def test_single_scenario_tab_visible(self, user):
        await user.open("/")
        await user.should_see("Single Scenario")

    async def test_batch_analysis_tab_visible(self, user):
        await user.open("/")
        await user.should_see("Batch Analysis")

    async def test_single_tab_has_system_inputs(self, user):
        await user.open("/")
        await user.should_see("Configuration")
        await user.should_see("Run Simulation")

    async def test_single_tab_has_parameter_sections(self, user):
        await user.open("/")
        await user.should_see("Electricity Pricing")
        await user.should_see("Transmission Costs")
        await user.should_see("Power Smoothing")
        await user.should_see("Heating Load")
