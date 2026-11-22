"""Tests for the NiceGUI frontend."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nicegui import ui
from nicegui.elements.upload_files import SmallFileUpload

from solar_battery_simulator import gui
from solar_battery_simulator.gui import (
    DEFAULTS,
    _find_data_file,
    _handle_upload,
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

    @pytest.mark.parametrize("ext", [".xlsx", ".xlsm", ".xls"])
    def test_finds_excel_variant_of_same_stem(self, tmp_path, monkeypatch, ext):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / f"consumption{ext}").write_bytes(b"x")
        assert _find_data_file("consumption.csv") == str(Path("data") / f"consumption{ext}")

    def test_exact_name_preferred_over_other_extension(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "consumption.xlsx").write_bytes(b"x")
        (tmp_path / "consumption.csv").write_text("x")
        assert _find_data_file("consumption.csv") == "consumption.csv"

    def test_upload_dir_beats_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        monkeypatch.setattr(gui, "_upload_dir", upload_dir)
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "consumption.csv").write_text("x")
        (upload_dir / "consumption.xlsx").write_bytes(b"x")
        assert _find_data_file("consumption.csv") == str(upload_dir / "consumption.xlsx")


class _FakeUpload:
    """Shaped like NiceGUI's UploadEventArguments, with its real FileUpload in ``file``."""

    def __init__(self, name: str, content: bytes):
        self.file = SmallFileUpload(name=name, content_type="", _data=content)


class _FakeLabel:
    text = ""


class TestHandleUpload:
    @pytest.fixture
    def upload_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "_upload_dir", tmp_path)
        return tmp_path

    async def test_keeps_uploaded_extension(self, upload_dir):
        label = _FakeLabel()
        upload = _FakeUpload("Moj-Elektro.XLSX", b"data")
        assert await _handle_upload(upload, "consumption.csv", label)
        assert (upload_dir / "consumption.xlsx").read_bytes() == b"data"
        assert not (upload_dir / "consumption.csv").exists()
        assert label.text == "consumption: found (Moj-Elektro.XLSX)"

    async def test_new_upload_replaces_other_format(self, upload_dir):
        (upload_dir / "consumption.csv").write_text("old")
        await _handle_upload(_FakeUpload("new.xls", b"new"), "consumption.csv", _FakeLabel())
        assert not (upload_dir / "consumption.csv").exists()
        assert (upload_dir / "consumption.xls").read_bytes() == b"new"

    async def test_rejects_unsupported_type(self, upload_dir):
        label = _FakeLabel()
        assert not await _handle_upload(_FakeUpload("data.pdf", b"x"), "consumption.csv", label)
        assert list(upload_dir.iterdir()) == []
        assert "unsupported" in label.text


class TestDefaults:
    def test_single_scenario_keys_present(self):
        required = [
            "solar_power",
            "inverter_power",
            "battery_capacity",
            "battery_c_rate",
            "battery_efficiency",
            "peak_price",
            "off_peak_price",
            "export_price",
        ]
        for key in required:
            assert key in DEFAULTS, f"Missing default: {key}"

    def test_batch_keys_present(self):
        required = [
            "solar_range",
            "inverter_range",
            "battery_range",
            "solar_cost_per_kw",
            "battery_cost_per_kwh",
            "discount_rate",
            "loan_rate",
            "loan_years",
        ]
        for key in required:
            assert key in DEFAULTS, f"Missing default: {key}"

    def test_transmission_block_keys(self):
        for i in range(1, 6):
            assert f"tb{i}" in DEFAULTS
            assert f"pf{i}" in DEFAULTS

    def test_default_values_reasonable(self):
        assert 0 < DEFAULTS["inverter_power"] <= DEFAULTS["solar_power"]
        assert DEFAULTS["battery_capacity"] >= 0
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

    cons = pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-15 00:15:00", periods=96, freq="15min").strftime(
                "%d. %m. %Y %H:%M:%S"
            ),
            "energy_kwh": [0.5] * 96,
            "power_kw": [2.0] * 96,
            "transmission_block": [3] * 96,
            "extra": [""] * 96,
        }
    )
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

    async def test_batch_tab_is_first_and_default(self, user):
        await user.open("/")
        # find() returns a set, so read the order from the tab bar's own children
        tab_bar = next(
            t
            for t in user.find(ui.tabs).elements
            if any(c.props.get("name") == "Batch Analysis" for c in t.default_slot.children)
        )
        tabs = [c.props["name"] for c in tab_bar.default_slot.children]
        assert tabs[:2] == ["Batch Analysis", "Single Scenario"]
        await user.should_see("Run Analysis")
        await user.should_not_see("Run Simulation")

    async def test_single_tab_has_system_inputs(self, user):
        await user.open("/")
        user.find("Single Scenario").click()
        await user.should_see("Configuration")
        await user.should_see("Run Simulation")

    async def test_single_tab_has_parameter_sections(self, user):
        await user.open("/")
        await user.should_see("Electricity Pricing")
        await user.should_see("Transmission Costs")
        await user.should_see("Power Smoothing")
        await user.should_see("Heating Load")

    async def test_pricing_has_export_mode(self, user):
        await user.open("/")
        await user.should_see("Export pricing")


class TestExportPrices:
    def test_fixed_is_default(self):
        assert DEFAULTS["export_price_mode"] == gui.EXPORT_MODE_FIXED
        assert gui._export_prices({})[0] is None

    def test_fixed_mode_uses_no_file(self):
        assert gui._export_prices({"export_price_mode": gui.EXPORT_MODE_FIXED})[0] is None

    def test_uploaded_file_wins_with_auto_alignment(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui, "_upload_dir", tmp_path / "uploads")
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "export_prices.xlsx").write_bytes(b"x")
        found, alignment = gui._export_prices({"export_price_mode": gui.EXPORT_MODE_MARKET})
        assert found == str(Path("data") / "export_prices.xlsx")
        assert alignment == gui.ALIGN_AUTO

    def test_cache_used_by_default_with_calendar_alignment(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui, "_upload_dir", tmp_path / "uploads")
        cache = tmp_path / "data" / "price_cache"
        cache.mkdir(parents=True)
        (cache / "export_prices.csv").write_text("timestamp,price_eur_mwh\n")
        found, alignment = gui._export_prices({"export_price_mode": gui.EXPORT_MODE_MARKET})
        assert Path(found) == Path("data") / "price_cache" / "export_prices.csv"
        assert alignment == gui.ALIGN_CALENDAR

    def test_no_cache_and_offline_raises(self, tmp_path, monkeypatch):
        from solar_battery_simulator import price_cache

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui, "_upload_dir", tmp_path / "uploads")

        def offline(_url, **_kwargs):
            raise OSError("network down")

        monkeypatch.setattr(price_cache, "_get_json", offline)
        with pytest.raises(FileNotFoundError, match="use fixed export pricing"):
            gui._export_prices({"export_price_mode": gui.EXPORT_MODE_MARKET})

    def test_cache_status_text(self):
        ok = {"path": "p", "start": "2025-11-23", "end": "2026-11-22", "days": 365, "errors": []}
        assert "365 days" in gui._price_cache_status_text(ok)
        down = {"errors": ["download failed"], "fetched": [], "kept": [], "pruned": []}
        assert "unavailable" in gui._price_cache_status_text(down)


async def test_batch_financial_has_loan_switch(user):
    from nicegui import ui

    await user.open("/")
    await user.should_see("Use loan")  # Batch is the default tab
    user.find(kind=ui.tabs).elements.pop().value = gui.SINGLE_TAB
    await user.should_not_see("Use loan")  # Financial is batch-only


class TestSharedSidebar:
    """Both tabs use one sidebar; only the tab-specific sections switch."""

    @staticmethod
    def _expansions(user):
        from nicegui import ui

        return {e.text: e for e in user.client.layout.descendants() if isinstance(e, ui.expansion)}

    async def test_common_inputs_exist_once(self, user):
        from nicegui import ui

        await user.open("/")
        labels = [n.props.get("label") for n in user.find(kind=ui.number).elements]
        for label in ("Peak price", "OVE-SPTE fee", "Heating energy", "Reference power"):
            assert labels.count(label) == 1

    async def test_tab_specific_sections_switch(self, user):
        from nicegui import ui

        await user.open("/")
        sections = self._expansions(user)
        # Batch is the default tab
        assert not sections["System Specs"].visible
        assert sections["Parameter Ranges"].visible
        assert sections["Battery"].visible
        assert sections["Financial"].visible and sections["Equipment Costs"].visible
        assert sections["Electricity Pricing"].visible

        tabs = user.find(kind=ui.tabs).elements.pop()
        tabs.value = gui.SINGLE_TAB
        assert sections["System Specs"].visible
        assert not sections["Parameter Ranges"].visible
        assert not sections["Battery"].visible
        assert not sections["Financial"].visible
        assert not sections["Equipment Costs"].visible
        assert sections["Electricity Pricing"].visible

    async def test_battery_inputs_bound_to_system_specs(self, user):
        import asyncio

        from nicegui import ui

        await user.open("/")
        rates = [
            n
            for n in user.client.layout.descendants()
            if isinstance(n, ui.number) and n.props.get("label") == "C-rate"
        ]
        assert len(rates) == 2
        rates[1].value = 0.7
        for _ in range(20):
            if rates[0].value == 0.7:
                break
            await asyncio.sleep(0.05)
        assert rates[0].value == 0.7
