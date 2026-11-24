"""Tests for batch preset export / import (JSON file + browser storage)."""

import json

import pytest

from solar_battery_simulator import gui
from solar_battery_simulator.gui import (
    PRESET_KEYS,
    apply_batch_preset,
    batch_preset_dict,
    dumps_batch_preset,
    loads_batch_preset,
    parse_batch_preset,
)


@pytest.fixture
def full_params():
    return {**gui.DEFAULTS, "tariff_group": "3"}


class TestPresetKeys:
    def test_covers_defaults_and_tariff_group(self):
        assert set(PRESET_KEYS) == set(gui.DEFAULTS) | {"tariff_group"}

    def test_excludes_data_files(self):
        assert "production_file" not in PRESET_KEYS
        assert "consumption_file" not in PRESET_KEYS
        assert "export_price_file" not in PRESET_KEYS


class TestRoundTrip:
    def test_dict_parse_round_trip(self, full_params):
        envelope = batch_preset_dict(full_params, "mine")
        assert envelope["app"] == "solar-battery-simulator"
        assert envelope["kind"] == "batch-preset"
        assert envelope["version"] == 1
        name, params = parse_batch_preset(json.loads(json.dumps(envelope)))
        assert name == "mine"
        assert params == {k: full_params[k] for k in PRESET_KEYS}

    def test_dumps_loads_bytes_round_trip(self, full_params):
        raw = dumps_batch_preset(full_params, "file-preset")
        assert isinstance(raw, bytes)
        name, params = loads_batch_preset(raw)
        assert name == "file-preset"
        assert params["solar_range"] == full_params["solar_range"]
        assert params["tariff_group"] == "3"

    def test_only_known_keys_saved(self, full_params):
        envelope = batch_preset_dict({**full_params, "bogus": 1}, "")
        assert "bogus" not in envelope["params"]

    def test_bare_params_mapping_tolerated(self):
        name, params = parse_batch_preset({"solar_range": "1,2", "nope": True})
        assert name == ""
        assert params == {"solar_range": "1,2"}

    def test_file_save_load(self, tmp_path, full_params):
        path = tmp_path / "preset.json"
        path.write_bytes(dumps_batch_preset(full_params, "on-disk"))
        assert loads_batch_preset(path.read_bytes())[0] == "on-disk"


class TestParseErrors:
    def test_rejects_non_object(self):
        with pytest.raises(ValueError):
            parse_batch_preset([1, 2])

    def test_rejects_empty_params(self):
        with pytest.raises(ValueError):
            parse_batch_preset({"params": {"bogus": 1}})

    def test_rejects_wrong_kind(self, full_params):
        envelope = batch_preset_dict(full_params)
        envelope["kind"] = "other"
        with pytest.raises(ValueError):
            parse_batch_preset(envelope)

    def test_rejects_unknown_version(self, full_params):
        envelope = batch_preset_dict(full_params)
        envelope["version"] = 999
        with pytest.raises(ValueError):
            parse_batch_preset(envelope)

    def test_rejects_invalid_json(self):
        with pytest.raises(ValueError):
            loads_batch_preset(b"{not json")


class TestApplyPreset:
    def test_sets_widgets_and_reports_keys(self):
        order = []

        class Widget:
            def __init__(self, key):
                self._key = key
                self._value = None

            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, v):
                order.append(self._key)
                self._value = v

        inputs = {key: Widget(key) for key in ("solar_range", "tb1", "tariff_group")}
        applied = apply_batch_preset(inputs, {"solar_range": "1,2", "tb1": 0.5, "unknown_key": 9})
        assert inputs["solar_range"].value == "1,2"
        assert inputs["tb1"].value == 0.5
        assert applied == ["solar_range", "tb1"]

    def test_tariff_group_applied_last(self, full_params):
        order = []

        class Widget:
            def __init__(self, key):
                self._key = key
                self._value = None

            @property
            def value(self):
                return self._value

            @value.setter
            def value(self, v):
                order.append(self._key)
                self._value = v

        inputs = {key: Widget(key) for key in full_params}
        preset = {"tb1": 0.5, "tariff_group": "3", "tb2": 0.6}
        applied = apply_batch_preset(inputs, preset)
        assert applied[-1] == "tariff_group"
        assert order[-1] == "tariff_group"


class TestPresetsSection:
    async def test_batch_tab_shows_presets(self, user):
        await user.open("/")
        await user.should_see("Batch Presets")
        await user.should_see("Save to file")
        await user.should_see("Load file")
