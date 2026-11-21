"""Tests for the network tariff file (cenikOmreznin.csv) and its GUI / CLI wiring."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import gui
from data_readers import DEFAULT_TARIFF_GROUP, NETWORK_TARIFF_FILE, load_network_tariffs
from power_flow_simulator import (
    MANUAL_POWER_FEE_DEFAULTS,
    MANUAL_TRANSMISSION_DEFAULTS,
    _cli_network_tariffs,
    _manual_tariff_flags_error,
)

REPO_TARIFFS = Path(__file__).parent.parent / NETWORK_TARIFF_FILE
BLOCKS = [f"block{i}" for i in range(1, 6)]


def _write(path: Path, rows: list[tuple], sep: str = ";", decimal: str = ",") -> Path:
    df = pd.DataFrame(
        rows,
        columns=[
            "Uporabniška skupina",
            "Časovni blok",
            "Tarifna postavka za moč [EUR/kW/mesec]",
            "Tarifna postavka za energijo [EUR/kWh]",
        ],
    )
    df.to_csv(path, sep=sep, decimal=decimal, index=False)
    return path


def _group_rows(group, power_base=1.0, energy_base=0.01):
    return [(group, b, power_base / b, energy_base + b / 1000) for b in range(1, 6)]


class TestLoadNetworkTariffs:
    def test_repository_file(self):
        tariffs = load_network_tariffs(REPO_TARIFFS)
        assert list(tariffs) == ["0", "1", "2", "3", "4"]
        g0 = tariffs["0"]
        assert g0["transmission_costs"]["block1"] == pytest.approx(0.02217)
        assert g0["monthly_power_fees"]["block1"] == pytest.approx(2.67611)
        assert g0["monthly_power_fees"]["block5"] == pytest.approx(0.00245)
        for rates in tariffs.values():
            assert list(rates["transmission_costs"]) == BLOCKS
            assert list(rates["monthly_power_fees"]) == BLOCKS

    def test_default_group_is_zero(self):
        assert DEFAULT_TARIFF_GROUP == "0"
        assert DEFAULT_TARIFF_GROUP in load_network_tariffs(REPO_TARIFFS)

    def test_comma_separated_decimal_point(self, tmp_path):
        path = _write(tmp_path / "t.csv", _group_rows(7), sep=",", decimal=".")
        tariffs = load_network_tariffs(path)
        assert tariffs["7"]["monthly_power_fees"]["block2"] == pytest.approx(0.5)
        assert tariffs["7"]["transmission_costs"]["block3"] == pytest.approx(0.013)

    def test_xlsx(self, tmp_path):
        rows = _group_rows(0) + _group_rows(1, power_base=2.0)
        df = pd.DataFrame(rows, columns=["Uporabniška skupina", "Časovni blok", "moč", "energija"])
        path = tmp_path / "t.xlsx"
        df.to_excel(path, index=False)
        tariffs = load_network_tariffs(path)
        assert tariffs["1"]["monthly_power_fees"]["block1"] == pytest.approx(2.0)

    def test_blocks_out_of_order(self, tmp_path):
        rows = list(reversed(_group_rows(0)))
        tariffs = load_network_tariffs(_write(tmp_path / "t.csv", rows))
        assert tariffs["0"]["monthly_power_fees"]["block1"] == pytest.approx(1.0)

    def test_positional_fallback(self, tmp_path):
        df = pd.DataFrame(_group_rows(3), columns=["a", "b", "c", "d"])
        path = tmp_path / "t.csv"
        df.to_csv(path, index=False)
        assert load_network_tariffs(path)["3"]["transmission_costs"]["block5"] == 0.015

    def test_missing_block_raises(self, tmp_path):
        path = _write(tmp_path / "t.csv", _group_rows(0)[:4])
        with pytest.raises(ValueError, match="blocks 1-5"):
            load_network_tariffs(path)

    def test_non_numeric_rate_raises(self, tmp_path):
        rows = _group_rows(0)
        rows[0] = (0, 1, "n/a", 0.01)
        with pytest.raises(ValueError, match="non-numeric"):
            load_network_tariffs(_write(tmp_path / "t.csv", rows))


class TestGuiTariffs:
    def test_resolve_group_from_file(self):
        tc, pf = gui.resolve_network_tariffs({"tariff_group": "2"})
        rates = load_network_tariffs(REPO_TARIFFS)["2"]
        assert tc == rates["transmission_costs"]
        assert pf == rates["monthly_power_fees"]

    def test_resolve_manual_uses_inputs(self):
        params = {"tariff_group": gui.TARIFF_MANUAL}
        params |= {f"tb{i}": i / 100 for i in range(1, 6)}
        params |= {f"pf{i}": float(i) for i in range(1, 6)}
        tc, pf = gui.resolve_network_tariffs(params)
        assert tc["block3"] == 0.03
        assert pf["block5"] == 5.0

    def test_resolve_unknown_group_raises(self):
        with pytest.raises(ValueError, match="not found"):
            gui.resolve_network_tariffs({"tariff_group": "99"})

    def test_uploaded_file_takes_precedence(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gui, "_upload_dir", tmp_path)
        _write(tmp_path / NETWORK_TARIFF_FILE, _group_rows("0", power_base=9.0))
        tc, pf = gui.resolve_network_tariffs({"tariff_group": "0"})
        assert pf["block1"] == pytest.approx(9.0)

    def test_falls_back_to_bundled_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(gui, "_upload_dir", tmp_path / "uploads")
        assert gui._tariff_file() == REPO_TARIFFS.resolve()

    def test_default_selection(self):
        tariffs = load_network_tariffs(REPO_TARIFFS)
        assert gui._default_tariff_group(tariffs) == "0"
        assert gui._default_tariff_group({"5": {}}) == "5"
        assert gui._default_tariff_group({}) == gui.TARIFF_MANUAL

    def test_options_keep_manual(self):
        options = gui._tariff_options(load_network_tariffs(REPO_TARIFFS))
        assert options[gui.TARIFF_MANUAL] == "Manual"
        assert options["0"] == "User group 0"

    async def test_dropdown_defaults_to_group_zero(self, user):
        await user.open("/")
        await user.should_see("Network tariff")
        selects = list(user.find("Network tariff").elements)
        assert selects and all(s.value == "0" for s in selects)
        tb1 = [n for n in user.find("Block 1").elements if getattr(n, "value", None) == 0.02217]
        assert tb1, "group 0 rates should be filled into the transmission inputs"


class TestCliTariffs:
    def _args(self, **kw):
        base = {f"transmission_block{i}": i / 100 for i in range(1, 6)}
        base |= {f"power_fee_block{i}": float(i) for i in range(1, 6)}
        base |= {"tariff_group": DEFAULT_TARIFF_GROUP, "tariff_file": str(REPO_TARIFFS)}
        return SimpleNamespace(**(base | kw))

    def test_default_group(self):
        tc, pf = _cli_network_tariffs(self._args())
        assert tc["block1"] == pytest.approx(0.02217)
        assert pf["block1"] == pytest.approx(2.67611)

    def test_manual(self):
        tc, pf = _cli_network_tariffs(self._args(tariff_group="manual"))
        assert tc["block2"] == 0.02
        assert pf["block4"] == 4.0

    def test_unknown_group_exits(self):
        with pytest.raises(SystemExit):
            _cli_network_tariffs(self._args(tariff_group="9"))


class TestManualTariffFlags:
    def _args(self, group=DEFAULT_TARIFF_GROUP, **kw):
        base = {f"transmission_block{i}": None for i in range(1, 6)}
        base |= {f"power_fee_block{i}": None for i in range(1, 6)}
        return SimpleNamespace(**(base | {"tariff_group": group} | kw))

    def test_no_flags_with_group_ok(self):
        assert _manual_tariff_flags_error(self._args()) is None

    def test_flag_without_manual_is_error(self):
        err = _manual_tariff_flags_error(self._args(transmission_block2=0.02, power_fee_block1=3))
        assert err is not None
        assert "--transmission-block2" in err
        assert "--power-fee-block1" in err
        assert "--tariff-group manual" in err

    def test_manual_fills_missing_with_defaults(self):
        args = self._args(group="manual", transmission_block1=0.5)
        assert _manual_tariff_flags_error(args) is None
        tc, pf = _cli_network_tariffs(args)
        assert tc["block1"] == 0.5
        assert tc["block2"] == MANUAL_TRANSMISSION_DEFAULTS[1]
        assert pf["block1"] == MANUAL_POWER_FEE_DEFAULTS[0]

    def test_cli_rejects_flag_without_manual(self):
        result = subprocess.run(
            [sys.executable, "power_flow_simulator.py", "--transmission-block1", "0.02"],
            cwd=REPO_TARIFFS.parent,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2
        assert "requires --tariff-group manual" in result.stderr
