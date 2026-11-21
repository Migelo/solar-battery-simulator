"""One-off investment subsidies (SubsidyConfig) and their effect on batch economics."""

import argparse

import pytest

import gui
from power_flow_simulator import MultiScenarioAnalyzer, SubsidyConfig, _cli_subsidy


def _analyzer(subsidy=None, **kwargs) -> MultiScenarioAnalyzer:
    return MultiScenarioAnalyzer(
        solar_cost_per_kw=500,
        inverter_cost_per_kw=100,
        battery_cost_per_kwh=300,
        subsidy=subsidy,
        **kwargs,
    )


class TestSubsidyConfig:
    def test_default_is_off(self):
        cfg = SubsidyConfig()
        assert not cfg.active
        assert cfg.calculate(10, 8, 10, 10_000)["subsidy"] == 0

    def test_per_kw_solar_and_inverter(self):
        cfg = SubsidyConfig(
            solar_enabled=True, solar_eur_per_kw=100, inverter_enabled=True, inverter_eur_per_kw=50
        )
        s = cfg.calculate(10, 8, 0, 100_000)
        assert s["subsidy_solar"] == 1000
        assert s["subsidy_inverter"] == 400
        assert s["subsidy"] == 1400

    def test_switch_off_ignores_rate(self):
        cfg = SubsidyConfig(solar_enabled=False, solar_eur_per_kw=100)
        assert not cfg.active
        assert cfg.calculate(10, 8, 0, 100_000)["subsidy"] == 0

    @pytest.mark.parametrize(
        ("battery_kwh", "eligible"),
        [(4, False), (5, True), (10, True), (25, True), (26, False), (0, False)],
    )
    def test_battery_ratio_window_on_solar(self, battery_kwh, eligible):
        cfg = SubsidyConfig(battery_enabled=True, battery_eur_per_kwh=200)
        s = cfg.calculate(10, 8, battery_kwh, 100_000)
        assert s["subsidy_battery"] == (battery_kwh * 200 if eligible else 0)

    def test_battery_basis_inverter_and_sum(self):
        kw = {"solar_kw": 10, "inverter_kw": 4, "battery_kwh": 12}
        common = {"battery_enabled": True, "battery_eur_per_kwh": 100}
        # 12 / 10 = 1.2 -> ok; 12 / 4 = 3.0 -> too large; 12 / 14 = 0.86 -> ok
        assert SubsidyConfig(**common, battery_basis="solar").battery_eligible(**kw)
        assert not SubsidyConfig(**common, battery_basis="inverter").battery_eligible(**kw)
        assert SubsidyConfig(**common, battery_basis="sum").battery_eligible(**kw)

    def test_custom_ratio_bounds(self):
        cfg = SubsidyConfig(
            battery_enabled=True, battery_eur_per_kwh=100, battery_min_ratio=1, battery_max_ratio=2
        )
        assert not cfg.battery_eligible(10, 10, 9)
        assert cfg.battery_eligible(10, 10, 20)
        assert not cfg.battery_eligible(10, 10, 21)

    def test_invalid_basis(self):
        with pytest.raises(ValueError, match="battery_basis"):
            SubsidyConfig(battery_basis="battery")

    def test_cap_fraction(self):
        cfg = SubsidyConfig(
            solar_enabled=True, solar_eur_per_kw=400, cap_enabled=True, cap_fraction=0.3
        )
        s = cfg.calculate(10, 10, 0, 10_000)
        assert s["subsidy_solar"] == 4000  # uncapped part stays visible
        assert s["subsidy"] == pytest.approx(3000)

    def test_cap_not_binding(self):
        cfg = SubsidyConfig(
            solar_enabled=True, solar_eur_per_kw=100, cap_enabled=True, cap_fraction=0.5
        )
        assert cfg.calculate(10, 10, 0, 10_000)["subsidy"] == 1000

    def test_never_exceeds_investment(self):
        cfg = SubsidyConfig(solar_enabled=True, solar_eur_per_kw=10_000)
        assert cfg.calculate(10, 10, 0, 6000)["subsidy"] == 6000


class TestAnalyzerInvestment:
    def test_no_subsidy_unchanged(self):
        inv = _analyzer().calculate_investment(10, 8, 10)
        assert inv["gross_investment"] == 10 * 500 + 8 * 100 + 10 * 300
        assert inv["total_investment"] == inv["gross_investment"]
        assert inv["subsidy"] == 0

    def test_subsidy_reduces_total_investment(self):
        cfg = SubsidyConfig(
            solar_enabled=True,
            solar_eur_per_kw=100,
            battery_enabled=True,
            battery_eur_per_kwh=50,
        )
        inv = _analyzer(cfg).calculate_investment(10, 8, 10)
        assert inv["subsidy"] == 1000 + 500
        assert inv["total_investment"] == inv["gross_investment"] - 1500

    def test_subsidy_improves_npv_and_payback(
        self, minimal_production_csv, minimal_consumption_csv
    ):
        files = {
            "production_file": str(minimal_production_csv),
            "consumption_file": str(minimal_consumption_csv),
            "solar_range": [10],
            "inverter_range": [8],
            "battery_range": [10],
            "loan_years": 0,
        }
        cfg = SubsidyConfig(solar_enabled=True, solar_eur_per_kw=200)
        base, sub = _analyzer(**files), _analyzer(cfg, **files)
        for a in (base, sub):
            a.generate_scenarios()
            a.run_all_scenarios()
        b, s = base.results[0], sub.results[0]
        assert s["subsidy"] == 2000
        assert s["total_investment"] == b["total_investment"] - 2000
        # Cash purchase: NPV improves by exactly the subsidy (maintenance is on gross value)
        assert s["npv_20_years"] == pytest.approx(b["npv_20_years"] + 2000)
        assert s["annual_maintenance_cost"] == pytest.approx(b["annual_maintenance_cost"])
        if b["savings_vs_baseline"] > 0:
            assert s["payback_years"] < b["payback_years"]

    def test_subsidy_reduces_loan_payment(self):
        cfg = SubsidyConfig(solar_enabled=True, solar_eur_per_kw=200)
        base, sub = _analyzer(), _analyzer(cfg)
        b = base.calculate_loan_payment(base.calculate_investment(10, 8, 0)["total_investment"])
        s = sub.calculate_loan_payment(sub.calculate_investment(10, 8, 0)["total_investment"])
        assert s < b


def _cli_args(**overrides):
    args = {
        "subsidy_solar_per_kw": 0.0,
        "subsidy_inverter_per_kw": 0.0,
        "subsidy_battery_per_kwh": 0.0,
        "subsidy_battery_min_ratio": 0.5,
        "subsidy_battery_max_ratio": 2.5,
        "subsidy_battery_basis": "solar",
        "subsidy_cap": None,
    }
    return argparse.Namespace(**{**args, **overrides})


class TestCli:
    def test_defaults_off(self):
        assert not _cli_subsidy(_cli_args()).active

    def test_flags(self):
        cfg = _cli_subsidy(
            _cli_args(subsidy_battery_per_kwh=150, subsidy_battery_basis="sum", subsidy_cap=0.4)
        )
        assert cfg.battery_enabled and not cfg.solar_enabled
        assert cfg.battery_basis == "sum"
        assert cfg.cap_enabled and cfg.cap_fraction == 0.4


class TestGui:
    def test_master_switch_off(self):
        params = {**gui.DEFAULTS, "subsidy_solar_per_kw": 100}
        assert not gui._subsidy_config(params).active

    def test_config_from_params(self):
        params = {
            **gui.DEFAULTS,
            "use_subsidies": True,
            "subsidy_solar_per_kw": 100,
            "subsidy_battery_per_kwh": 50,
            "subsidy_battery_basis": "inverter",
            "subsidy_cap_enabled": True,
            "subsidy_cap_fraction": 0.25,
        }
        cfg = gui._subsidy_config(params)
        assert cfg.solar_enabled and cfg.solar_eur_per_kw == 100
        assert cfg.battery_enabled and cfg.battery_basis == "inverter"
        assert not cfg.inverter_enabled
        assert cfg.cap_enabled and cfg.cap_fraction == 0.25

    def test_comparison_has_subsidy_column(self):
        assert ("Subsidy (€)", "subsidy", 0) in gui.COMPARISON_COLUMNS


async def test_subsidies_section_batch_only(user):
    from nicegui import ui

    await user.open("/")
    await user.should_see("Apply subsidies")  # Batch is the default tab
    user.find(kind=ui.tabs).elements.pop().value = gui.SINGLE_TAB
    await user.should_not_see("Apply subsidies")
