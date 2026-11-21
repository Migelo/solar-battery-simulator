"""Reference capacity of the production file (``reference_solar_kw`` / --reference-solar-power)."""

import pytest

import gui
from power_flow_simulator import (
    DEFAULT_REFERENCE_SOLAR_KW,
    MultiScenarioAnalyzer,
    PowerFlowSimulator,
)


def _scaled(prod, cons, reference_kw=None):
    kwargs = {} if reference_kw is None else {"reference_solar_kw": reference_kw}
    sim = PowerFlowSimulator(
        solar_panel_power_kw=10.0,
        inverter_power_kw=1000.0,  # no clipping
        production_file=str(prod),
        consumption_file=str(cons),
        **kwargs,
    )
    sim.load_and_align_data()
    sim.scale_solar_generation()
    return sim


def test_default_reference_is_640_6(minimal_production_csv, minimal_consumption_csv):
    sim = _scaled(minimal_production_csv, minimal_consumption_csv)
    assert sim.baseline_solar_kw == DEFAULT_REFERENCE_SOLAR_KW == 640.6


def test_scaling_uses_reference(sunny_production_csv, minimal_consumption_csv):
    default = _scaled(sunny_production_csv, minimal_consumption_csv)
    half = _scaled(sunny_production_csv, minimal_consumption_csv, DEFAULT_REFERENCE_SOLAR_KW / 2)
    baseline = half.df["baseline_solar_kw"]
    assert half.df["scaled_solar_kw"].sum() == pytest.approx(
        2 * default.df["scaled_solar_kw"].sum()
    )
    assert half.df["scaled_solar_kw"].max() == pytest.approx(
        baseline.max() * 10.0 / (DEFAULT_REFERENCE_SOLAR_KW / 2)
    )


@pytest.mark.parametrize("bad", [0, -5.0])
def test_non_positive_reference_rejected(bad):
    with pytest.raises(ValueError, match="reference_solar_kw"):
        PowerFlowSimulator(reference_solar_kw=bad)


def test_reference_reaches_batch_simulation(sunny_production_csv, minimal_consumption_csv):
    """The reference is part of the batch cache key: a change must not reuse cached results."""

    def exported(reference_kw):
        analyzer = MultiScenarioAnalyzer(
            solar_range=[20.0],
            inverter_range=[1000.0],
            battery_range=[0.0],
            production_file=str(sunny_production_csv),
            consumption_file=str(minimal_consumption_csv),
            reference_solar_kw=reference_kw,
        )
        result = analyzer._cached_batch_simulation(
            20.0, 1000.0, 0.0, 0.0, 0.0, 0.9, 0.15, 0.1, 0.05
        )
        return result["simulation_results"]["solar_generation_kwh"].sum()

    default, half = exported(DEFAULT_REFERENCE_SOLAR_KW), exported(DEFAULT_REFERENCE_SOLAR_KW / 2)
    assert default > 0
    assert half == pytest.approx(2 * default)


def test_cli_flag(monkeypatch):
    import power_flow_simulator

    captured = {}

    class Stop(Exception):
        pass

    def fake_init(_self, *_args, **kwargs):
        captured.update(kwargs)
        raise Stop

    monkeypatch.setattr(power_flow_simulator.PowerFlowSimulator, "__init__", fake_init)
    for argv, expected in [
        ([], DEFAULT_REFERENCE_SOLAR_KW),
        (["--reference-solar-power", "100"], 100.0),
    ]:
        monkeypatch.setattr("sys.argv", ["power_flow_simulator.py", *argv])
        with pytest.raises(Stop):
            power_flow_simulator.main()
        assert captured["reference_solar_kw"] == expected


def test_gui_default():
    assert gui.DEFAULTS["reference_solar_kw"] == DEFAULT_REFERENCE_SOLAR_KW


async def test_gui_shows_reference_input(user):
    await user.open("/")
    await user.should_see("Reference power")


async def test_gui_reference_single_input(user):
    """The sidebar is shared by both tabs, so there is one reference input."""
    from nicegui import ui

    await user.open("/")
    numbers = [
        n for n in user.find(kind=ui.number).elements if n.props.get("label") == "Reference power"
    ]
    assert len(numbers) == 1
