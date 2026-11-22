"""Tests for Slovenian holiday calendar and batch-cache invalidation on input changes."""

import datetime as dt

import pandas as pd

from solar_battery_simulator.simulator import (
    MultiScenarioAnalyzer,
    PowerFlowSimulator,
    easter_sunday,
    slovenian_holidays,
)


def test_easter_sunday_known_dates():
    assert easter_sunday(2024) == dt.date(2024, 3, 31)
    assert easter_sunday(2025) == dt.date(2025, 4, 20)
    assert easter_sunday(2026) == dt.date(2026, 4, 5)
    assert easter_sunday(2019) == dt.date(2019, 4, 21)


def test_slovenian_holidays_2025():
    days = slovenian_holidays(2025)
    expected = {
        dt.date(2025, 1, 1),
        dt.date(2025, 1, 2),
        dt.date(2025, 2, 8),
        dt.date(2025, 4, 20),  # Easter Sunday
        dt.date(2025, 4, 21),  # Easter Monday
        dt.date(2025, 4, 27),
        dt.date(2025, 5, 1),
        dt.date(2025, 5, 2),
        dt.date(2025, 6, 8),  # Whit Sunday
        dt.date(2025, 6, 25),
        dt.date(2025, 8, 15),
        dt.date(2025, 10, 31),
        dt.date(2025, 11, 1),
        dt.date(2025, 12, 25),
        dt.date(2025, 12, 26),
    }
    assert days == expected


def _write_day(tmp_path, day: str, power_kw: float = 2.0):
    """Write one day of production + consumption for the given date."""
    prod = tmp_path / "production.csv"
    pd.DataFrame({"id": range(1, 97), "solar_power_kw": [0.0] * 96}).to_csv(prod, index=False)
    cons = tmp_path / "consumption.csv"
    pd.DataFrame(
        {
            "datetime": pd.date_range(f"{day} 00:15:00", periods=96, freq="15min").strftime(
                "%d. %m. %Y %H:%M:%S"
            ),
            "energy_kwh": [power_kw / 4] * 96,
            "power_kw": [power_kw] * 96,
            "transmission_block": [3] * 96,
            "extra": [""] * 96,
        }
    ).to_csv(cons, index=False)
    return str(prod), str(cons)


def test_2025_holiday_is_off_peak(tmp_path):
    """Easter Monday 2025 is a weekday holiday: no peak hours that day."""
    prod, cons = _write_day(tmp_path, "2025-04-21")
    sim = PowerFlowSimulator(production_file=prod, consumption_file=cons)
    sim.load_and_align_data()
    day = sim.df[sim.df["datetime"].dt.date == dt.date(2025, 4, 21)]
    assert day["is_holiday"].all()
    assert not day["is_peak_hour"].any()


def test_batch_cache_invalidated_when_file_replaced(tmp_path):
    """Replacing a data file under the same path must not return cached old results."""
    prod, cons = _write_day(tmp_path, "2025-03-12", power_kw=2.0)
    analyzer = MultiScenarioAnalyzer(
        solar_range=[10],
        inverter_range=[10],
        battery_range=[0],
        production_file=prod,
        consumption_file=cons,
    )
    args = (10.0, 10.0, 0.0, 0.0, 0.0, 0.9, 0.15, 0.10, 0.05)
    first = analyzer._cached_batch_simulation(*args)

    _write_day(tmp_path, "2025-03-12", power_kw=5.0)  # same path, new contents
    second = analyzer._cached_batch_simulation(*args)

    assert not first["simulation_failed"] and not second["simulation_failed"]
    assert (
        second["simulation_results"]["grid_import_kwh"].sum()
        > first["simulation_results"]["grid_import_kwh"].sum()
    )


def test_loaded_inputs_reloaded_when_file_replaced(tmp_path):
    """The in-memory input cache is keyed on file contents, not just the path."""
    prod, cons = _write_day(tmp_path, "2025-03-12", power_kw=2.0)
    first = PowerFlowSimulator(production_file=prod, consumption_file=cons)
    first.load_and_align_data()

    _write_day(tmp_path, "2025-03-12", power_kw=5.0)  # same path, new contents
    second = PowerFlowSimulator(production_file=prod, consumption_file=cons)
    second.load_and_align_data()

    assert first.df["consumption_kwh"].sum() == 96 * 0.5
    assert second.df["consumption_kwh"].sum() == 96 * 1.25


def test_cached_inputs_not_mutated_by_heating_load(tmp_path):
    """A simulator modifying its frame (heating load) must not leak into later loads."""
    prod, cons = _write_day(tmp_path, "2025-01-15")
    heated = PowerFlowSimulator(
        production_file=prod,
        consumption_file=cons,
        heating_config={"heating_kwh": 1000.0, "start_hour": 7, "end_hour": 0},
    )
    heated.load_and_align_data()
    plain = PowerFlowSimulator(production_file=prod, consumption_file=cons)
    plain.load_and_align_data()

    assert heated.df["consumption_kwh"].sum() > plain.df["consumption_kwh"].sum()
    assert plain.df["consumption_kwh"].sum() == 96 * 0.5


def test_load_warnings_repeat_on_cached_loads(tmp_path, capsys):
    """Loader output (e.g. the length-mismatch warning) is printed on every load."""
    prod, cons = _write_day(tmp_path, "2025-03-12")
    pd.DataFrame({"id": range(1, 98), "solar_power_kw": [0.0] * 97}).to_csv(prod, index=False)
    for _ in range(2):
        PowerFlowSimulator(production_file=prod, consumption_file=cons).load_and_align_data()
    assert capsys.readouterr().out.count("Warning: production has 97 intervals") == 2
