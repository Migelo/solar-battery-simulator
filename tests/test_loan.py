"""Loan financing vs cash purchase in the batch economics."""

import pytest

from solar_battery_simulator import simulator
from solar_battery_simulator.gui import _loan_years, _npv_years
from solar_battery_simulator.simulator import MultiScenarioAnalyzer


def _analyzer(**kwargs):
    return MultiScenarioAnalyzer(
        solar_range=[10], inverter_range=[10], battery_range=[0], discount_rate=0.05, **kwargs
    )


@pytest.mark.parametrize("loan_rate", [0.0, 0.03])
def test_no_loan_has_no_payment(loan_rate):
    assert _analyzer(loan_rate=loan_rate, loan_years=0).calculate_loan_payment(10_000) == 0


def test_no_loan_npv_pays_investment_upfront():
    a = _analyzer(loan_years=0)
    annuity = sum(1 / 1.05**y for y in range(1, 21))
    npv = a.calculate_npv(10_000, annual_savings=1_500, annual_maintenance_cost=200)
    assert npv == pytest.approx(-10_000 + 1_300 * annuity)


def test_zero_rate_loan_matches_cash_without_discounting():
    a = MultiScenarioAnalyzer(
        solar_range=[10],
        inverter_range=[10],
        battery_range=[0],
        discount_rate=0.0,
        loan_rate=0.0,
        loan_years=10,
    )
    cash = MultiScenarioAnalyzer(
        solar_range=[10], inverter_range=[10], battery_range=[0], discount_rate=0.0, loan_years=0
    )
    assert a.calculate_npv(10_000, 1_500, 200) == pytest.approx(
        cash.calculate_npv(10_000, 1_500, 200)
    )


def test_loan_npv_unchanged():
    a = _analyzer(loan_rate=0.03, loan_years=10)
    payment = a.calculate_loan_payment(10_000)
    expected = sum((1_300 - (payment if y <= 10 else 0)) / 1.05**y for y in range(1, 21))
    assert a.calculate_npv(10_000, 1_500, 200) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"use_loan": True, "loan_years": 10}, 10),
        ({"use_loan": False, "loan_years": 10}, 0),
        ({"use_loan": True, "loan_years": 0}, 0),
        ({"use_loan": True, "loan_years": None}, 0),
        ({"loan_years": 7}, 7),
    ],
)
def test_gui_loan_years(params, expected):
    assert _loan_years(params) == expected


@pytest.mark.parametrize(("loan_years", "npv_years"), [(10, 20), (20, 20), (25, 30), (30, 30)])
def test_loan_matches_cash_when_loan_rate_equals_discount_rate(loan_years, npv_years):
    """Financing at the discount rate is NPV-neutral whatever the loan term."""
    loan = _analyzer(loan_rate=0.05, loan_years=loan_years, npv_years=npv_years)
    cash = _analyzer(loan_years=0, npv_years=npv_years)
    assert loan.calculate_npv(10_000, 1_500, 200) == pytest.approx(
        cash.calculate_npv(10_000, 1_500, 200)
    )


def test_npv_horizon_is_configurable():
    a = _analyzer(loan_years=0, npv_years=25)
    annuity = sum(1 / 1.05**y for y in range(1, 26))
    assert a.calculate_npv(10_000, 1_500, 200) == pytest.approx(-10_000 + 1_300 * annuity)


def test_long_loan_within_long_horizon():
    a = _analyzer(loan_rate=0.03, loan_years=30, npv_years=30)
    payment = a.calculate_loan_payment(10_000)
    expected = sum((1_300 - payment) / 1.05**y for y in range(1, 31))
    assert a.calculate_npv(10_000, 1_500, 200) == pytest.approx(expected)


def test_loan_longer_than_horizon_rejected():
    with pytest.raises(ValueError, match="may not exceed npv_years"):
        _analyzer(loan_years=25, npv_years=20)


@pytest.mark.parametrize("npv_years", [0, -5])
def test_npv_years_must_be_positive(npv_years):
    with pytest.raises(ValueError, match="at least 1"):
        _analyzer(loan_years=0, npv_years=npv_years)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--loan-years", "25"], "may not exceed --npv-years"),
        (["--npv-years", "8"], "may not exceed --npv-years"),
        (["--npv-years", "0", "--loan-years", "0"], "at least 1"),
    ],
)
def test_cli_rejects_loan_beyond_horizon(argv, message, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["solar-sim", *argv])
    with pytest.raises(SystemExit):
        simulator.main()
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        ({"loan_years": 25, "npv_years": 20}, 20),
        ({"loan_years": 25, "npv_years": 30}, 25),
        ({"loan_years": 25}, 20),
    ],
)
def test_gui_loan_years_capped_at_horizon(params, expected):
    assert _loan_years(params) == expected


@pytest.mark.parametrize(("value", "expected"), [(25, 25), (25.0, 25), (None, 20), (-3, 1)])
def test_gui_npv_years(value, expected):
    assert _npv_years({"npv_years": value}) == expected


def test_long_zero_rate_loan_matches_cash_without_discounting():
    kwargs = {"solar_range": [10], "inverter_range": [10], "battery_range": [0], "npv_years": 30}
    loan = MultiScenarioAnalyzer(**kwargs, discount_rate=0.0, loan_rate=0.0, loan_years=30)
    cash = MultiScenarioAnalyzer(**kwargs, discount_rate=0.0, loan_years=0)
    assert loan.calculate_npv(10_000, 1_500, 200) == pytest.approx(
        cash.calculate_npv(10_000, 1_500, 200)
    )


def test_long_expensive_loan_destroys_value_vs_cash():
    kwargs = {"solar_range": [10], "inverter_range": [10], "battery_range": [0], "npv_years": 30}
    loan = MultiScenarioAnalyzer(**kwargs, discount_rate=0.03, loan_rate=0.06, loan_years=30)
    cash = MultiScenarioAnalyzer(**kwargs, discount_rate=0.03, loan_years=0)
    assert loan.calculate_npv(10_000, 1_500, 200) < cash.calculate_npv(10_000, 1_500, 200)


async def test_gui_npv_horizon_caps_loan_years(user):
    from nicegui import ui

    await user.open("/")

    def number(label):
        return next(
            e
            for e in user.client.layout.descendants()
            if isinstance(e, ui.number) and e.props.get("label") == label
        )

    horizon, loan = number("NPV yrs"), number("Loan yrs")
    assert (horizon.value, loan.value, loan.max) == (20, 10, 20)
    horizon.value = 30
    assert loan.max == 30 and loan.value == 10  # typing only moves the limit
    horizon.value = 6
    found = user.find(kind=ui.number)
    found.elements = {horizon}
    found.trigger("blur")
    assert (loan.max, loan.value) == (6, 6)
