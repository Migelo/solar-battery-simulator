"""Loan financing vs cash purchase in the batch economics."""

import pytest

from solar_battery_simulator.gui import _loan_years
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
    npv = a.calculate_npv_20_years(10_000, annual_savings=1_500, annual_maintenance_cost=200)
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
    assert a.calculate_npv_20_years(10_000, 1_500, 200) == pytest.approx(
        cash.calculate_npv_20_years(10_000, 1_500, 200)
    )


def test_loan_npv_unchanged():
    a = _analyzer(loan_rate=0.03, loan_years=10)
    payment = a.calculate_loan_payment(10_000)
    expected = sum((1_300 - (payment if y <= 10 else 0)) / 1.05**y for y in range(1, 21))
    assert a.calculate_npv_20_years(10_000, 1_500, 200) == pytest.approx(expected)


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


@pytest.mark.parametrize("loan_years", [10, 20, 25, 30])
def test_loan_matches_cash_when_loan_rate_equals_discount_rate(loan_years):
    """Financing at the discount rate is NPV-neutral whatever the loan term."""
    loan = _analyzer(loan_rate=0.05, loan_years=loan_years)
    cash = _analyzer(loan_years=0)
    assert loan.calculate_npv_20_years(10_000, 1_500, 200) == pytest.approx(
        cash.calculate_npv_20_years(10_000, 1_500, 200)
    )


def test_long_zero_rate_loan_matches_cash_without_discounting():
    loan = MultiScenarioAnalyzer(
        solar_range=[10],
        inverter_range=[10],
        battery_range=[0],
        discount_rate=0.0,
        loan_rate=0.0,
        loan_years=30,
    )
    cash = MultiScenarioAnalyzer(
        solar_range=[10], inverter_range=[10], battery_range=[0], discount_rate=0.0, loan_years=0
    )
    assert loan.calculate_npv_20_years(10_000, 1_500, 200) == pytest.approx(
        cash.calculate_npv_20_years(10_000, 1_500, 200)
    )


def test_long_expensive_loan_destroys_value_vs_cash():
    """Debt beyond the horizon is still a liability: r > d must lower NPV."""
    kwargs = {"solar_range": [10], "inverter_range": [10], "battery_range": [0]}
    loan = MultiScenarioAnalyzer(**kwargs, discount_rate=0.03, loan_rate=0.06, loan_years=30)
    cash = MultiScenarioAnalyzer(**kwargs, discount_rate=0.03, loan_years=0)
    assert loan.calculate_npv_20_years(10_000, 1_500, 200) < cash.calculate_npv_20_years(
        10_000, 1_500, 200
    )
