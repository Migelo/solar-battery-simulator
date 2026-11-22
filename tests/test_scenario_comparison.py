"""Scenario Comparison table: energy/cost totals and Slovenian-locale exports."""

import io

import openpyxl
import pandas as pd
import pytest

from solar_battery_simulator.gui import (
    COMPARISON_COLUMNS,
    comparison_csv_bytes,
    comparison_dataframe,
    comparison_table_rows,
    comparison_xlsx_bytes,
)
from solar_battery_simulator.simulator import MultiScenarioAnalyzer

NEW_COLUMNS = [
    "Solar produced (kWh)",
    "Exported (kWh)",
    "Imported (kWh)",
    "Consumed (kWh)",
    "Variable cost (€/yr)",
    "Fixed cost (€/yr)",
]


@pytest.fixture
def batch_results(minimal_production_csv, minimal_consumption_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep the joblib .cache out of the repo
    analyzer = MultiScenarioAnalyzer(
        solar_range=[5, 10],
        inverter_range=[5],
        battery_range=[0, 10],
        production_file=str(minimal_production_csv),
        consumption_file=str(minimal_consumption_csv),
    )
    analyzer.generate_scenarios()
    analyzer.run_all_scenarios()
    return analyzer.results


def test_new_columns_present(batch_results):
    df = comparison_dataframe(batch_results)
    for col in NEW_COLUMNS:
        assert col in df.columns
    assert list(df.columns) == [label for label, _, _ in COMPARISON_COLUMNS]
    assert df["Savings (€/yr)"].is_monotonic_decreasing


def test_totals_are_consistent(batch_results):
    for r in batch_results:
        # Net system cost = per-kWh import cost - export revenue + fixed fees
        assert r["solar_battery_cost"] == pytest.approx(
            r["variable_energy_cost"] - r["export_revenue"] + r["fixed_cost"]
        )
        assert r["fixed_cost"] == pytest.approx(r["power_fee_cost"] + r["ove_spte_cost"])
        assert r["variable_energy_cost"] > 0
    df = comparison_dataframe(batch_results)
    # Consumption is the same in every scenario (sanity check column)
    assert df["Consumed (kWh)"].nunique() == 1
    assert (df["Solar produced (kWh)"] > 0).all()


def test_table_rows_formatted(batch_results):
    rows = comparison_table_rows(comparison_dataframe(batch_results))
    assert len(rows) == len(batch_results)
    assert isinstance(rows[0]["Imported (kWh)"], str)


def test_csv_slovenian_locale():
    df = pd.DataFrame([{label: 1234.5 for label, _, _ in COMPARISON_COLUMNS}])
    raw = comparison_csv_bytes(df)
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM so Excel reads UTF-8 (€)
    header, line = raw.decode("utf-8-sig").splitlines()
    assert header.split(";")[0] == "Solar (kW)"
    assert line.split(";") == ["1234,5"] * len(COMPARISON_COLUMNS)
    back = pd.read_csv(io.BytesIO(raw), sep=";", decimal=",", encoding="utf-8-sig")
    assert back.iloc[0, 0] == pytest.approx(1234.5)


def test_xlsx_numeric_with_number_formats(batch_results):
    df = comparison_dataframe(batch_results)
    wb = openpyxl.load_workbook(io.BytesIO(comparison_xlsx_bytes(df)))
    ws = wb["Scenarios"]
    headers = [c.value for c in ws[1]]
    assert headers == list(df.columns)
    col = headers.index("Imported (kWh)") + 1
    cell = ws.cell(row=2, column=col)
    assert isinstance(cell.value, float)
    assert cell.value == pytest.approx(df.iloc[0]["Imported (kWh)"])
    assert cell.number_format == "[$-424]#,##0"
    assert ws.cell(row=2, column=headers.index("Payback (yr)") + 1).number_format == (
        "[$-424]#,##0.0"
    )
    # Percent stored as a fraction in a Slovenian "45,3 %" format
    pct = ws.cell(row=2, column=headers.index("Self-Suff (%)") + 1)
    assert pct.number_format == "[$-424]0.0\\ %"
    assert pct.value == pytest.approx(df.iloc[0]["Self-Suff (%)"] / 100)
    assert wb.properties.language == "sl-SI"
    # Every data cell carries the Slovenian locale tag
    for row in ws.iter_rows(min_row=2):
        assert all(c.number_format.startswith("[$-424]") for c in row)
