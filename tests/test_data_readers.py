"""Tests for data_readers: CSV / XLSX / XLS input, legacy and smart-meter layouts."""

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_readers import (
    detect_consumption_format,
    load_consumption,
    load_production,
    read_table,
)
from power_flow_simulator import PowerFlowSimulator

FIXTURES = Path(__file__).parent / "fixtures"

N = 8
SM_TIMES = pd.date_range("2025-01-01 00:15", periods=N, freq="15min")
SM_ENERGY = np.round(0.1 + 0.01 * np.arange(N), 4)
LEGACY_TIMES = pd.date_range("2024-01-15 00:15", periods=N, freq="15min")
LEGACY_ENERGY = np.round(0.5 + 0.1 * np.arange(N), 4)
PROD_KW = np.round(1.5 * np.arange(N), 3)


def smart_meter_frame(timestamps_as_text: bool = True) -> pd.DataFrame:
    """Synthetic smart-meter export (fake meter id / GSRN), same layout as the portal."""
    times = SM_TIMES.strftime("%Y-%m-%dT%H:%M") if timestamps_as_text else SM_TIMES
    return pd.DataFrame(
        {
            "Merilno mesto": ["0-000000"] * N,
            "GSRN MM": ["000000000000000000"] * N,
            "Časovna značka": times,
            "Leto": [2025] * N,
            "Mesec": [1] * N,
            "Energija A+": SM_ENERGY,
            "Energija A-": [0.0] * N,
            "Energija R+": [0.01] * N,
            "Energija R-": [0.0] * N,
            "P+ Prejeta delovna moč": SM_ENERGY * 4,
            "P- Oddana delovna moč": [0.0] * N,
            "Q+ Prejeta jalova moč": [0.04] * N,
            "Q- Oddana jalova moč": [0.0] * N,
            "Blok": [4] * N,
            "Blok za dogovorjeno moč": [4] * N,
            "Dogovorjena moč": [0.0] * N,
            "Status odčitka A+": [None] * N,
            "Status odčitka A-": [None] * N,
            "Status odčitka R+": [None] * N,
            "Status odčitka R-": [None] * N,
            "Obračunski podatek": ["Da"] * N,
        }
    )


def legacy_frame(timestamps_as_text: bool = True) -> pd.DataFrame:
    times = LEGACY_TIMES.strftime("%d. %m. %Y %H:%M:%S") if timestamps_as_text else LEGACY_TIMES
    return pd.DataFrame(
        {
            "Time": times,
            "Energy (kWh)": LEGACY_ENERGY,
            "Power (kW)": LEGACY_ENERGY * 4,
            "Transmission fee block": [3] * N,
        }
    )


def production_frame() -> pd.DataFrame:
    return pd.DataFrame({"Time stamp": range(1, N + 1), "System power generated | (kW)": PROD_KW})


def write_xlsx(path: Path, df: pd.DataFrame, sheet: str = "Sheet1", legend: bool = False):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet, index=False)
        if legend:
            pd.DataFrame({"Koda statusa odčitka": ["X"], "Opis": ["legend"]}).to_excel(
                writer, sheet_name="Šifrant", index=False
            )
    return path


def assert_smart_meter_result(df: pd.DataFrame) -> None:
    assert list(df.columns) == [
        "datetime",
        "consumption_kwh",
        "consumption_kw",
        "transmission_block",
    ]
    assert len(df) == N
    pd.testing.assert_series_equal(
        df["datetime"], pd.Series(SM_TIMES.astype("datetime64[ns]"), name="datetime")
    )
    np.testing.assert_allclose(df["consumption_kwh"], SM_ENERGY)
    np.testing.assert_allclose(df["consumption_kw"], SM_ENERGY * 4)
    assert df["transmission_block"].tolist() == [4] * N
    assert df["transmission_block"].dtype.kind == "i"


def assert_legacy_result(df: pd.DataFrame) -> None:
    assert len(df) == N
    assert df["datetime"].tolist() == list(LEGACY_TIMES)
    np.testing.assert_allclose(df["consumption_kwh"], LEGACY_ENERGY)
    np.testing.assert_allclose(df["consumption_kw"], LEGACY_ENERGY * 4)
    assert df["transmission_block"].tolist() == [3] * N


# ---------------------------------------------------------------------------
# read_table
# ---------------------------------------------------------------------------


class TestReadTable:
    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_table(tmp_path / "nope.csv")

    @pytest.mark.parametrize("ext", [".txt", ".json", ".ods", ""])
    def test_unsupported_extension(self, tmp_path, ext):
        path = tmp_path / f"data{ext}"
        path.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="Unsupported file type"):
            read_table(path)

    def test_extension_is_case_insensitive(self, tmp_path):
        path = write_xlsx(tmp_path / "DATA.XLSX", production_frame())
        assert len(read_table(path)) == N

    def test_header_only_csv_raises(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("a,b\n")
        with pytest.raises(ValueError, match="No data rows"):
            read_table(path)

    def test_blank_rows_and_trailing_separator_dropped(self, tmp_path):
        path = tmp_path / "c.csv"
        path.write_text("a,b,\n1,2,\n,,\n3,4,\n")
        df = read_table(path)
        assert list(df.columns) == ["a", "b"]
        assert df["a"].tolist() == [1, 3]

    def test_named_empty_column_is_kept(self, tmp_path):
        """Only unnamed empty columns are dropped, so positional layouts don't shift."""
        path = tmp_path / "c.csv"
        path.write_text("a,empty,b\n1,,2\n3,,4\n")
        assert list(read_table(path).columns) == ["a", "empty", "b"]

    def test_semicolon_csv_with_decimal_comma(self, tmp_path):
        path = tmp_path / "eu.csv"
        path.write_text("id;kw\n1;0,5\n2;1,25\n")
        df = read_table(path)
        assert df["kw"].tolist() == [0.5, 1.25]

    def test_tab_separated_csv(self, tmp_path):
        path = tmp_path / "tab.csv"
        path.write_text("id\tkw\n1\t0.5\n2\t1.5\n")
        assert read_table(path)["kw"].tolist() == [0.5, 1.5]

    def test_utf8_bom(self, tmp_path):
        path = tmp_path / "bom.csv"
        path.write_bytes("Časovna značka,x\n1,2\n".encode("utf-8-sig"))
        assert list(read_table(path).columns) == ["Časovna značka", "x"]

    def test_cp1250_encoding(self, tmp_path):
        path = tmp_path / "win.csv"
        path.write_bytes("Časovna značka,x\n1,2\n".encode("cp1250"))
        assert list(read_table(path).columns) == ["Časovna značka", "x"]

    def test_xlsx_reads_first_sheet_only(self, tmp_path):
        path = write_xlsx(tmp_path / "sm.xlsx", smart_meter_frame(), sheet="0-000000", legend=True)
        df = read_table(path)
        assert "Časovna značka" in df.columns
        assert len(df) == N

    def test_xls(self):
        df = read_table(FIXTURES / "production.xls")
        assert len(df) == N


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_smart_meter(self):
        assert detect_consumption_format(smart_meter_frame()) == "smart_meter"

    def test_legacy(self):
        assert detect_consumption_format(legacy_frame()) == "legacy"

    def test_header_matching_ignores_case_whitespace_and_unicode_form(self):
        df = smart_meter_frame().rename(
            columns={
                "Časovna značka": unicodedata.normalize("NFD", "  ČASOVNA ZNAČKA "),
                "Blok": "BLOK",
            }
        )
        assert detect_consumption_format(df) == "smart_meter"

    def test_unrecognised(self):
        with pytest.raises(ValueError, match="Unrecognised consumption file layout"):
            detect_consumption_format(pd.DataFrame({"a": [1], "b": [2]}))


# ---------------------------------------------------------------------------
# load_consumption — smart-meter export
# ---------------------------------------------------------------------------


class TestSmartMeterConsumption:
    def test_csv(self, tmp_path):
        path = tmp_path / "sm.csv"
        smart_meter_frame().to_csv(path, index=False)
        assert_smart_meter_result(load_consumption(path))

    def test_xlsx_with_datetime_cells_and_legend_sheet(self, tmp_path):
        path = write_xlsx(
            tmp_path / "sm.xlsx",
            smart_meter_frame(timestamps_as_text=False),
            sheet="0-000000",
            legend=True,
        )
        assert_smart_meter_result(load_consumption(path))

    def test_xlsx_with_text_timestamps(self, tmp_path):
        path = write_xlsx(tmp_path / "sm.xlsx", smart_meter_frame())
        assert_smart_meter_result(load_consumption(path))

    def test_xls(self):
        assert_smart_meter_result(load_consumption(FIXTURES / "smart_meter.xls"))

    def test_csv_and_xlsx_give_identical_results(self, tmp_path):
        csv_path = tmp_path / "sm.csv"
        smart_meter_frame().to_csv(csv_path, index=False)
        xlsx_path = write_xlsx(tmp_path / "sm.xlsx", smart_meter_frame(timestamps_as_text=False))
        pd.testing.assert_frame_equal(load_consumption(csv_path), load_consumption(xlsx_path))

    def test_power_derived_from_energy_when_p_column_absent(self, tmp_path):
        path = tmp_path / "sm.csv"
        smart_meter_frame().drop(columns=["P+ Prejeta delovna moč"]).to_csv(path, index=False)
        df = load_consumption(path)
        np.testing.assert_allclose(df["consumption_kw"], SM_ENERGY * 4)

    def test_semicolon_decimal_comma_export(self, tmp_path):
        path = tmp_path / "sm.csv"
        smart_meter_frame().to_csv(path, index=False, sep=";", decimal=",")
        assert_smart_meter_result(load_consumption(path))

    def test_missing_energy_value_raises(self, tmp_path):
        df = smart_meter_frame()
        df.loc[3, "Energija A+"] = np.nan
        path = tmp_path / "sm.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match=r"Energija A\+.*1 missing.*row 4"):
            load_consumption(path)

    def test_bad_timestamp_raises(self, tmp_path):
        df = smart_meter_frame()
        df.loc[2, "Časovna značka"] = "not a date"
        path = tmp_path / "sm.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="timestamp"):
            load_consumption(path)

    def test_invalid_block_raises(self, tmp_path):
        df = smart_meter_frame()
        df.loc[0, "Blok"] = 6
        path = tmp_path / "sm.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match=r"block must be 1-5, found \[6\]"):
            load_consumption(path)


# ---------------------------------------------------------------------------
# load_consumption — legacy layout
# ---------------------------------------------------------------------------


class TestLegacyConsumption:
    def test_csv_with_trailing_empty_column(self, tmp_path):
        path = tmp_path / "consumption.csv"
        df = legacy_frame()
        df[""] = ""
        df.to_csv(path, index=False)
        assert_legacy_result(load_consumption(path))

    def test_xlsx_with_datetime_cells(self, tmp_path):
        path = write_xlsx(tmp_path / "consumption.xlsx", legacy_frame(timestamps_as_text=False))
        assert_legacy_result(load_consumption(path))

    def test_xlsx_with_text_timestamps(self, tmp_path):
        path = write_xlsx(tmp_path / "consumption.xlsx", legacy_frame())
        assert_legacy_result(load_consumption(path))

    def test_xls(self):
        assert_legacy_result(load_consumption(FIXTURES / "legacy_consumption.xls"))

    def test_wrong_timestamp_format_raises(self, tmp_path):
        df = legacy_frame()
        df["Time"] = LEGACY_TIMES.strftime("%Y/%m/%d %H:%M")
        path = tmp_path / "consumption.csv"
        df.to_csv(path, index=False)
        with pytest.raises(ValueError, match="timestamp"):
            load_consumption(path)


# ---------------------------------------------------------------------------
# load_production
# ---------------------------------------------------------------------------


class TestProduction:
    def check(self, df):
        assert list(df.columns) == ["timestamp_id", "solar_power_kw"]
        np.testing.assert_allclose(df["solar_power_kw"], PROD_KW)
        assert df["solar_power_kw"].dtype == float

    def test_csv(self, tmp_path):
        path = tmp_path / "production.csv"
        production_frame().to_csv(path, index=False)
        self.check(load_production(path))

    def test_xlsx(self, tmp_path):
        self.check(load_production(write_xlsx(tmp_path / "production.xlsx", production_frame())))

    def test_xls(self):
        self.check(load_production(FIXTURES / "production.xls"))

    def test_negative_values_preserved(self, tmp_path):
        """The shipped production.csv has small negative night values; don't alter them."""
        path = tmp_path / "production.csv"
        path.write_text('"Time stamp","kW"\n"1",-0.081608\n"2",0.5\n')
        assert load_production(path)["solar_power_kw"].tolist() == [-0.081608, 0.5]

    def test_single_column_raises(self, tmp_path):
        path = tmp_path / "production.csv"
        path.write_text("kw\n1\n2\n")
        with pytest.raises(ValueError, match="needs 2 columns"):
            load_production(path)

    def test_non_numeric_power_raises(self, tmp_path):
        path = tmp_path / "production.csv"
        path.write_text("id,kw\n1,0.5\n2,abc\n")
        with pytest.raises(ValueError, match="non-numeric"):
            load_production(path)


# ---------------------------------------------------------------------------
# Simulator integration
# ---------------------------------------------------------------------------


def _run(production_file, consumption_file):
    sim = PowerFlowSimulator(
        solar_panel_power_kw=640.6,
        inverter_power_kw=532.8,
        battery_capacity_kwh=1.0,
        production_file=str(production_file),
        consumption_file=str(consumption_file),
    )
    sim.load_and_align_data()
    sim.scale_solar_generation()
    sim.simulate_power_flows()
    return sim


class TestSimulatorIntegration:
    def test_excel_inputs_match_csv_inputs(self, tmp_path):
        prod_csv = tmp_path / "production.csv"
        production_frame().to_csv(prod_csv, index=False)
        cons_csv = tmp_path / "consumption.csv"
        smart_meter_frame().to_csv(cons_csv, index=False)
        cons_xlsx = write_xlsx(
            tmp_path / "consumption.xlsx", smart_meter_frame(timestamps_as_text=False)
        )

        from_csv = _run(prod_csv, cons_csv)
        from_excel = _run(FIXTURES / "production.xls", cons_xlsx)

        pd.testing.assert_frame_equal(from_csv.df, from_excel.df)
        pd.testing.assert_frame_equal(from_csv.simulation_results, from_excel.simulation_results)

    def test_smart_meter_data_flows_into_simulation(self):
        sim = _run(FIXTURES / "production.xls", FIXTURES / "smart_meter.xls")
        assert len(sim.df) == N
        np.testing.assert_allclose(sim.df["consumption_kwh"], SM_ENERGY)
        assert (sim.df["transmission_block"] == 4).all()
