"""Readers for the simulator's input files.

Both inputs can be supplied as CSV, XLSX/XLSM (openpyxl) or legacy XLS (xlrd).

Consumption files are accepted in two layouts, detected from the header:

* **Smart-meter export** — the 15-minute export from the Slovenian distribution
  portal (columns ``Časovna značka``, ``Energija A+``, ``P+ Prejeta delovna moč``,
  ``Blok``, ...). In XLSX exports the data is on the first sheet; the second
  (``Šifrant``) sheet is a legend and is ignored.
* **Legacy** — the original ``consumption.csv`` layout, read positionally:
  timestamp (``D. M. YYYY HH:MM:SS``), energy kWh, power kW, transmission block,
  and an optional trailing empty column.

Production files are read positionally: an interval id column followed by the
baseline solar power in kW.
"""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path

import pandas as pd

CSV_EXTENSIONS = (".csv",)
EXCEL_EXTENSIONS = (".xlsx", ".xlsm", ".xls")
SUPPORTED_EXTENSIONS = CSV_EXTENSIONS + EXCEL_EXTENSIONS

LEGACY_DATETIME_FORMAT = "%d. %m. %Y %H:%M:%S"

# Smart-meter export column names (matched after normalisation, see _norm).
_SM_TIMESTAMP = "časovna značka"
_SM_ENERGY = "energija a+"
_SM_POWER = "p+ prejeta delovna moč"
_SM_BLOCK = "blok"

# Encodings tried in order for CSV files. cp1250 covers CSVs re-saved by a
# Slovenian-locale Excel; latin-1 is last because it never fails to decode.
_CSV_ENCODINGS = ("utf-8-sig", "cp1250", "latin-1")

CONSUMPTION_COLUMNS = ["datetime", "consumption_kwh", "consumption_kw", "transmission_block"]


def _norm(name: object) -> str:
    """Normalise a column name for matching (Unicode NFC, trimmed, lower-case)."""
    return unicodedata.normalize("NFC", str(name)).strip().lower()


def _sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t").delimiter
    except csv.Error:
        return ","


def _read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in _CSV_ENCODINGS:
        try:
            with open(path, encoding=encoding, newline="") as fh:
                sample = fh.read(8192)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        sep = _sniff_delimiter(sample)
        # Semicolon-separated files come from European-locale Excel, which
        # also writes decimal commas.
        decimal = "," if sep == ";" else "."
        return pd.read_csv(path, sep=sep, decimal=decimal, encoding=encoding)
    raise ValueError(f"Could not decode {path}: {last_error}")


def read_table(path: str | Path) -> pd.DataFrame:
    """Read the first table from a CSV, XLSX/XLSM or XLS file into a DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the extension is unsupported or the file has no rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    ext = path.suffix.lower()
    if ext in CSV_EXTENSIONS:
        df = _read_csv(path)
    elif ext in (".xlsx", ".xlsm"):
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    elif ext == ".xls":
        df = pd.read_excel(path, sheet_name=0, engine="xlrd")
    else:
        raise ValueError(
            f"Unsupported file type {ext!r} for {path.name}; "
            f"use one of: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Drop blank rows, and empty *unnamed* columns (trailing separators). Named
    # empty columns are kept so positional layouts never shift.
    df = df.dropna(how="all")
    unnamed_empty = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
    df = df.drop(columns=unnamed_empty).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"No data rows found in {path}")
    return df


def _to_numeric(series: pd.Series, column: str, path: Path) -> pd.Series:
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        # Tolerate decimal commas in text cells (e.g. "0,113").
        series = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    values = pd.to_numeric(series, errors="coerce")
    missing = values.isna()
    if missing.any():
        first = int(missing.to_numpy().argmax())
        raise ValueError(
            f"{path.name}: column {column!r} has {int(missing.sum())} missing or "
            f"non-numeric value(s); first at data row {first + 1}"
        )
    return values.astype(float)


def _to_datetime(series: pd.Series, path: Path, fmt: str | None = None) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        result = series
    else:
        text = series.astype(str).str.strip()
        result = pd.to_datetime(text, format=fmt, errors="coerce")
    bad = result.isna()
    if bad.any():
        first = int(bad.to_numpy().argmax())
        raise ValueError(
            f"{path.name}: {int(bad.sum())} timestamp(s) could not be parsed; "
            f"first at data row {first + 1}: {series.iloc[first]!r}"
        )
    return pd.to_datetime(result).astype("datetime64[ns]")


def _validate_blocks(blocks: pd.Series, path: Path) -> pd.Series:
    invalid = ~blocks.isin([1, 2, 3, 4, 5])
    if invalid.any():
        bad_values = [f"{v:g}" for v in sorted(blocks[invalid].unique().tolist())]
        bad_values_text = "[" + ", ".join(bad_values) + "]"
        raise ValueError(f"{path.name}: transmission block must be 1-5, found {bad_values_text}")
    return blocks.astype(int)


def detect_consumption_format(df: pd.DataFrame) -> str:
    """Return ``"smart_meter"`` or ``"legacy"`` for a consumption table."""
    names = {_norm(c) for c in df.columns}
    if {_SM_TIMESTAMP, _SM_ENERGY, _SM_BLOCK} <= names:
        return "smart_meter"
    if len(df.columns) >= 4:
        return "legacy"
    raise ValueError(
        "Unrecognised consumption file layout: expected a smart-meter export "
        f"('Časovna značka', 'Energija A+', 'Blok', ...) or the legacy 4-column "
        f"layout, got columns {list(df.columns)}"
    )


def load_consumption(path: str | Path) -> pd.DataFrame:
    """Load a consumption file into the simulator's standard columns.

    Returns:
        DataFrame with ``datetime``, ``consumption_kwh``, ``consumption_kw`` and
        ``transmission_block`` (int 1-5), one row per 15-minute interval.
    """
    path = Path(path)
    df = read_table(path)
    fmt = detect_consumption_format(df)

    if fmt == "smart_meter":
        cols = {_norm(c): c for c in df.columns}
        energy = _to_numeric(df[cols[_SM_ENERGY]], cols[_SM_ENERGY], path)
        if _SM_POWER in cols:
            power = _to_numeric(df[cols[_SM_POWER]], cols[_SM_POWER], path)
        else:
            power = energy * 4.0  # 15-min interval energy -> average kW
        out = pd.DataFrame(
            {
                "datetime": _to_datetime(df[cols[_SM_TIMESTAMP]], path),
                "consumption_kwh": energy,
                "consumption_kw": power,
                "transmission_block": _validate_blocks(
                    _to_numeric(df[cols[_SM_BLOCK]], cols[_SM_BLOCK], path), path
                ),
            }
        )
    else:
        raw = df.iloc[:, :4]
        names = list(raw.columns)
        out = pd.DataFrame(
            {
                "datetime": _to_datetime(raw.iloc[:, 0], path, LEGACY_DATETIME_FORMAT),
                "consumption_kwh": _to_numeric(raw.iloc[:, 1], str(names[1]), path),
                "consumption_kw": _to_numeric(raw.iloc[:, 2], str(names[2]), path),
                "transmission_block": _validate_blocks(
                    _to_numeric(raw.iloc[:, 3], str(names[3]), path), path
                ),
            }
        )

    return out.reset_index(drop=True)


def load_production(path: str | Path) -> pd.DataFrame:
    """Load a production file: interval id column, then baseline solar power (kW).

    Returns:
        DataFrame with ``timestamp_id`` and ``solar_power_kw`` columns.
    """
    path = Path(path)
    df = read_table(path)
    if len(df.columns) < 2:
        raise ValueError(
            f"{path.name}: production file needs 2 columns (interval id, solar power kW), "
            f"got {list(df.columns)}"
        )
    return pd.DataFrame(
        {
            "timestamp_id": df.iloc[:, 0].to_numpy(),
            "solar_power_kw": _to_numeric(df.iloc[:, 1], str(df.columns[1]), path),
        }
    ).reset_index(drop=True)
