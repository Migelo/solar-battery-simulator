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

import numpy as np
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


# ---------------------------------------------------------------------------
# Export (market) prices
# ---------------------------------------------------------------------------
#
# Two layouts are accepted, detected from the header:
#
# * **HUPX download** — the CSV/XLSX from labs.hupx.hu (DAM aggregated trading
#   data), or rows from its JSON API: ``Delivery day`` plus ``Hour`` (60-min MTU,
#   until 30 Sep 2025) or ``Quarter hour`` (15-min MTU, from 1 Oct 2025) and
#   ``Price`` in EUR/MWh. Products are numbered by local (CET/CEST) clock
#   position; the repeated hour on the autumn DST day is suffixed ``B``
#   (``3B`` / ``9B``-``12B``) and averaged with its first occurrence. An optional
#   ``Region`` column is filtered to ``region``.
# * **Generic** — timestamp (start of interval, local time) then price. The price
#   is EUR/MWh unless its header mentions ``kwh``. Any regular resolution of
#   15 or 60 minutes works.

_PRICE_DAY = ("delivery day", "deliveryday")
_PRICE_QH = ("quarter hour", "productqh")
_PRICE_H = ("hour", "producth")
_PRICE = "price"
_PRICE_REGION = "region"

# Calendar alignment may fill at most this fraction of intervals (leap day,
# DST hour, a missing day) before the price file is rejected as not covering
# the simulation period.
MAX_PRICE_GAP_FRACTION = 0.01

PRICE_COLUMNS = ["interval_start", "price_eur_kwh"]


def _find_col(cols: dict[str, str], names: tuple[str, ...]) -> str | None:
    return next((cols[n] for n in names if n in cols), None)


def _product_number(series: pd.Series, path_name: str) -> pd.Series:
    """Clock position of HUPX product labels: ``"9"`` and the DST repeat ``"9B"`` -> 9."""
    text = series.astype(str).str.strip().str.upper()
    number = text.str.extract(r"^(\d+)\s*[AB]?$")[0]
    if number.isna().any():
        bad = text[number.isna()].iloc[0]
        raise ValueError(f"{path_name}: unrecognised HUPX product label {bad!r}")
    return number.astype(int)


def hupx_rows_to_prices(
    df: pd.DataFrame, region: str = "SI", source: str = "HUPX data"
) -> pd.DataFrame:
    """Convert HUPX DAM rows (download or API) to 15-min ``interval_start`` / EUR/kWh."""
    cols: dict[str, str] = {_norm(c): c for c in df.columns}
    day_col = _find_col(cols, _PRICE_DAY)
    qh_col = _find_col(cols, _PRICE_QH)
    h_col = _find_col(cols, _PRICE_H)
    if day_col is None or _PRICE not in cols or (qh_col is None and h_col is None):
        raise ValueError(
            f"{source}: expected HUPX columns 'Delivery day', 'Hour' or 'Quarter hour', "
            f"'Price'; got {list(df.columns)}"
        )
    if _PRICE_REGION in cols:
        regions = df[cols[_PRICE_REGION]].astype(str).str.strip().str.upper()
        df = df[regions == region.upper()]
        if df.empty:
            raise ValueError(f"{source}: no rows for region {region!r}")

    # DeliveryDay is the local delivery date written as midnight "UTC" (…T00:00:00Z).
    day = pd.to_datetime(
        df[day_col].astype(str).str.strip().str[:10], format="%Y-%m-%d", errors="coerce"
    )
    if day.isna().any():
        raise ValueError(f"{source}: could not parse delivery day {df[day_col].iloc[0]!r}")
    minutes = 15 if qh_col is not None else 60
    number = _product_number(df[qh_col if qh_col is not None else h_col], source)
    per_day = 24 * 60 // minutes
    valid = number.between(1, per_day)
    if not valid.all():
        print(f"Warning: {source}: dropping {int((~valid).sum())} product(s) outside 1-{per_day}")
    price = _to_numeric(df[cols[_PRICE]], str(cols[_PRICE]), Path(source)) / 1000.0
    out = pd.DataFrame(
        {
            "interval_start": day + pd.to_timedelta((number - 1) * minutes, unit="min"),
            "price_eur_kwh": price,
        }
    )[valid.to_numpy()]
    return _normalise_prices(out, minutes)


def _normalise_prices(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Average duplicate starts (autumn DST), sort, and expand to 15-min intervals."""
    df = df.groupby("interval_start", as_index=False).agg({"price_eur_kwh": "mean"})
    if minutes > 15:
        repeats = minutes // 15
        offsets = pd.to_timedelta(np.tile(np.arange(repeats) * 15, len(df)), unit="min")
        df = pd.DataFrame(
            {
                "interval_start": df["interval_start"].repeat(repeats).to_numpy() + offsets,
                "price_eur_kwh": df["price_eur_kwh"].repeat(repeats).to_numpy(),
            }
        )
    return df.sort_values("interval_start").reset_index(drop=True)[PRICE_COLUMNS]


def detect_price_format(df: pd.DataFrame) -> str:
    """Return ``"hupx"`` or ``"generic"`` for an export price table."""
    names = {_norm(c) for c in df.columns}
    if names & set(_PRICE_DAY) and names & set(_PRICE_QH + _PRICE_H):
        return "hupx"
    if len(df.columns) >= 2:
        return "generic"
    raise ValueError(
        "Unrecognised export price file: expected a HUPX download ('Delivery day', "
        f"'Hour'/'Quarter hour', 'Price') or 'timestamp, price' columns; got {list(df.columns)}"
    )


def load_export_prices(path: str | Path, region: str = "SI") -> pd.DataFrame:
    """Load an export price file into 15-min ``interval_start`` / ``price_eur_kwh`` rows.

    ``interval_start`` is naive local time (CET/CEST), the start of each 15-min interval.
    """
    path = Path(path)
    df = read_table(path)
    if detect_price_format(df) == "hupx":
        return hupx_rows_to_prices(df, region=region, source=path.name)

    ts = _to_datetime(df.iloc[:, 0], path)
    price_name = str(df.columns[1])
    price = _to_numeric(df.iloc[:, 1], price_name, path)
    if "kwh" not in _norm(price_name):
        price = price / 1000.0  # EUR/MWh -> EUR/kWh
    steps = ts.sort_values().diff().dropna().dt.total_seconds() / 60
    minutes = int(steps.mode().iloc[0]) if len(steps) else 15
    if minutes not in (15, 60):
        raise ValueError(f"{path.name}: price interval must be 15 or 60 minutes, got {minutes}")
    return _normalise_prices(pd.DataFrame({"interval_start": ts, "price_eur_kwh": price}), minutes)


ALIGN_AUTO = "auto"
ALIGN_CALENDAR = "calendar"


def align_export_prices(
    interval_end: pd.Series,
    prices: pd.DataFrame,
    interval_minutes: int = 15,
    method: str = ALIGN_AUTO,
) -> pd.Series:
    """Price (EUR/kWh) for each simulation interval.

    ``interval_end`` are the simulator's timestamps, which label the *end* of each
    interval (``00:15`` is 00:00-00:15). Matching is by exact local timestamp when the
    price file covers the simulated dates; otherwise by calendar position
    (month, day, time of day) so e.g. 2025 prices can be applied to 2024 consumption —
    if the price file spans several years the latest one wins. ``method="calendar"``
    always matches by calendar position (used for the rolling last-year price cache, so
    any consumption year is compared against the latest 365 days of prices). Up to
    ``MAX_PRICE_GAP_FRACTION`` of intervals may be gap-filled from the nearest price.

    Raises:
        ValueError: If the prices cover too little of the simulated period.
    """
    start = pd.to_datetime(interval_end) - pd.Timedelta(minutes=interval_minutes)
    by_time = prices.set_index("interval_start")["price_eur_kwh"]
    exact = start.map(by_time)

    def _calendar_key(ts: pd.Series) -> pd.Series:
        return pd.Series(ts.dt.strftime("%m-%d %H:%M"), index=ts.index)

    by_calendar = (
        prices.assign(key=_calendar_key(prices["interval_start"]))
        .drop_duplicates("key", keep="last")
        .set_index("key")["price_eur_kwh"]
    )
    calendar = _calendar_key(start).map(by_calendar)

    if method not in (ALIGN_AUTO, ALIGN_CALENDAR):
        raise ValueError(f"Unknown price alignment {method!r}")
    if method == ALIGN_AUTO and exact.notna().sum() >= calendar.notna().sum():
        aligned, used = exact, "exact timestamps"
    else:
        aligned, used = calendar, "calendar position (month/day/time of the price year)"

    missing = int(aligned.isna().sum())
    if missing > MAX_PRICE_GAP_FRACTION * len(aligned):
        first, last = prices["interval_start"].min(), prices["interval_start"].max()
        raise ValueError(
            f"Export prices cover only {len(aligned) - missing} of {len(aligned)} simulation "
            f"intervals (price data {first:%Y-%m-%d} to {last:%Y-%m-%d}, simulation "
            f"{start.min():%Y-%m-%d} to {start.max():%Y-%m-%d}). Provide prices for the "
            "whole simulated year."
        )
    print(f"Export prices aligned by {used}; {missing} interval(s) gap-filled")
    return aligned.ffill().bfill().astype(float).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Network tariffs (cenik omrežnin)
# ---------------------------------------------------------------------------
NETWORK_TARIFF_FILE = "cenikOmreznin.csv"
BUNDLED_TARIFF_FILE = Path(__file__).resolve().parent / NETWORK_TARIFF_FILE  # shipped default


def default_tariff_file() -> Path:
    """Tariff file in ``data/`` or the working directory, else the copy shipped in the package."""
    for candidate in (Path("data") / NETWORK_TARIFF_FILE, Path(NETWORK_TARIFF_FILE)):
        if candidate.exists():
            return candidate
    return BUNDLED_TARIFF_FILE


DEFAULT_TARIFF_GROUP = "0"
_TARIFF_GROUP = "skupina"
_TARIFF_BLOCK = "blok"
_TARIFF_POWER = "moč"
_TARIFF_ENERGY = "energij"


def _col_containing(df: pd.DataFrame, fragment: str) -> str | None:
    return next((c for c in df.columns if fragment in _norm(c)), None)


def load_network_tariffs(path: str | Path) -> dict[str, dict[str, dict[str, float]]]:
    """Load a network tariff table: user group x time block -> power and energy rates.

    Columns are matched by name (``Uporabniška skupina``, ``Časovni blok``,
    ``Tarifna postavka za moč [EUR/kW/mesec]``, ``Tarifna postavka za energijo
    [EUR/kWh]``), else read positionally in that order. Decimal commas are accepted.

    Returns ``{group: {"transmission_costs": {"block1": EUR/kWh, ...},
    "monthly_power_fees": {"block1": EUR/kW/month, ...}}}`` with groups in file order.

    Raises:
        ValueError: If a group does not list all five blocks exactly once.
    """
    path = Path(path)
    df = read_table(path)
    named = [
        _col_containing(df, f)
        for f in (_TARIFF_GROUP, _TARIFF_BLOCK, _TARIFF_POWER, _TARIFF_ENERGY)
    ]
    if None in named or len(set(named)) < 4:
        if len(df.columns) < 4:
            raise ValueError(
                f"{path.name}: expected columns user group, time block, power rate, "
                f"energy rate; got {list(df.columns)}"
            )
        named = list(df.columns[:4])
    group_col, block_col, power_col, energy_col = (str(c) for c in named)

    groups = df[group_col].astype(str).str.strip().str.removesuffix(".0")
    blocks = _validate_blocks(_to_numeric(df[block_col], block_col, path), path)
    power = _to_numeric(df[power_col], power_col, path)
    energy = _to_numeric(df[energy_col], energy_col, path)

    tariffs: dict[str, dict[str, dict[str, float]]] = {}
    for group in dict.fromkeys(groups):
        mask = groups == group
        group_blocks = blocks[mask].tolist()
        if sorted(group_blocks) != [1, 2, 3, 4, 5]:
            raise ValueError(
                f"{path.name}: user group {group} must list blocks 1-5 once each, "
                f"found {sorted(group_blocks)}"
            )
        tariffs[group] = {
            "transmission_costs": {
                f"block{b}": float(v) for b, v in zip(group_blocks, energy[mask], strict=True)
            },
            "monthly_power_fees": {
                f"block{b}": float(v) for b, v in zip(group_blocks, power[mask], strict=True)
            },
        }
    return tariffs
