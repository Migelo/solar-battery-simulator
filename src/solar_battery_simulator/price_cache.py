#!/usr/bin/env python3
"""Rolling one-year cache of day-ahead prices for market export pricing.

Prices come from the Energy-Charts API (Fraunhofer ISE, https://api.energy-charts.info,
CC BY 4.0; data from Bundesnetzagentur | SMARD.de), which serves the full history of the
Slovenian (SI) day-ahead price — the same SIPX/BSP price that HUPX publishes — without
an account.

Layout (under ``data/price_cache/`` by default)::

    SI_2025-11.csv ... SI_2026-11.csv   one file per month, 15-min rows
    export_prices.csv                   the latest 365 days, combined

``update_price_cache()`` keeps the cache at exactly one year of lookback:

* past months are downloaded once and then kept (day-ahead prices are final);
* the current month (and any month that was incomplete when fetched) is refreshed,
  picking up newly published days including tomorrow's prices;
* month files entirely older than the window are deleted;
* ``export_prices.csv`` is rewritten with the newest 365 days of prices.

A failed download leaves the existing cache untouched. The GUI refreshes the cache at
startup and daily; the CLI runs it with ``uv run solar-price-cache``.

Files use the generic price layout (``timestamp,price_eur_mwh``; timestamp = interval
start, local CET/CEST time), so ``export_prices.csv`` works as ``--export-price-file``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from solar_battery_simulator.data_readers import load_export_prices

API_URL = "https://api.energy-charts.info/price"
LOCAL_TZ = "Europe/Ljubljana"
DEFAULT_CACHE_DIR = Path("data") / "price_cache"
COMBINED_NAME = "export_prices.csv"
LOOKBACK_DAYS = 365
_USER_AGENT = "solar-battery-simulator"


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def _get_json(url: str, timeout: float = 60.0, retries: int = 4) -> dict:
    """GET JSON, backing off on HTTP 429/5xx (honours ``Retry-After``)."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                body: dict = json.load(resp)
            return body
        except urllib.error.HTTPError as exc:
            if attempt == retries or not (exc.code == 429 or exc.code >= 500):
                raise
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            delay = float(retry_after) if retry_after.isdigit() else 5.0 * 2**attempt
            time.sleep(min(delay, 120.0))
    raise RuntimeError("unreachable")


def fetch_prices(start: dt.date, end: dt.date, region: str = "SI") -> pd.DataFrame:
    """Day-ahead prices for local days [start, end] as 15-min ``interval_start`` /
    ``price_eur_mwh`` rows (hourly products are repeated over their four quarters;
    the repeated autumn DST hour is averaged)."""
    query = urllib.parse.urlencode({"bzn": region, "start": f"{start}", "end": f"{end}"})
    body = _get_json(f"{API_URL}?{query}")
    seconds = body.get("unix_seconds") or []
    prices = body.get("price") or []
    if not seconds:
        return pd.DataFrame(columns=["interval_start", "price_eur_mwh"])

    utc = pd.to_datetime(np.asarray(seconds, dtype="int64"), unit="s", utc=True)
    # Each product lasts until the next one starts; the last one keeps the previous step.
    steps = np.diff(np.asarray(seconds, dtype="int64")) // 60
    minutes = np.append(steps, steps[-1] if len(steps) else 15)
    minutes = np.clip(minutes, 15, 60)
    repeats = (minutes // 15).astype(int)
    offsets = np.concatenate([np.arange(n) * 15 for n in repeats])
    expanded_utc = utc.repeat(repeats) + pd.to_timedelta(offsets, unit="min")
    local = expanded_utc.tz_convert(LOCAL_TZ).tz_localize(None)
    df = pd.DataFrame(
        {
            "interval_start": local,
            "price_eur_mwh": np.repeat(np.asarray(prices, dtype=float), repeats),
        }
    ).dropna()
    df = df.groupby("interval_start", as_index=False).agg({"price_eur_mwh": "mean"})
    return df.sort_values("interval_start").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cache files
# ---------------------------------------------------------------------------
def _month_path(cache_dir: Path, region: str, month: pd.Period) -> Path:
    return cache_dir / f"{region}_{month.strftime('%Y-%m')}.csv"


def _write(df: pd.DataFrame, path: Path) -> None:
    out = pd.DataFrame(
        {
            "timestamp": df["interval_start"].dt.strftime("%Y-%m-%d %H:%M"),
            "price_eur_mwh": df["price_eur_mwh"].round(3),
        }
    )
    tmp = path.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(path)  # atomic: readers never see a half-written file


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.rename(columns={"timestamp": "interval_start"})


def _days_covered(df: pd.DataFrame) -> set[dt.date]:
    return set(df["interval_start"].dt.date) if len(df) else set()


def _month_days(month: pd.Period) -> list[dt.date]:
    return list(pd.date_range(month.start_time, month.end_time, freq="D").date)


def update_price_cache(
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    region: str = "SI",
    lookback_days: int = LOOKBACK_DAYS,
    today: dt.date | None = None,
) -> dict:
    """Bring the cache up to date and rewrite the combined one-year file.

    Returns a summary dict (``fetched``, ``kept``, ``pruned`` month lists, the combined
    ``path``, its ``start``/``end`` days and ``errors``). Download errors are collected,
    not raised, so an offline run leaves the existing cache in place.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    today = today or dt.date.today()
    last_day = today + dt.timedelta(days=1)  # tomorrow's prices are published ~13:00
    first_day = today - dt.timedelta(days=lookback_days)
    months = pd.period_range(first_day, last_day, freq="M")
    summary: dict = {"fetched": [], "kept": [], "pruned": [], "errors": []}

    stale = []
    for month in months:
        path = _month_path(cache_dir, region, month)
        wanted = [d for d in _month_days(month) if first_day <= d <= last_day]
        have = _days_covered(_read(path)) if path.exists() else set()
        # Past days are final: a finished month with every wanted day cached is kept.
        complete = all(d in have for d in wanted)
        if path.exists() and complete and month.end_time.date() < today:
            summary["kept"].append(str(month))
        else:
            stale.append(month)

    # One request for the whole stale span (the API serves a year per call), split by month.
    if stale:
        try:
            df = fetch_prices(stale[0].start_time.date(), stale[-1].end_time.date(), region)
        except Exception as exc:  # network/HTTP/parse errors: keep what we have
            summary["errors"].append(f"download {stale[0]}..{stale[-1]} failed: {exc}")
            df = None
        if df is not None:
            by_month = df["interval_start"].dt.to_period("M")
            for month in stale:
                path = _month_path(cache_dir, region, month)
                part = df[by_month == month]
                # Never replace a cached month with a shorter (partial) response.
                existing = len(_read(path)) if path.exists() else 0
                if len(part) and len(part) >= existing:
                    _write(part, path)
                    summary["fetched"].append(str(month))
                else:
                    summary["errors"].append(f"{month}: no new prices, kept cache")

    # Prune months that ended before the window.
    keep = {_month_path(cache_dir, region, m).name for m in months}
    for path in sorted(cache_dir.glob(f"{region}_????-??.csv")):
        if path.name not in keep:
            path.unlink()
            summary["pruned"].append(path.stem.split("_", 1)[1])

    files = [_month_path(cache_dir, region, m) for m in months]
    frames = [_read(p) for p in files if p.exists()]
    if not frames:
        summary["errors"].append("no cached prices available")
        return summary
    combined = pd.concat(frames, ignore_index=True).drop_duplicates("interval_start")
    combined = combined.sort_values("interval_start")
    # Newest 365 days of what is actually published (ends today or tomorrow).
    end_day = combined["interval_start"].max().date()
    start_day = end_day - dt.timedelta(days=lookback_days - 1)
    combined = combined[combined["interval_start"].dt.date >= start_day]
    combined_path = cache_dir / COMBINED_NAME
    _write(combined, combined_path)
    days = len(_days_covered(combined))
    summary.update(path=str(combined_path), start=str(start_day), end=str(end_day), days=days)
    if days < lookback_days:
        summary["errors"].append(f"combined file covers {days} of {lookback_days} days")
    return summary


def market_price_file(cache_dir: str | Path = DEFAULT_CACHE_DIR, refresh: bool = True) -> str:
    """Path of the last-year price file, updating the cache first when ``refresh``.

    An outdated cache is still used (with a warning) when the update fails.

    Raises:
        RuntimeError: If there is no cached price file at all.
    """
    if refresh:
        summary = update_price_cache(cache_dir)
        for err in summary["errors"]:
            print(f"Price cache: {err}")
        if "path" in summary:
            print(f"Price cache: {summary['start']} .. {summary['end']} ({summary['days']} days)")
    path = cached_price_file(cache_dir)
    if path is None:
        raise RuntimeError(
            "No day-ahead prices cached and the download failed; check the network or "
            "use fixed export pricing"
        )
    return path


def cached_price_file(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> str | None:
    """Path of the combined one-year price file, if the cache has been built."""
    path = Path(cache_dir) / COMBINED_NAME
    return str(path) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Update the rolling 1-year price cache")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--region", default="SI")
    parser.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    args = parser.parse_args()
    s = update_price_cache(args.cache_dir, args.region, args.lookback_days)
    print(f"Fetched: {', '.join(s['fetched']) or '-'}")
    print(f"Kept:    {', '.join(s['kept']) or '-'}")
    print(f"Pruned:  {', '.join(s['pruned']) or '-'}")
    for err in s["errors"]:
        print(f"Error:   {err}")
    if "path" in s:
        prices = load_export_prices(s["path"])
        print(
            f"{s['path']}: {s['start']} .. {s['end']} ({s['days']} days), "
            f"{len(prices)} intervals, "
            f"mean {prices['price_eur_kwh'].mean() * 1000:.2f} EUR/MWh"
        )


if __name__ == "__main__":
    main()
