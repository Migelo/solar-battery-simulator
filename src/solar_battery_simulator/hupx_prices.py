#!/usr/bin/env python3
"""Download HUPX day-ahead (DAM) prices for use as export prices.

Uses the public HUPX Labs data API (https://labs.hupx.hu, "DAM - Aggregated trading
data"). Delivery days up to 30 Sep 2025 come from the 60-min MTU dataset, later days
from the 15-min MTU dataset. The output CSV (``timestamp,price_eur_mwh``; timestamp =
interval start, local CET/CEST time) can be passed to the simulator as
``--export-price-file``.

Note: without signing in, the API only serves roughly the last two months. For older
history, download the CSV from the HUPX Labs website while signed in and pass that file
to the simulator directly — it reads the HUPX download format as-is.

Example:
    uv run solar-hupx-prices --start 2026-10-01 --end 2026-10-31 --out data/export_prices.csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import urllib.parse
import urllib.request

import pandas as pd

from solar_battery_simulator.data_readers import hupx_rows_to_prices

API_BASE = "https://labs.hupx.hu/data/v1"
HOURLY_DATASET = "dam_aggregated_trading_data"
QUARTER_HOURLY_DATASET = "dam_aggregated_trading_data_15min"
FIRST_15MIN_DAY = dt.date(2025, 10, 1)
_USER_AGENT = "solar-battery-simulator"


def _get_json(url: str, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https host
        body: dict = json.load(resp)
    return body


def _fetch_dataset(dataset: str, start: dt.date, end: dt.date, region: str) -> list[dict]:
    """All rows of one dataset for [start, end], following ``nextPage`` links."""
    flt = (
        f"DeliveryDay__gte__{start:%Y-%m-%d},DeliveryDay__lte__{end:%Y-%m-%d},Region__in__{region}"
    )
    url: str | None = f"{API_BASE}/{dataset}?" + urllib.parse.urlencode({"filter": flt})
    rows: list[dict] = []
    while url:
        body = _get_json(url)
        rows.extend(body.get("data") or [])
        url = body.get("nextPage")
    return rows


def _month_chunks(start: dt.date, end: dt.date):
    cur = start
    while cur <= end:
        nxt = (cur.replace(day=1) + dt.timedelta(days=32)).replace(day=1)
        yield cur, min(end, nxt - dt.timedelta(days=1))
        cur = nxt


def fetch_hupx_prices(start: dt.date, end: dt.date, region: str = "SI") -> pd.DataFrame:
    """Fetch DAM prices for ``region`` over [start, end] as 15-min EUR/kWh rows.

    Returns the ``data_readers.PRICE_COLUMNS`` frame (``interval_start``,
    ``price_eur_kwh``). Days the API did not return are reported, not invented.
    """
    if end < start:
        raise ValueError("end date is before start date")
    frames = []
    for chunk_start, chunk_end in _month_chunks(start, end):
        ranges = []
        if chunk_start < FIRST_15MIN_DAY:
            ranges.append(
                (HOURLY_DATASET, chunk_start, min(chunk_end, FIRST_15MIN_DAY - dt.timedelta(1)))
            )
        if chunk_end >= FIRST_15MIN_DAY:
            ranges.append((QUARTER_HOURLY_DATASET, max(chunk_start, FIRST_15MIN_DAY), chunk_end))
        for dataset, a, b in ranges:
            rows = _fetch_dataset(dataset, a, b, region)
            print(f"  {dataset} {a}..{b}: {len(rows)} rows")
            if rows:
                frames.append(hupx_rows_to_prices(pd.DataFrame(rows), region, source="HUPX API"))
    if not frames:
        raise ValueError(
            f"HUPX API returned no {region} prices for {start}..{end}. Without signing in "
            "only about the last two months are available; download older data from "
            "labs.hupx.hu instead."
        )
    prices = pd.concat(frames, ignore_index=True)
    got = set(prices["interval_start"].dt.date)
    wanted = pd.date_range(start, end, freq="D").date
    missing = [d for d in wanted if d not in got]
    if missing:
        print(
            f"Warning: no prices for {len(missing)} of {len(wanted)} day(s) "
            f"(first {missing[0]}, last {missing[-1]})"
        )
    return prices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=dt.date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--region", default="SI", choices=["SI", "HU", "RS"])
    parser.add_argument("--out", default="data/export_prices.csv", help="Output CSV path")
    args = parser.parse_args()

    print(f"Fetching HUPX DAM prices ({args.region}) {args.start}..{args.end}")
    prices = fetch_hupx_prices(args.start, args.end, args.region)
    out = pd.DataFrame(
        {
            "timestamp": prices["interval_start"].dt.strftime("%Y-%m-%d %H:%M"),
            "price_eur_mwh": (prices["price_eur_kwh"] * 1000).round(3),
        }
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(
        f"Wrote {len(out)} intervals to {args.out} (mean {out['price_eur_mwh'].mean():.2f} EUR/MWh)"
    )


if __name__ == "__main__":
    main()
