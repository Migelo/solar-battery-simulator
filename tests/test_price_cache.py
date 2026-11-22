"""Tests for the rolling one-year price cache (network mocked)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from solar_battery_simulator import price_cache
from solar_battery_simulator.data_readers import load_export_prices
from solar_battery_simulator.price_cache import COMBINED_NAME, fetch_prices, update_price_cache

TZ = "Europe/Ljubljana"


def _fake_api(published_until: dt.date, calls: list):
    """Fake Energy-Charts: 15-min prices (hourly before Oct 2025) up to ``published_until``."""

    def get_json(url, **_kwargs):
        q = dict(p.split("=") for p in url.split("?", 1)[1].split("&"))
        start = dt.date.fromisoformat(q["start"])
        end = min(dt.date.fromisoformat(q["end"]), published_until)
        calls.append((start, end))
        if end < start:
            return {"unix_seconds": [], "price": []}
        local = pd.date_range(
            pd.Timestamp(start),
            pd.Timestamp(end) + pd.Timedelta(days=1),
            freq="15min",
            tz=TZ,
            inclusive="left",
            nonexistent="shift_forward",
            ambiguous="NaT",
        ).dropna()
        # Hourly products before the 15-min MTU go-live on 1 Oct 2025
        local = local[(local >= pd.Timestamp("2025-10-01", tz=TZ)) | (local.minute == 0)]
        secs = [int(ts.timestamp()) for ts in local]
        return {"unix_seconds": secs, "price": [100.0 + (s // 900) % 7 for s in secs]}

    return get_json


@pytest.fixture
def api(monkeypatch):
    state = {"published": dt.date(2026, 11, 22), "calls": []}
    monkeypatch.setattr(
        price_cache,
        "_get_json",
        lambda url, **_k: _fake_api(state["published"], state["calls"])(url),
    )
    return state


def _combined(tmp_path):
    return load_export_prices(tmp_path / COMBINED_NAME)


def test_builds_full_year_in_one_request(tmp_path, api):
    s = update_price_cache(tmp_path, today=dt.date(2026, 11, 21))
    assert len(api["calls"]) == 1
    assert s["errors"] == []
    assert (s["start"], s["end"], s["days"]) == ("2025-11-23", "2026-11-22", 365)
    assert len(list(tmp_path.glob("SI_*.csv"))) == 13  # Nov 2025 .. Nov 2026
    prices = _combined(tmp_path)
    assert prices["interval_start"].min() == pd.Timestamp("2025-11-23 00:00")
    assert prices["interval_start"].max() == pd.Timestamp("2026-11-22 23:45")


def test_rerun_refetches_only_current_month(tmp_path, api):
    update_price_cache(tmp_path, today=dt.date(2026, 11, 21))
    api["calls"].clear()
    s = update_price_cache(tmp_path, today=dt.date(2026, 11, 21))
    assert api["calls"] == [(dt.date(2026, 11, 1), dt.date(2026, 11, 22))]
    assert s["fetched"] == ["2026-11"]
    assert len(s["kept"]) == 12


def test_window_rolls_forward_and_prunes(tmp_path, api):
    update_price_cache(tmp_path, today=dt.date(2026, 11, 21))
    api["published"] = dt.date(2027, 1, 6)
    s = update_price_cache(tmp_path, today=dt.date(2027, 1, 5))
    assert s["pruned"] == ["2025-11", "2025-12"]
    assert not (tmp_path / "SI_2025-11.csv").exists()
    assert (s["start"], s["end"], s["days"]) == ("2026-01-07", "2027-01-06", 365)
    # Nov 2026 was incomplete when first fetched (only up to the 22nd) -> refetched
    assert "2026-11" in s["fetched"] and "2026-12" in s["fetched"]


@pytest.mark.usefixtures("api")
def test_offline_keeps_existing_cache(tmp_path, monkeypatch):
    update_price_cache(tmp_path, today=dt.date(2026, 11, 21))
    before = (tmp_path / COMBINED_NAME).read_text()

    def offline(_url, **_kwargs):
        raise OSError("network down")

    monkeypatch.setattr(price_cache, "_get_json", offline)
    s = update_price_cache(tmp_path, today=dt.date(2026, 11, 21))
    assert any("network down" in e for e in s["errors"])
    assert s["days"] == 365
    assert (tmp_path / COMBINED_NAME).read_text() == before


def test_offline_without_cache_reports_error(tmp_path, monkeypatch):
    def offline(_url, **_kwargs):
        raise OSError("network down")

    monkeypatch.setattr(price_cache, "_get_json", offline)
    s = update_price_cache(tmp_path, today=dt.date(2026, 11, 21))
    assert "path" not in s
    assert price_cache.cached_price_file(tmp_path) is None


@pytest.mark.usefixtures("api")
def test_fetch_expands_hourly_and_handles_dst():
    df = fetch_prices(dt.date(2025, 9, 30), dt.date(2025, 10, 1))
    assert len(df) == 192  # hourly day expanded to 96 quarters + one 15-min day
    assert df["interval_start"].diff().dropna().eq(pd.Timedelta(minutes=15)).all()
    dst = fetch_prices(dt.date(2026, 10, 25), dt.date(2026, 10, 25))
    assert len(dst) == 96  # repeated 02:00-03:00 averaged into one local hour
    assert np.isfinite(dst["price_eur_mwh"]).all()


@pytest.mark.usefixtures("api")
def test_combined_file_feeds_simulator_alignment(tmp_path):
    """A 2025/26 rolling window aligns onto 2024 consumption by calendar position."""
    from solar_battery_simulator.data_readers import align_export_prices

    update_price_cache(tmp_path, today=dt.date(2026, 11, 21))
    ends = pd.Series(pd.date_range("2024-01-01 00:15", "2025-01-01 00:00", freq="15min"))
    aligned = align_export_prices(ends, _combined(tmp_path))
    assert len(aligned) == len(ends) and aligned.notna().all()
