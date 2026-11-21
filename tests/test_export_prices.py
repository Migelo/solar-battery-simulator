"""Tests for market (HUPX) export prices: parsing, alignment, costs, fetcher."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import hupx_prices
from data_readers import align_export_prices, load_export_prices
from power_flow_simulator import PowerFlowSimulator, _input_fingerprint


def _hupx_15min_csv(path, day="2026-10-20", prices=None, labels=None):
    labels = labels or [str(i) for i in range(1, 97)]
    prices = prices if prices is not None else [100.0 + i for i in range(len(labels))]
    pd.DataFrame(
        {
            "Delivery day": [f"{day}T00:00:00Z"] * len(labels),
            "Quarter hour": labels,
            "Price": prices,
            "Traded volume": [1.0] * len(labels),
            "Baseload price": [1.0] * len(labels),
            "Status": ["final"] * len(labels),
        }
    ).to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_hupx_15min_download(tmp_path):
    prices = load_export_prices(_hupx_15min_csv(tmp_path / "p.csv"))
    assert len(prices) == 96
    assert prices["interval_start"].iloc[0] == pd.Timestamp("2026-10-20 00:00")
    assert prices["interval_start"].iloc[-1] == pd.Timestamp("2026-10-20 23:45")
    assert prices["price_eur_kwh"].iloc[0] == pytest.approx(0.100)  # EUR/MWh -> EUR/kWh


def test_hupx_dst_repeated_hour_is_averaged(tmp_path):
    labels = [str(i) for i in range(1, 13)] + ["9B", "10B", "11B", "12B"]
    labels += [str(i) for i in range(13, 97)]
    values = [100.0] * len(labels)
    values[8] = 200.0  # QH 9 (02:00)
    values[12] = 300.0  # QH 9B (second 02:00)
    prices = load_export_prices(_hupx_15min_csv(tmp_path / "p.csv", "2026-10-25", values, labels))
    assert len(prices) == 96
    at_two = prices.set_index("interval_start").loc[pd.Timestamp("2026-10-25 02:00")]
    assert at_two["price_eur_kwh"] == pytest.approx(0.250)


def test_hupx_hourly_expanded_to_quarter_hours(tmp_path):
    path = tmp_path / "p.csv"
    pd.DataFrame(
        {
            "Delivery day": ["2024-06-01T00:00:00Z"] * 24,
            "Hour": list(range(1, 25)),
            "Price": [float(h) for h in range(24)],
        }
    ).to_csv(path, index=False)
    prices = load_export_prices(path)
    assert len(prices) == 96
    assert prices["price_eur_kwh"].iloc[4:8].tolist() == pytest.approx([0.001] * 4)


def test_hupx_region_filter(tmp_path):
    path = tmp_path / "p.xlsx"
    pd.DataFrame(
        {
            "DeliveryDay": ["2026-10-20T00:00:00Z"] * 2,
            "ProductH": ["1", "1"],
            "Price": [50.0, 70.0],
            "Region": ["HU", "SI"],
        }
    ).to_excel(path, index=False)
    assert load_export_prices(path)["price_eur_kwh"].iloc[0] == pytest.approx(0.070)
    assert load_export_prices(path, region="HU")["price_eur_kwh"].iloc[0] == pytest.approx(0.050)
    with pytest.raises(ValueError, match="region 'RS'"):
        load_export_prices(path, region="RS")


@pytest.mark.parametrize(
    ("header", "value", "expected"),
    [("price_eur_mwh", 80.0, 0.08), ("price", 80.0, 0.08), ("price_eur_kwh", 0.08, 0.08)],
)
def test_generic_price_units(tmp_path, header, value, expected):
    path = tmp_path / "p.csv"
    t = pd.date_range("2024-01-01", periods=8, freq="15min")
    pd.DataFrame({"timestamp": t, header: [value] * 8}).to_csv(path, index=False)
    prices = load_export_prices(path)
    assert prices["price_eur_kwh"].tolist() == pytest.approx([expected] * 8)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def _prices(start, periods, values=None):
    t = pd.date_range(start, periods=periods, freq="15min")
    v = values if values is not None else np.arange(periods) / 1000.0
    return pd.DataFrame({"interval_start": t, "price_eur_kwh": v})


def test_align_exact_uses_interval_start():
    ends = pd.Series(pd.date_range("2024-01-01 00:15", periods=4, freq="15min"))
    aligned = align_export_prices(ends, _prices("2024-01-01", 4))
    assert aligned.tolist() == pytest.approx([0.0, 0.001, 0.002, 0.003])


def test_align_by_calendar_for_other_year():
    ends = pd.Series(pd.date_range("2024-03-05 00:15", periods=96, freq="15min"))
    prices = _prices("2025-03-05", 96)
    aligned = align_export_prices(ends, prices)
    assert aligned.tolist() == pytest.approx(prices["price_eur_kwh"].tolist())


def test_calendar_alignment_ignores_overlapping_year():
    """With method="calendar" the price year's values are used even when the prices also
    contain the consumption's own timestamps (rolling last-year cache)."""
    ends = pd.Series(pd.date_range("2026-03-05 00:15", periods=4, freq="15min"))
    prices = pd.concat(
        [_prices("2026-03-05", 4, [0.9] * 4), _prices("2027-03-05", 4, [0.1] * 4)],
        ignore_index=True,
    )
    assert align_export_prices(ends, prices).tolist() == pytest.approx([0.9] * 4)
    assert align_export_prices(ends, prices, method="calendar").tolist() == pytest.approx([0.1] * 4)


def test_align_fills_small_gaps_and_rejects_large():
    ends = pd.Series(pd.date_range("2024-01-01 00:15", periods=200, freq="15min"))
    # One missing interval out of 200 (0.5%) is filled from its neighbour
    prices = _prices("2024-01-01", 200).drop(index=50)
    assert align_export_prices(ends, prices).notna().all()
    with pytest.raises(ValueError, match="cover only"):
        align_export_prices(ends, _prices("2024-01-01", 100))


# ---------------------------------------------------------------------------
# Simulator integration
# ---------------------------------------------------------------------------
def _price_file(tmp_path, values):
    path = tmp_path / "prices.csv"
    t = pd.date_range("2024-01-15 00:00", periods=96, freq="15min")
    pd.DataFrame({"timestamp": t, "price_eur_mwh": values}).to_csv(path, index=False)
    return path


@pytest.fixture
def sunny_production_csv(tmp_path):
    """Production at baseline scale (640.6 kW peak) so a 20 kW system exports at midday."""
    hours = np.arange(96) / 4
    solar = np.clip(np.sin((hours - 6) * np.pi / 12), 0, None) * 640.6
    path = tmp_path / "sunny_production.csv"
    pd.DataFrame({"timestamp_id": range(1, 97), "solar_power_kw": solar}).to_csv(path, index=False)
    return path


def _run(prod, cons, resolution="15min", export_price_file=None, export_price=0.0, fee=0.0):
    sim = PowerFlowSimulator(
        solar_panel_power_kw=20.0,
        inverter_power_kw=20.0,
        battery_capacity_kwh=5.0,
        production_file=str(prod),
        consumption_file=str(cons),
        time_resolution=resolution,
        export_price_file=str(export_price_file) if export_price_file else None,
        export_fee_eur_mwh=fee,
    )
    sim.load_and_align_data()
    sim.scale_solar_generation()
    sim.simulate_power_flows()
    return sim, sim.calculate_costs_and_savings(export_price=export_price)


@pytest.mark.parametrize("market", [False, True])
def test_export_fee_deducted_per_mwh(
    tmp_path, sunny_production_csv, minimal_consumption_csv, market
):
    prices = _price_file(tmp_path, np.linspace(-20.0, 200.0, 96)) if market else None
    _, base = _run(
        sunny_production_csv, minimal_consumption_csv, export_price_file=prices, export_price=0.07
    )
    _, fee = _run(
        sunny_production_csv,
        minimal_consumption_csv,
        export_price_file=prices,
        export_price=0.07,
        fee=12.5,
    )
    deduction = base["export_kwh"] / 1000.0 * 12.5
    assert base["export_fee_cost"] == 0.0
    assert fee["export_fee_cost"] == pytest.approx(deduction)
    assert fee["export_revenue"] == pytest.approx(base["export_revenue"] - deduction)
    assert fee["avg_export_price"] == pytest.approx(base["avg_export_price"] - 0.0125)
    assert fee["solar_battery_cost"] == pytest.approx(base["solar_battery_cost"] + deduction)


def test_export_fee_reaches_batch_simulation(sunny_production_csv, minimal_consumption_csv):
    from power_flow_simulator import MultiScenarioAnalyzer

    def savings(fee):
        analyzer = MultiScenarioAnalyzer(
            solar_range=[20.0],
            inverter_range=[20.0],
            battery_range=[0.0],
            production_file=str(sunny_production_csv),
            consumption_file=str(minimal_consumption_csv),
            export_fee_eur_mwh=fee,
        )
        return analyzer._cached_batch_simulation(20.0, 20.0, 0.0, 0.0, 0.0, 0.9, 0.15, 0.1, 0.05)

    no_fee, with_fee = savings(0.0), savings(50.0)
    exported = no_fee["simulation_results"]["grid_export_kwh"].sum()
    assert exported > 0
    assert with_fee["cost_analysis"]["export_revenue"] == pytest.approx(
        no_fee["cost_analysis"]["export_revenue"] - exported / 1000.0 * 50.0
    )


def test_constant_market_price_matches_fixed(
    tmp_path, sunny_production_csv, minimal_consumption_csv
):
    _, fixed = _run(sunny_production_csv, minimal_consumption_csv, export_price=0.07)
    _, market = _run(
        sunny_production_csv,
        minimal_consumption_csv,
        export_price_file=_price_file(tmp_path, [70.0] * 96),
        export_price=999.0,  # ignored when a price file is given
    )
    assert fixed["export_kwh"] > 0
    assert market["export_revenue"] == pytest.approx(fixed["export_revenue"])
    assert market["solar_battery_cost"] == pytest.approx(fixed["solar_battery_cost"])


@pytest.mark.parametrize("resolution", ["15min", "1h"])
def test_market_revenue_is_interval_weighted(
    tmp_path, sunny_production_csv, minimal_consumption_csv, resolution
):
    values = np.linspace(-20.0, 200.0, 96)  # includes negative prices
    sim, costs = _run(
        sunny_production_csv,
        minimal_consumption_csv,
        resolution=resolution,
        export_price_file=_price_file(tmp_path, values),
    )
    expected = (
        sim.simulation_results["grid_export_kwh"].to_numpy()
        * sim.df["export_price_eur_kwh"].to_numpy()
    ).sum()
    assert costs["export_revenue"] == pytest.approx(expected)
    assert costs["avg_export_price"] == pytest.approx(expected / costs["export_kwh"])
    # Midday (when solar exports) is priced above the flat mean of the ramp
    assert costs["avg_export_price"] > values.mean() / 1000.0


def test_fingerprint_includes_price_file(tmp_path):
    a = _price_file(tmp_path, [10.0] * 96)
    before = _input_fingerprint("p", "c", str(a))
    assert before != _input_fingerprint("p", "c", None)
    pd.DataFrame({"timestamp": ["2024-01-15 00:00"], "price_eur_mwh": [99.0]}).to_csv(a)
    assert _input_fingerprint("p", "c", str(a)) != before


# ---------------------------------------------------------------------------
# Fetcher (network mocked)
# ---------------------------------------------------------------------------
def test_fetch_splits_mtu_and_follows_pages(monkeypatch):
    calls = []

    def fake_get_json(url, **_kwargs):
        calls.append(url)
        if "page=2" in url:
            return {
                "data": [
                    {
                        "DeliveryDay": "2025-10-01T00:00:00Z",
                        "ProductQH": "2",
                        "Price": 20.0,
                        "Region": "SI",
                    }
                ]
            }
        if "15min" in url:
            return {
                "data": [
                    {
                        "DeliveryDay": "2025-10-01T00:00:00Z",
                        "ProductQH": "1",
                        "Price": 10.0,
                        "Region": "SI",
                    }
                ],
                "nextPage": url + "&page=2",
            }
        return {
            "data": [
                {
                    "DeliveryDay": "2025-09-30T00:00:00Z",
                    "ProductH": "24",
                    "Price": 40.0,
                    "Region": "SI",
                }
            ]
        }

    monkeypatch.setattr(hupx_prices, "_get_json", fake_get_json)
    prices = hupx_prices.fetch_hupx_prices(dt.date(2025, 9, 30), dt.date(2025, 10, 1))
    assert any("/dam_aggregated_trading_data?" in u for u in calls)
    assert any("/dam_aggregated_trading_data_15min?" in u for u in calls)
    by_time = prices.set_index("interval_start")["price_eur_kwh"]
    assert by_time[pd.Timestamp("2025-09-30 23:45")] == pytest.approx(0.040)
    assert by_time[pd.Timestamp("2025-10-01 00:15")] == pytest.approx(0.020)


def test_fetch_empty_raises(monkeypatch):
    monkeypatch.setattr(hupx_prices, "_get_json", lambda *_a, **_k: {"data": []})
    with pytest.raises(ValueError, match="no SI prices"):
        hupx_prices.fetch_hupx_prices(dt.date(2024, 1, 1), dt.date(2024, 1, 2))
