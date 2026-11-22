"""Tests for the zoomable full-year power flow chart in the Single Scenario tab."""

import pandas as pd

from gui import power_flow_chart_options, power_flow_days, power_flow_range, zoom_dates

# Interval-end labels of one year: 2024-01-01 00:15 .. 2025-01-01 00:00
FIRST = pd.Timestamp("2024-01-01 00:15")
LAST = pd.Timestamp("2025-01-01 00:00")


def _results(n=4 * 24 * 10):
    t = pd.date_range("2024-01-01 00:15", periods=n, freq="15min")
    return pd.DataFrame(
        {
            "datetime": t,
            "solar_generation_kwh": 1.0,
            "consumption_kwh": 2.0,
            "battery_soc_percent": 50.0,
            "grid_import_kwh": 1.0,
            "grid_export_kwh": 0.5,
        }
    )


def test_days_midnight_label_belongs_to_previous_day():
    assert power_flow_days(FIRST, LAST) == (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))


class TestRange:
    def test_end_date_inclusive(self):
        assert power_flow_range("2024-07-15", "2024-07-21", FIRST, LAST) == (
            "2024-07-15 00:00",
            "2024-07-22 00:00",
        )

    def test_single_day(self):
        assert power_flow_range("2024-07-15", "2024-07-15", FIRST, LAST) == (
            "2024-07-15 00:00",
            "2024-07-16 00:00",
        )

    def test_swapped_dates_reordered(self):
        assert power_flow_range("2024-07-21", "2024-07-15", FIRST, LAST)[0] == "2024-07-15 00:00"

    def test_clamped_to_data(self):
        assert power_flow_range("2023-05-01", "2026-01-01", FIRST, LAST) == (
            "2024-01-01 00:15",
            "2025-01-01 00:00",
        )

    def test_start_after_data_still_shows_last_day(self):
        assert power_flow_range("2025-03-01", "2025-03-05", FIRST, LAST) == (
            "2024-12-31 00:00",
            "2025-01-01 00:00",
        )


class TestChartOptions:
    def test_contains_every_interval(self):
        r = _results()
        opts = power_flow_chart_options(r, kw_factor=4)
        src = opts["dataset"]["source"]
        assert len(src) == len(r)
        assert src[0] == ["2024-01-01 00:15", 4.0, 8.0, 50.0, 4.0, -2.0]

    def test_default_zoom_is_first_week_linked_across_panels(self):
        opts = power_flow_chart_options(_results(), kw_factor=4)
        for dz in opts["dataZoom"]:
            assert dz["xAxisIndex"] == [0, 1, 2]
            assert dz["startValue"] == "2024-01-01 00:15"
            assert dz["endValue"] == "2024-01-08 00:00"


class TestZoomDates:
    def test_slider_percent(self):
        assert zoom_dates({"start": 50, "end": 100}, FIRST, LAST) == ("2024-07-02", "2024-12-31")

    def test_inside_zoom_batch(self):
        assert zoom_dates({"batch": [{"start": 0, "end": 2}]}, FIRST, LAST) == (
            "2024-01-01",
            "2024-01-08",
        )

    def test_end_never_before_start(self):
        start, end = zoom_dates({"start": 50, "end": 50}, FIRST, LAST)
        assert start == end

    def test_own_dispatch_ignored(self):
        event = {"startValue": "2024-07-15 00:00", "endValue": "2024-07-22 00:00"}
        assert zoom_dates(event, FIRST, LAST) is None

    def test_missing_percent_ignored(self):
        assert zoom_dates({}, FIRST, LAST) is None
        assert zoom_dates({"start": 10}, FIRST, LAST) is None
