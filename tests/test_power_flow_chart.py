"""Tests for the zoomable full-year power flow chart in the Single Scenario tab."""

import pandas as pd

from gui import POWER_FLOW_WINDOWS, power_flow_chart_options, power_flow_window, zoom_start_date

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


class TestWindow:
    def test_seven_days_from_date(self):
        assert power_flow_window("2024-07-15", "7 days", FIRST, LAST) == (
            "2024-07-15 00:00",
            "2024-07-22 00:00",
        )

    def test_whole_year_is_none(self):
        assert power_flow_window("2024-07-15", "Whole year", FIRST, LAST) is None

    def test_clamped_to_data_end(self):
        start, end = power_flow_window("2024-12-30", "7 days", FIRST, LAST)
        assert end == "2025-01-01 00:00"
        assert start == "2024-12-25 00:00"

    def test_clamped_to_data_start(self):
        start, _ = power_flow_window("2023-05-01", "1 day", FIRST, LAST)
        assert start == "2024-01-01 00:15"

    def test_all_windows_handled(self):
        for window in POWER_FLOW_WINDOWS:
            power_flow_window("2024-03-01", window, FIRST, LAST)


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
            assert dz["endValue"] == "2024-01-08 00:15"

    def test_whole_year_zoom(self):
        opts = power_flow_chart_options(_results(), kw_factor=4, window="Whole year")
        assert opts["dataZoom"][0]["start"] == 0 and opts["dataZoom"][0]["end"] == 100


class TestZoomStartDate:
    def test_slider_percent(self):
        assert zoom_start_date({"start": 50}, FIRST, LAST) == "2024-07-02"

    def test_inside_zoom_batch(self):
        assert zoom_start_date({"batch": [{"start": 0, "end": 2}]}, FIRST, LAST) == "2024-01-01"

    def test_own_dispatch_ignored(self):
        event = {"startValue": "2024-07-15 00:00", "endValue": "2024-07-22 00:00"}
        assert zoom_start_date(event, FIRST, LAST) is None

    def test_missing_start_ignored(self):
        assert zoom_start_date({}, FIRST, LAST) is None
