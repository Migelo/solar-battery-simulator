# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python-based solar PV + battery storage simulation tool. Analyzes electricity consumption data from Slovenian smart meters to simulate and optimize battery storage systems, modeling cost savings, ROI, NPV, and system sizing at 15-minute intervals.

## Common Commands

```bash
# Run the GUI (NiceGUI web app on http://localhost:8080)
uv run gui.py

# Run the simulator CLI (single scenario)
uv run power_flow_simulator.py --solar-power 15 --battery-capacity 20

# Run the simulator CLI (batch mode)
uv run power_flow_simulator.py --batch-mode --solar-range 10,15,20 --battery-range 0,10,20 --inverter-range 8,10,15

# Run with Docker Compose (place CSV/Excel input files in data/)
docker compose up --build

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_simulation.py

# Run a specific test
uv run pytest tests/test_simulation.py::test_energy_conservation -v

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy power_flow_simulator.py

# Pre-commit hooks (install once, then runs automatically)
pre-commit install
pre-commit run --all-files

# Install dependencies including test extras
uv sync --extra test
```

## Architecture

**`gui.py`** — NiceGUI web frontend with two tabs (Single Scenario / Batch Analysis). Key design decisions:
- Runs simulations via `asyncio.to_thread` to keep the UI responsive
- Matplotlib plots use a scoped dark theme (`_GUI_PLOT_RC` via `plt.rc_context`) — does **not** mutate global rcParams, so CLI plots in `power_flow_simulator.py` are unaffected
- Plots rendered as inline base64 `<img>` tags (not `ui.image()` which collapses in dynamic tab panels)
- Batch analysis shows a live progress bar polling `len(analyzer.results)` via `ui.timer`
- Dark mode preference persisted in `app.storage.user` (server-side) — `ui.dark_mode()` and `ui.switch()` both bound via `bind_value`; `app.storage.browser` is cookie-based and becomes read-only after HTTP response, so it cannot persist WebSocket-driven changes
- Data file lookup: uploaded files → `data/` subdir → working directory
- `ui.run()` guarded with `if __name__ in {"__main__", "__mp_main__"}` for test compatibility
- Sidebar uses `ui.scroll_area` with explicit `height: calc(100vh - 170px)` for scrolling

**`power_flow_simulator.py`** (~3,500 lines) — all simulation logic.

### Two Main Classes

**`PowerFlowSimulator`** — Single scenario simulation engine
- `load_and_align_data()`: Loads production (solar baseline) and consumption (usage) files via `data_readers`, parses timestamps, adds time features (hour, weekday, month, holiday, transmission block). Holidays come from `slovenian_holidays(year)` (fixed dates + Easter-based) for every year in the data
- `apply_heating_load()`: Optional synthetic heat pump load (Oct–Mar) with monthly weighting factors
- `scale_solar_generation()`: Scales from the reference capacity of the production file (`reference_solar_kw`, default `DEFAULT_REFERENCE_SOLAR_KW` = 640.6 kWp; CLI `--reference-solar-power`, GUI "Reference power" next to the production upload; part of the batch cache key) to target capacity, applies inverter clipping
- `simulate_power_flows()`: Core 15-minute loop — manages battery charge/discharge, grid import/export, optional power smoothing (peak shaving)
- `calculate_costs_and_savings()`: Economic analysis with 5-block transmission costs, monthly power fees, OVE-SPTE weighted costs

**`MultiScenarioAnalyzer`** — Batch analysis across parameter combinations
- Generates all valid (solar × inverter × battery) scenarios (constraint: inverter ≤ solar)
- Runs each through `PowerFlowSimulator` with joblib caching (`_cached_batch_simulation_func`); the cache key includes a SHA-256 of both input files (`_input_fingerprint`), so replacing a file under the same path invalidates it
- Calculates ROI, NPV (20-year), payback periods, loan amortization
- Produces heatmap visualizations (savings, ROI, self-sufficiency, Pareto)
- `optimize_npv_differential_evolution()`: scipy-based optimization for optimal system configuration

### Simulation Data Flow

1. Load input files (CSV/Excel, via `data_readers`) → aligned DataFrame (35,040 rows/year = 4 intervals/hr × 24hr × 365 days)
2. Optional: add heating load to consumption
3. Scale solar from baseline, apply inverter clipping
4. Per-interval loop: net demand → battery charge/discharge → grid import/export → SOC tracking
5. Calculate costs: baseline (grid-only) vs solar+battery scenario → savings

### Key Battery Logic

Each 15-min interval: surplus solar charges battery (respecting C-rate and efficiency losses), deficit discharges battery first then imports from grid. Power smoothing optionally uses battery reserve to reduce peak grid demand below configurable thresholds.

### Slovenian Tariff System

5-block transmission costs vary by season (high: Nov–Mar, low: Mar–Nov) and workday/time-of-day. Block 1 is most expensive (high season weekday peak hours), Block 3 cheapest. Monthly power fees are charged per kW per block. OVE-SPTE fee uses weighted formula: `4 × Block1_max_power + 8 × Block2_max_power`.

## Input Data

Two input files required, each CSV, XLSX/XLSM (openpyxl) or XLS (xlrd). Parsing lives in **`data_readers.py`** (`read_table`, `load_production`, `load_consumption`, `load_export_prices`, `align_export_prices`); `load_and_align_data()` only calls it.
- `production`: Solar generation baseline, read positionally (interval id, solar_power_kw)
- `consumption`: either the legacy layout (Time `D. M. YYYY HH:MM:SS`, Energy kWh, Power kW, Transmission fee block — positional) or the Slovenian smart-meter export (`Časovna značka`, `Energija A+`, `P+ Prejeta delovna moč`, `Blok`) — detected from the header via `detect_consumption_format()`
- `export_prices` (optional, `export_price_file=` / `--export-price-file`): HUPX DAM download (60-min `Hour` or 15-min `Quarter hour`, EUR/MWh, DST repeats `9B` averaged, `Region` filtered) or generic `timestamp,price` (interval start; EUR/MWh unless header has `kwh`). Aligned onto the 15-min grid *before* the 1h resample, as interval start = consumption timestamp − 15 min (consumption labels are interval ends). Exact timestamp match, else month/day/time match for another year; >1% gaps raise. When present, column `export_price_eur_kwh` in `self.df` replaces the fixed `export_price` in `calculate_costs_and_savings()` (which also returns `export_revenue` (net of fee), `export_fee_cost`, `export_kwh`, `avg_export_price`). `export_fee_eur_mwh` (CLI `--export-fee`, default 0) is deducted per exported MWh in fixed and market mode and is part of the batch cache key. The price file is part of the batch cache fingerprint
- **`price_cache.py`**: rolling 1-year SI day-ahead price cache in `data/price_cache/` (gitignored): `SI_YYYY-MM.csv` per month + combined `export_prices.csv` (newest 365 days, 15-min, generic `timestamp,price_eur_mwh`). Source: Energy-Charts API (`/price?bzn=SI&start=&end=`, local dates, end inclusive, hourly before Oct 2025 expanded to 15-min, one request for all stale months, backoff on 429). `update_price_cache(today=...)` keeps finished complete months, refetches the current/incomplete ones, prunes months before the window, never overwrites with shorter data, collects errors instead of raising. GUI refreshes it via `app.on_startup` + every 6 h (`PRICE_CACHE_AUTO_UPDATE=0` disables; set in `tests/conftest.py`), . Export pricing defaults to **fixed**; market mode (GUI select / CLI `--export-pricing market`) uses the cache by default (an uploaded / `data/` `export_prices` file overrides it) with `export_price_alignment="calendar"` — always month/day/time against the last 365 days, whatever the consumption year; own files use `"auto"` (exact timestamps if covered). The alignment is part of the batch cache key
- **`hupx_prices.py`**: CLI fetcher for the public HUPX Labs API (`/data/v1/dam_aggregated_trading_data[_15min]`, split at 2025-10-01, follows `nextPage`). Anonymous access only serves ~the last 2 months; older history needs a signed-in download from labs.hupx.hu. Not used by the GUI, so not in the Dockerfile

GUI lookup (`_find_data_file`): uploaded files → `data/` → working directory, matching the stem with any supported extension (exact name first). Uploads keep their own extension and replace earlier uploads of the same input in other formats. CLI: `--production-file`, `--consumption-file`.

Both files exist in the repository root and are auto-detected by the GUI.

## Docker Deployment

`docker compose up --build` builds and runs the GUI on port 8080. Place the production and consumption files (CSV or Excel) in a `data/` directory — it is volume-mounted into the container at `/app/data`. The `Dockerfile` uses uv and copies only `pyproject.toml`, `uv.lock`, `data_readers.py`, `power_flow_simulator.py`, `price_cache.py`, and `gui.py` — add any new top-level module there.

## Code Quality

- **Formatter/linter**: Ruff (line-length 100, target py313). `gui.py` has per-file-ignores for B018/SIM117 (NiceGUI widget context managers)
- **Type checker**: mypy (with pandas-stubs, types-seaborn)
- **Pre-commit hooks**: trailing whitespace, ruff lint+format, mypy
- **CI** (`.github/workflows/tests.yml`, push/PR to main): lint job (ruff check + format, mypy on `data_readers.py`); test job on Python 3.12/3.13 (pytest with coverage → Codecov, then a CLI smoke run on the `.xls` fixtures); docker job (build, then read the `.xls` fixtures inside the image)
- Python ≥3.12 required; pyproject.toml targets py313

## Test Structure

Tests in `tests/` using pytest with fixtures in `conftest.py`:
- `test_simulation.py`: Energy conservation, battery bounds, solar surplus, grid constraints
- `test_costs.py`: Baseline costs, time-of-use pricing
- `test_heating_load.py`: Energy distribution, monthly weighting, edge cases
- `test_integration.py`: Full pipeline, determinism, edge cases (zero solar/consumption), MultiScenarioAnalyzer
- `test_gui.py`: GUI unit tests (incl. file lookup/upload) + NiceGUI `User` fixture tests (page structure, tabs, inputs)
- `test_holidays_and_cache.py`: Easter/holiday calendar, holiday off-peak pricing, batch cache invalidation on file replacement
- `test_price_cache.py`: rolling window, single request, reuse/refetch, pruning, offline behaviour, DST/hourly expansion (API mocked, `today` injected)
- `test_export_prices.py`: HUPX/generic price parsing, DST, alignment, market vs fixed revenue, fetcher (network mocked)
- `test_data_readers.py`: CSV/XLSX/XLS reading, layout detection, delimiters/encodings, validation errors, simulator parity across formats. CSV/XLSX inputs are generated in `tmp_path`; `tests/fixtures/*.xls` are committed synthetic files (fake meter IDs) because pandas can't write XLS — regenerate with LibreOffice `soffice --headless --convert-to 'xls:MS Excel 97'`

GUI tests use `nicegui.testing.user_plugin` (headless, no browser needed). Config in `pyproject.toml`: `asyncio_mode = "auto"`, `main_file = "gui.py"`.

3 expected warnings on zero-division in edge case tests (cosmetic, in print statements).

## Visual GUI Debugging with Playwright MCP

The Playwright MCP plugin can be used for interactive visual testing and debugging of the GUI. Start the server first, then use Playwright tools to navigate, click, take screenshots, and inspect the page.

```bash
# 1. Start the GUI server
uv run gui.py

# 2. Use Playwright MCP tools:
#    - browser_navigate to http://localhost:8080
#    - browser_snapshot for accessibility tree (preferred for actions)
#    - browser_take_screenshot for visual verification
#    - browser_click / browser_type to interact with elements
```

**Setup**: The Playwright MCP plugin is configured to use system Chromium (`/usr/bin/chromium-browser`) since Chrome is not installed. Config is in `~/.claude/plugins/marketplaces/claude-plugins-official/external_plugins/playwright/.mcp.json` with `--browser chromium --executable-path /usr/bin/chromium-browser`.

**Tips**:
- Use `browser_snapshot` (not screenshots) to get element refs for clicking/typing
- If Chromium fails to launch with "Opening in existing browser session", kill stale processes: `pkill -f "chromium-browser.*mcp-chromium"`
- The plugin config is in the plugin cache and may be overwritten on plugin updates — re-apply the chromium config if that happens
