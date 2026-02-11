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

# Run with Docker Compose (place CSV files in data/)
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
- `load_and_align_data()`: Loads `production.csv` (solar baseline) and `consumption.csv` (usage), parses timestamps, adds time features (hour, weekday, month, holiday, transmission block)
- `apply_heating_load()`: Optional synthetic heat pump load (Oct–Mar) with monthly weighting factors
- `scale_solar_generation()`: Scales from 640.6 kW baseline to target capacity, applies inverter clipping
- `simulate_power_flows()`: Core 15-minute loop — manages battery charge/discharge, grid import/export, optional power smoothing (peak shaving)
- `calculate_costs_and_savings()`: Economic analysis with 5-block transmission costs, monthly power fees, OVE-SPTE weighted costs

**`MultiScenarioAnalyzer`** — Batch analysis across parameter combinations
- Generates all valid (solar × inverter × battery) scenarios (constraint: inverter ≤ solar)
- Runs each through `PowerFlowSimulator` with joblib caching (`_cached_batch_simulation_func`)
- Calculates ROI, NPV (20-year), payback periods, loan amortization
- Produces heatmap visualizations (savings, ROI, self-sufficiency, Pareto)
- `optimize_npv_differential_evolution()`: scipy-based optimization for optimal system configuration

### Simulation Data Flow

1. Load CSVs → aligned DataFrame (35,040 rows/year = 4 intervals/hr × 24hr × 365 days)
2. Optional: add heating load to consumption
3. Scale solar from baseline, apply inverter clipping
4. Per-interval loop: net demand → battery charge/discharge → grid import/export → SOC tracking
5. Calculate costs: baseline (grid-only) vs solar+battery scenario → savings

### Key Battery Logic

Each 15-min interval: surplus solar charges battery (respecting C-rate and efficiency losses), deficit discharges battery first then imports from grid. Power smoothing optionally uses battery reserve to reduce peak grid demand below configurable thresholds.

### Slovenian Tariff System

5-block transmission costs vary by season (high: Nov–Mar, low: Mar–Nov) and workday/time-of-day. Block 1 is most expensive (high season weekday peak hours), Block 3 cheapest. Monthly power fees are charged per kW per block. OVE-SPTE fee uses weighted formula: `4 × Block1_max_power + 8 × Block2_max_power`.

## Input Data

Two CSV files required (looked up in uploaded files first, then `data/` subdir, then working directory):
- `production.csv`: Solar generation baseline (timestamp_id, solar_power_kw)
- `consumption.csv`: Electricity usage (Time, Energy kWh, Power kW, Transmission fee block)

Both files exist in the repository root and are auto-detected by the GUI.

## Docker Deployment

`docker compose up --build` builds and runs the GUI on port 8080. Place `production.csv` and `consumption.csv` in a `data/` directory — it is volume-mounted into the container at `/app/data`. The `Dockerfile` uses uv and copies only `pyproject.toml`, `uv.lock`, `power_flow_simulator.py`, and `gui.py`.

## Code Quality

- **Formatter/linter**: Ruff (line-length 100, target py313). `gui.py` has per-file-ignores for B018/SIM117 (NiceGUI widget context managers)
- **Type checker**: mypy (with pandas-stubs, types-seaborn)
- **Pre-commit hooks**: trailing whitespace, ruff lint+format, mypy
- **CI**: GitHub Actions runs `uv run pytest --tb=short -v` on push/PR to main
- Python ≥3.12 required; pyproject.toml targets py313

## Test Structure

56 tests in `tests/` using pytest with fixtures in `conftest.py`:
- `test_simulation.py`: Energy conservation, battery bounds, solar surplus, grid constraints
- `test_costs.py`: Baseline costs, time-of-use pricing
- `test_heating_load.py`: Energy distribution, monthly weighting, edge cases
- `test_integration.py`: Full pipeline, determinism, edge cases (zero solar/consumption), MultiScenarioAnalyzer
- `test_gui.py`: GUI unit tests + NiceGUI `User` fixture tests (page structure, tabs, inputs)

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
