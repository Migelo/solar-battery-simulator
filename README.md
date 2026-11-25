# Solar Battery Simulator

A Python tool for simulating and optimizing solar photovoltaic (PV) systems with battery energy storage. Analyzes electricity consumption data to model cost savings, ROI, and system sizing.

## Features

- **Power flow simulation** - 15-minute interval modeling of solar generation, battery charge/discharge, and grid interaction
- **Realistic equipment modeling** - Inverter clipping, battery C-rates, round-trip efficiency losses
- **Multi-scenario batch analysis** - Test hundreds of solar/battery/inverter combinations
- **Economic analysis** - ROI, NPV, payback periods with time-of-use pricing
- **Heating load modeling** - Simulate adding heat pump load to existing consumption
- **Slovenian tariff support** - 5-block network tariffs per user group from `cenikOmreznin.csv` (bundled in the package; a copy in `data/` overrides it) (or entered manually), OVE-SPTE fees, seasonal pricing

## Windows: download and double-click

1. Download **[SolarBatterySimulator.exe](https://github.com/Migelo/solar-battery-simulator/releases/latest/download/SolarBatterySimulator.exe)**
   and put it in a folder of its own (e.g. `Documents\Solar Simulator`).
2. Double-click it. Windows may say *"Windows protected your PC"* because the app isn't
   code-signed: click **More info → Run anyway** (only needed the first time).
3. Your browser opens the simulator. The first start takes up to a minute; later starts are faster.
   Keep the black window open while you work; close it to stop the simulator.

Nothing needs to be installed. Put your production and consumption files in a `data` folder
next to the exe, or upload them in the browser. The simulator only accepts connections from
the same PC, so no firewall prompt appears and nobody else on your network can open it.

Running from source instead? Double-click `windows\start_windows.bat` (installs [uv](https://docs.astral.sh/uv/)
on first use).

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/YOUR_USERNAME/solar-battery-simulator.git
cd solar-battery-simulator
```

## Quick Start

### Web GUI
```bash
uv run solar-gui   # http://localhost:8080
```
Or with Docker: `docker compose up --build` (input files in `data/`).

### Single Scenario
```bash
uv run solar-sim --solar-power 15 --battery-capacity 20
```

### Batch Analysis
```bash
uv run solar-sim --batch-mode \
  --solar-range 10,15,20 \
  --battery-range 0,10,20 \
  --inverter-range 8,10,15
```

### With Heating Load
Model adding a heat pump (e.g., 4000 kWh/season):
```bash
uv run solar-sim --batch-mode \
  --solar-range 10,15,20 \
  --battery-range 10,20 \
  --add-heating-load --heating-kwh 4000
```

## Input Data

Requires two input files. Each may be **CSV, XLSX/XLSM or legacy XLS** (Excel files are read from the first sheet). Pass them to the CLI with `--production-file` / `--consumption-file`, or upload them in the GUI. Both default to `data/production.csv` / `data/consumption.csv` (sample files for the site are committed there).

**production.csv** - Solar generation data
```csv
timestamp_id,solar_power_kw
1,0.0
2,1.5
3,4.2
```

**consumption.csv** - Electricity consumption
```csv
Time,Energy (kWh),Power (kW),Transmission fee block,
1. 1. 2024 00:15:00,0.45,1.8,4,
1. 1. 2024 00:30:00,0.52,2.1,4,
```

The consumption file can also be the **smart-meter 15-minute export** from the distribution portal, as downloaded (CSV or XLSX). It is detected by its header; `Časovna značka`, `Energija A+` (kWh), `P+ Prejeta delovna moč` (kW, optional) and `Blok` are used and other columns are ignored:
```csv
Merilno mesto,GSRN MM,Časovna značka,Leto,Mesec,Energija A+,Energija A-,...,P+ Prejeta delovna moč,...,Blok,...
7-000000,000000000000000000,2025-01-01T00:15,2025,1,0.1130,0.0000,...,0.4520,...,4,...
```

CSV files may be comma-, semicolon- or tab-separated (semicolon files use decimal commas) and UTF-8 or Windows-1250 encoded. Missing values, unparseable timestamps or blocks outside 1–5 stop the run with an error naming the column and row.

### Export prices (optional)

Exported energy is paid either a **fixed price** (default, `--export-price` €/kWh) or **market prices** (`--export-pricing market`, or *Export pricing → Market price* in the GUI): the Slovenian day-ahead price of each interval, taken from the rolling cache of the **last 365 days** and matched to the consumption by month/day/time of day — whatever year the consumption data is from.

**Rolling 1-year price cache (default source).** `solar-price-cache` (`price_cache.py`) keeps the last 365 days of SI day-ahead prices in `data/price_cache/` (one CSV per month plus a combined `export_prices.csv`), downloaded from the free [Energy-Charts API](https://api.energy-charts.info) (Fraunhofer ISE, CC BY 4.0, data: Bundesnetzagentur | SMARD.de) — the same SIPX price HUPX publishes. Finished months are downloaded once and kept, the current month is refreshed (including tomorrow's prices once published), months older than the window are deleted, and a failed download keeps the existing cache.
- GUI: refreshed at startup and every 6 hours; used when *Export pricing → Market price (SI day-ahead, last 365 days)* is selected. An uploaded price file takes precedence.
- CLI: `--export-pricing market` updates the cache and uses it; `uv run solar-price-cache` only updates it.
- Docker: the cache lives in the mounted `data/` directory, so it survives restarts.

**Export fee.** `--export-fee` (GUI: *Export fee*, €/MWh, default 0) is deducted from every exported MWh in both fixed and market pricing — e.g. a supplier's margin on spot prices.

**Own price file.** `--export-price-file` (overrides `--export-price` and the cache), or upload it in the GUI. Own files are matched by exact timestamp when they cover the consumption dates, otherwise by month/day/time.

Accepted price files (CSV/XLSX/XLS):
- the **HUPX Labs download** of *Day-Ahead Market → Aggregated trading data* as-is — 60-min MTU (`Delivery day, Hour, Price, …`, until 30 Sep 2025) or 15-min MTU (`Delivery day, Quarter hour, Price, …`, from 1 Oct 2025), price in €/MWh. The repeated DST hour (`9B`–`12B`) is averaged; a `Region` column, if present, is filtered to SI.
- a generic `timestamp,price` file: timestamp = interval start (local time), 15- or 60-min steps, price in €/MWh unless the header contains `kwh`.

Each interval's export is paid that interval's price (negative prices reduce revenue). If the prices are from another year than the consumption data they are matched by month/day/time of day; up to 1% of intervals (leap day, DST hour) are gap-filled, otherwise the run stops — the file must cover a full year.

`solar-hupx-prices` downloads prices from the public HUPX API (without signing in only about the last two months are served):
```bash
uv run solar-hupx-prices --start 2026-10-01 --end 2026-10-31 --out data/export_prices.csv
```

## Output

All CLI result files go into `--out-dir` (default `outputs/run-YYYYMMDD-HHMMSS`,
local time). The directory is created with parents; if it already holds files the
run stops unless `--overwrite` is given. Every written path is printed at the end.

```bash
uv run solar-sim --solar-power 400 --battery-capacity 250 --out-dir /tmp/smoke-out
```

### Single Mode
- `power_flow_simulation_results.csv` - Detailed 15-minute interval data
- `system_summary.csv` - System configuration and annual totals

### Batch Mode
- `multi_scenario_comparison.csv` - Ranked scenario results
- `multi_scenario_comprehensive_results.csv` - Full dataset
- Heatmap visualizations (savings, ROI, self-sufficiency, NPV)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--solar-power` | 400 | Solar panel capacity (kW) |
| `--inverter-power` | 300 | Inverter AC output limit (kW) |
| `--battery-capacity` | 250 | Battery storage (kWh) |
| `--battery-c-rate` | 0.5 | Charge/discharge rate |
| `--battery-efficiency` | 0.9 | Round-trip efficiency |
| `--peak-price` | 0.147 | Peak electricity (€/kWh) |
| `--off-peak-price` | 0.107 | Off-peak electricity (€/kWh) |
| `--export-price` | 0.01 | Fixed export price (€/kWh) |
| `--export-price-file` | – | Market export prices (HUPX download or `timestamp,price`) |
| `--export-fee` | 0 | Fee deducted per exported MWh (€/MWh), fixed or market pricing |
| `--export-pricing` | fixed | `fixed` or `market` (last-365-days SI day-ahead cache) |

See `uv run solar-sim --help` for all options.

## Architecture

```
src/solar_battery_simulator/
├── gui.py            # NiceGUI web app (solar-gui)
├── simulator.py      # simulation engine + CLI (solar-sim)
├── data_readers.py   # CSV / Excel input parsing, network tariffs
├── price_cache.py    # rolling SI day-ahead price cache (solar-price-cache)
├── hupx_prices.py    # HUPX price fetcher (solar-hupx-prices)
└── cenikOmreznin.csv # default network tariffs
data/                 # input files (samples committed; price_cache/ gitignored)
windows/              # exe build (PyInstaller) and start_windows.bat launcher
docs/HOW_TO_USE.md    # CLI examples
tests/

simulator.py
├── PowerFlowSimulator      # Single scenario simulation
│   ├── load_and_align_data()
│   ├── apply_heating_load()
│   ├── scale_solar_generation()
│   ├── simulate_power_flows()
│   └── calculate_costs_and_savings()
│
└── MultiScenarioAnalyzer   # Batch analysis
    ├── generate_scenarios()
    ├── run_all_scenarios()
    ├── create_*_heatmap()
    └── optimize_npv_differential_evolution()
```

## License

MIT

## Contributing

Pull requests welcome. Please add tests for new features.
