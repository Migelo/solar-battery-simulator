# Solar Battery Simulator

A Python tool for simulating and optimizing solar photovoltaic (PV) systems with battery energy storage. Analyzes electricity consumption data to model cost savings, ROI, and system sizing.

## Features

- **Power flow simulation** - 15-minute interval modeling of solar generation, battery charge/discharge, and grid interaction
- **Realistic equipment modeling** - Inverter clipping, battery C-rates, round-trip efficiency losses
- **Multi-scenario batch analysis** - Test hundreds of solar/battery/inverter combinations
- **Economic analysis** - ROI, NPV, payback periods with time-of-use pricing
- **Heating load modeling** - Simulate adding heat pump load to existing consumption
- **Slovenian tariff support** - 5-block transmission costs, OVE-SPTE fees, seasonal pricing

## Installation

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/YOUR_USERNAME/solar-battery-simulator.git
cd solar-battery-simulator
```

## Quick Start

### Single Scenario
```bash
uv run power_flow_simulator.py --solar-power 15 --battery-capacity 20
```

### Batch Analysis
```bash
uv run power_flow_simulator.py --batch-mode \
  --solar-range 10,15,20 \
  --battery-range 0,10,20 \
  --inverter-range 8,10,15
```

### With Heating Load
Model adding a heat pump (e.g., 4000 kWh/season):
```bash
uv run power_flow_simulator.py --batch-mode \
  --solar-range 10,15,20 \
  --battery-range 10,20 \
  --add-heating-load --heating-kwh 4000
```

## Input Data

Requires two CSV files:

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

## Output

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
| `--solar-power` | 8.5 | Solar panel capacity (kW) |
| `--inverter-power` | 8.0 | Inverter AC output limit (kW) |
| `--battery-capacity` | 10.0 | Battery storage (kWh) |
| `--battery-c-rate` | 0.5 | Charge/discharge rate |
| `--battery-efficiency` | 0.9 | Round-trip efficiency |
| `--peak-price` | 0.147 | Peak electricity (€/kWh) |
| `--off-peak-price` | 0.107 | Off-peak electricity (€/kWh) |

See `uv run power_flow_simulator.py --help` for all options.

## Architecture

```
power_flow_simulator.py
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
