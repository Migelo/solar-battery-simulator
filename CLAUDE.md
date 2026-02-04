# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based electricity consumption and solar battery storage analysis tool. The project analyzes electricity usage data (from Slovenian smart meters) to simulate and optimize battery storage systems with varying solar panel capacities.

## Environment Setup

- Python 3.13 with virtual environment located at `.venv/`
- Activate virtual environment: `source .venv/bin/activate`
- Required packages: pandas, numpy, matplotlib, seaborn (already installed in venv)

## Core Architecture

### Main Components

**BatterySimulator Class** (`battery_simulator.py:16`)
- Primary simulation engine that models battery storage behavior
- Handles electricity pricing (peak vs off-peak rates)
- Simulates 15-minute interval battery charge/discharge cycles
- Battery parameters: 100% round-trip efficiency (configurable), 0.5C charge/discharge rates
- Supports optional off-peak charging strategy

**Data Processing** (`battery_simulator.py:42`)
- Processes CSV data from Slovenian smart meters with columns: timestamp, grid import (A+), solar export (A-)
- Handles electricity pricing tiers: €0.23/kWh (peak: 6am-10pm weekdays), €0.18/kWh (off-peak)
- Supports solar scaling factors to model different solar panel capacities

**Analysis Functions**
- `run_capacity_analysis()`: Tests battery capacities from 1-20 kWh by default (`battery_simulator.py:201`)
- `run_solar_scaling_analysis()`: Models 1x to 5x solar capacity scaling (`battery_simulator.py:217`)
- `simulate_battery()`: Core simulation logic with SOC tracking and cost calculations (`battery_simulator.py:89`)
- `create_savings_heatmap()`: Generates heatmap visualization of savings across battery/solar combinations (`battery_simulator.py:393`)
- `create_roi_heatmap()`: Creates ROI payback period analysis with customizable equipment costs (`battery_simulator.py:444`)

### Key Algorithms

**Battery Operation Logic** (`battery_simulator.py:108`)
1. Off-peak charging: When enabled, proactively charges to 80% SOC during off-peak hours (`battery_simulator.py:116`)
2. Solar surplus charging: Charges battery with available solar surplus (accounting for efficiency losses)
3. Energy demand: Discharges battery first, then imports from grid as needed
4. SOC bounds enforcement and charge/discharge rate limiting (0.5C rates)
5. Cost calculation using time-of-use pricing

**Analysis Outputs**
- Annual cost savings calculations vs baseline (no battery)
- Battery utilization metrics (average/min/max SOC, annual cycles)
- ROI analysis with customizable equipment costs
- Time-series visualization of daily maximum SOC

## Running the Analysis

**Execute scripts using uv:**
```bash
uv run power_flow_simulator.py
uv run battery_simulator.py
```

**Optional command line arguments:**
```bash
# Enable off-peak charging (battery charges during nights/weekends)
python3 battery_simulator.py --off-peak-charging

# Custom electricity prices (EUR/kWh)
python3 battery_simulator.py --peak-price 0.25 --off-peak-price 0.15

# Combined options with custom prices
python3 battery_simulator.py --off-peak-charging --peak-price 0.30 --off-peak-price 0.12

# Uses default transmission costs (€0.01809-0.01998/kWh range)
python3 battery_simulator.py --peak-price 0.119900 --off-peak-price 0.097900

# Custom transmission costs override defaults
python3 battery_simulator.py --peak-price 0.119900 --off-peak-price 0.097900 \
  --transmission-block1 0.025 --transmission-block3 0.015
```

**Available command line options:**
- `--off-peak-charging`: Enable proactive battery charging during off-peak hours
- `--peak-price PRICE`: Peak hour electricity price in EUR/kWh (default: 0.23)
- `--off-peak-price PRICE`: Off-peak hour electricity price in EUR/kWh (default: 0.18)
- `--transmission-block1 COST`: Block 1 transmission cost in EUR/kWh (default: 0.01998)
- `--transmission-block2 COST`: Block 2 transmission cost in EUR/kWh (default: 0.01833)
- `--transmission-block3 COST`: Block 3 transmission cost in EUR/kWh (default: 0.01809)
- `--transmission-block4 COST`: Block 4 transmission cost in EUR/kWh (default: 0.01855)
- `--transmission-block5 COST`: Block 5 transmission cost in EUR/kWh (default: 0.01873)

**Key input requirements:**
- `data.csv`: Smart meter data with columns including 'Časovna značka', 'Energija A+', 'Energija A-'
- Data format: 15-minute intervals with timestamp, grid import/export values in kWh

**Generated outputs:**
- Multiple PNG visualization files showing SOC comparisons, heatmaps
- `battery_capacity_analysis.csv`: Summary results for all tested capacities
- `battery_operation_*kwh.csv`: Detailed operation logs for optimal battery size

## Data Structure

The CSV data represents Slovenian smart meter readings:
- `Časovna značka`: Timestamp in 15-minute intervals
- `Energija A+`: Energy imported from grid (kWh)
- `Energija A-`: Energy exported to grid (kWh, from solar panels)
- Analysis spans full year of data for accurate seasonal modeling

## Visualization Components

- Daily SOC tracking plots for multiple battery capacities
- Solar scaling analysis with different panel capacities  
- ROI heatmaps showing payback periods
- Cost savings heatmaps across battery/solar combinations

## Transmission Cost System

The simulator supports a sophisticated 5-block transmission cost system with seasonal and workday/non-workday variations:

**Time Blocks with Default Pricing:**
- **Block 1** (€0.01998/kWh): Most expensive, only high season workdays, 7h-14h and 16h-20h
- **Block 2** (€0.01833/kWh): High season all days, low season workdays, 7h-14h and 16h-20h  
- **Block 3** (€0.01809/kWh): Cheapest rate, various time periods
- **Block 4** (€0.01855/kWh): Various time periods
- **Block 5** (€0.01873/kWh): Only low season non-workdays, midnight-6h and 22h-midnight

**Seasons:**
- **High season**: November 1st to March 1st (winter heating season)
- **Low season**: March 1st to November 1st (rest of year)

**Time Periods by Block:**
- **7h-14h, 16h-20h**: Peak demand periods (blocks 1-3 depending on season/workday)
- **6h-7h, 14h-16h, 20h-22h**: Medium demand periods (blocks 2-4)
- **Midnight-6h, 22h-midnight**: Low demand periods (blocks 3-5)

## Specific Generated Files

**Visualization Files (PNG):**
- `daily_max_soc_comparison_*kwh.png`: Daily max SOC over time for different battery capacities
- `solar_scaling_soc_*kwh_*x.png`: SOC comparison across different solar scale factors
- `solar_battery_savings_heatmap.png`: Annual savings across battery/solar combinations
- `solar_battery_roi_heatmap.png`: ROI payback periods with investment cost assumptions

**Data Files (CSV):**
- `battery_capacity_analysis.csv`: Summary results for all tested battery capacities
- `battery_operation_*kwh.csv`: Detailed 15-minute interval logs for specific battery sizes
- add the blocks