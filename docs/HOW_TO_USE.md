# How to Use

## Running Scripts

Use `uv run` with the installed commands:

```bash
uv run solar-gui   # web GUI
uv run solar-sim   # simulator CLI
```

## Power Flow Simulator Examples

### Single Scenario
```bash
uv run solar-sim --solar-power 15.0 --battery-capacity 20.0
```

### Batch Analysis
```bash
uv run solar-sim --batch-mode \
  --solar-range 10,15,20 --battery-range 0,10,20 \
  --electricity-price 0.15 --export-price 0.0
```

### Market Export Prices
Pay exports the SI day-ahead price of each interval (last 365 days, cached in
`data/price_cache/` and refreshed on each run) instead of the fixed price:
```bash
uv run solar-sim --solar-power 15 --battery-capacity 20 --export-pricing market
```
Or use your own price file (e.g. a HUPX download):
```bash
uv run solar-sim --solar-power 15 --battery-capacity 20 \
  --export-price-file data/Labs_DAM_Aggregated_Trading_Data_15min.xlsx
```

### With Heating Load
Model adding a heat pump with 4000 kWh/season consumption:
```bash
uv run solar-sim --batch-mode \
  --solar-range 10,15,20 --battery-range 10,15,20 \
  --add-heating-load --heating-kwh 4000
```

### Heating Load Options
- `--add-heating-load`: Enable synthetic heating load during Oct-Mar
- `--heating-kwh`: Total heating energy per season in kWh (default: 3333)
- `--heating-start-hour`: Daily heating start hour (default: 7)
- `--heating-end-hour`: Daily heating end hour, 0=midnight (default: 0)

## Help
```bash
uv run solar-sim --help
```
