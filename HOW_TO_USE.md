# How to Use

## Running Scripts

Use `uv run` to execute Python scripts:

```bash
# Power flow simulator (recommended)
uv run power_flow_simulator.py

# Battery simulator (legacy)
uv run battery_simulator.py
```

## Power Flow Simulator Examples

### Single Scenario
```bash
uv run power_flow_simulator.py --solar-power 15.0 --battery-capacity 20.0
```

### Batch Analysis
```bash
uv run power_flow_simulator.py --batch-mode \
  --solar-range 10,15,20 --battery-range 0,10,20 \
  --electricity-price 0.15 --export-price 0.0
```

### With Heating Load
Model adding a heat pump with 4000 kWh/season consumption:
```bash
uv run power_flow_simulator.py --batch-mode \
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
uv run power_flow_simulator.py --help
```
