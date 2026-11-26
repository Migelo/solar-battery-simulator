# AGENTS.md

Solar PV + battery simulator for Slovenian smart-meter data (15-min intervals): savings, ROI, NPV,
system sizing. NiceGUI web app + CLI.

## Layout

```
src/solar_battery_simulator/  package (installed editable by uv; absolute imports only)
  gui.py           NiceGUI app            -> uv run solar-gui
  simulator.py     engine + CLI           -> uv run solar-sim
  data_readers.py  CSV/XLSX/XLS input, network tariffs
  price_cache.py   SI day-ahead price cache -> uv run solar-price-cache
  hupx_prices.py   HUPX API fetcher       -> uv run solar-hupx-prices
  cenikOmreznin.csv  default network tariffs (BUNDLED_TARIFF_FILE)
data/      sample production/consumption files (committed); price_cache/ is gitignored
windows/   exe entry (windows_exe.py), PyInstaller build (build_exe.py), start_windows.bat
docs/      user docs
tests/     pytest; fixtures/*.xls
```

## Running tests and tools

Always use the project's local uv environment, never a global Python or `pip install`:

- `uv sync --extra test` once, then `uv run pytest` (`uv run pytest tests/test_x.py::test_y -v`)
- `uv run ruff check .` / `uv run ruff format .` (line length 100)
- Tools that aren't dependencies: `uv run --with`, e.g. mypy with CI's pins:
  `uv run --with mypy==1.18.2 --with pandas-stubs --with types-seaborn mypy --ignore-missing-imports --no-strict-optional src/solar_battery_simulator/data_readers.py`

GUI tests use `nicegui.testing.user_plugin` (`main_file` in `pyproject.toml`, run via runpy —
hence absolute imports and the `if __name__ in {"__main__", "__mp_main__"}` guard).
`tests/conftest.py` sets `PRICE_CACHE_AUTO_UPDATE=0`.

## Things that aren't obvious from the code

- **Paths**: data lookup is upload dir → `data/` → cwd (`_find_data_file`, any supported
  extension for the stem). Tariff file: same, then the bundled copy. CLI inputs default to
  `data/production.csv` / `data/consumption.csv`.
- **Timestamps**: consumption labels are interval *ends*; a midnight label belongs to the previous
  day. Export prices align on interval start = label − 15 min.
- **Batch cache** (joblib, `_cached_batch_simulation_func`): anything that changes results must be
  in the key — inputs are fingerprinted by SHA-256, plus reference power, export fee, price file and
  alignment. Add new parameters to it.
- **Input cache**: `_load_aligned_inputs` memoizes the parsed, aligned inputs in-process (keyed
  on the same fingerprint) and replays their console output; `load_and_align_data` copies the
  frame, so never mutate the cached one.
- **Investment**: `total_investment` = gross − subsidy drives payback, loan and NPV; maintenance
  (1 %) stays on gross. `loan_years <= 0` = cash at year 0. NPV horizon is `npv_years`
  (default 20, result column `npv`); `loan_years > npv_years` is rejected (engine, CLI) or
  capped (GUI), so no debt outlives the horizon.
- **Tariffs**: 5 blocks, high season Nov–Mar; OVE-SPTE = `4 × Block1_max + 8 × Block2_max`.
  `--transmission-block*`/`--power-fee-block*` require `--tariff-group manual`.
- **GUI**: one shared sidebar `inputs` dict for both tabs (`_only_on()` hides tab-specific
  sections); simulations run in `asyncio.to_thread`; all plots are `ui.echart` (no matplotlib in
  the GUI). Dark mode lives in `app.storage.user` — `app.storage.browser` is read-only after the
  HTTP response.
- **Excel/CSV exports** use Slovenian formats (`[$-424]`, `;` + decimal comma + BOM for CSV).
- **Windows**: `start_windows.bat` must keep CRLF (`.gitattributes -text`). The exe chdirs to its
  own folder; PyInstaller can't cross-compile, so CI builds and smoke-tests it.
- **Env**: `GUI_HOST` (default 127.0.0.1; Docker sets 0.0.0.0, compose publishes on 127.0.0.1),
  `GUI_PORT` (8080, else a free port), `GUI_OPEN_BROWSER=0`, `PRICE_CACHE_AUTO_UPDATE=0`.
- **Docker** copies `pyproject.toml`, `uv.lock` and `src/` only; `data/` is a volume.
- `tests/fixtures/*.xls` are synthetic (pandas can't write XLS); regenerate with
  `soffice --headless --convert-to 'xls:MS Excel 97'`.

## CI and shipping

`tests.yml`: ruff + mypy (`data_readers.py`), pytest on 3.12/3.13 + CLI smoke run, Windows tests +
launcher start, Docker build. `windows-exe.yml`: builds the exe on every PR, attaches it to a
release on `v*` tags. `main` is protected: branch → PR → squash auto-merge.
Bulk mechanical commits (moves, reformatting) go in `.git-blame-ignore-revs`
(`git config blame.ignoreRevsFile .git-blame-ignore-revs`; GitHub uses it automatically).
