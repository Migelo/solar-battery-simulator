# AGENTS.md

Project guide: see `CLAUDE.md`.

## Running tests and tools

Run everything through the project's local uv environment, never a global Python or `pip install`:

- Tests: `uv run pytest` (install extras once with `uv sync --extra test`)
- Tools that aren't project dependencies: add them for that one run with `uv run --with`, e.g.
  `uv run --with mypy==1.18.2 --with pandas-stubs --with types-seaborn mypy --ignore-missing-imports --no-strict-optional data_readers.py`
  (same pins as CI)
