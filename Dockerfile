FROM python:3.15-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

COPY data_readers.py power_flow_simulator.py price_cache.py gui.py cenikOmreznin.csv ./

ENV GUI_HOST=0.0.0.0 GUI_OPEN_BROWSER=0

EXPOSE 8080

CMD ["uv", "run", "gui.py"]
