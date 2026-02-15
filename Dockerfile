FROM python:3.13-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev

COPY power_flow_simulator.py gui.py ./

EXPOSE 8080

CMD ["uv", "run", "gui.py"]
