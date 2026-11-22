FROM python:3.15-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ src/
RUN uv sync --frozen --no-dev

ENV GUI_HOST=0.0.0.0 GUI_OPEN_BROWSER=0

EXPOSE 8080

CMD ["uv", "run", "--no-sync", "solar-gui"]
