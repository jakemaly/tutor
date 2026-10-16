FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /workspace

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY backend/pyproject.toml backend/uv.lock ./backend/
RUN uv sync --project backend --all-groups --frozen

COPY backend ./backend

EXPOSE 8000

CMD ["uv", "run", "--project", "backend", "uvicorn", "app.main:app", "--app-dir", "backend", "--reload", "--host", "0.0.0.0", "--port", "8000"]
