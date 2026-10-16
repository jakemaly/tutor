# Architecture and future work locations

## Current request flow

The React frontend displays the assistant as **Tutor** and sends text conversation history to `/api/chat`. During development, Vite proxies `/api` to FastAPI. FastAPI validates the request, runs the LangGraph workflow in `backend/app/graph.py`, and streams text from the configured model created in `backend/app/models.py`. Model connection settings are read by the backend from environment variables.

The frontend and API contract are intentionally small. Future changes should preserve `/api/chat` request and response behavior unless a task explicitly updates the interface and its tests.

## Planned areas

- `backend/app/observability/` — future tracing and instrumentation, including a possible Langfuse integration. Keep keys and provider configuration on the backend.
- `training/` — future data preparation, training, and evaluation workflows.
- `evals/` — future checks of tutoring behavior and response quality.

These locations contain docstring-only placeholders. They do not implement integrations, prepare data, train a model, or add a parallel inference path. Runtime inference stays in `backend/app/models.py` and `backend/app/graph.py`.
