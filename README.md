# Socratic AI Tutor

Socratic AI Tutor is a student learning prototype with a chat interface called **Tutor**. Today it is a simple text chat: a React frontend streams replies from a FastAPI backend, which runs a small LangGraph workflow and calls a configurable OpenAI-compatible chat endpoint. It does not yet implement tutoring strategies, student accounts, or learning analytics.

## Start with Docker

Requirements: Docker Desktop or Docker Engine with the Compose plugin.

```sh
cp .env.example .env
docker compose up --build
```

Open <http://localhost:5173>. The API health endpoint is <http://localhost:8000/api/health>. The interface and health endpoint work before model credentials are configured. Stop the services with `Ctrl+C`; run `docker compose down` to remove the containers.

To enable chat, edit `.env` and set a compatible Chat Completions endpoint and model:

```dotenv
LLM_BASE_URL=https://your-endpoint.example/v1
LLM_MODEL=your-model-name
LLM_API_KEY=your-key-if-required
```

The backend reads these settings. Keep credentials in `.env`; do not add them to frontend variables or commit them. Restart the backend after changing model settings with `docker compose restart backend`.

## Run without Docker

Requirements: Node.js 22.12+, npm, Python 3.11+, and [uv](https://docs.astral.sh/uv/).

```sh
npm install
uv sync --project backend --all-groups
npm run dev
```

This starts Vite at <http://localhost:5173> and FastAPI at <http://localhost:8000>. Configure the model in the repository-root `.env` file as shown above. The backend and frontend can also be run separately with the `dev` scripts in `package.json` and `frontend/package.json`.

## Project map

- `frontend/` — Tutor chat interface and Vite development server.
- `backend/app/main.py` — HTTP routes and streaming bridge; `/api/chat` is the current chat endpoint.
- `backend/app/graph.py` — LangGraph workflow used for chat.
- `backend/app/models.py` — configurable model adapter used by runtime inference.
- `backend/app/observability/` — future Langfuse instrumentation.
- `training/` — future data preparation, model training, and training evaluation.
- `evals/` — future tutor behavior evaluations.
- `docs/` — [architecture notes](docs/architecture.md), [student onboarding](docs/student-onboarding.md), and research notes.

Runtime inference stays in `backend/app/models.py` and `backend/app/graph.py`; the future training files are only placeholders and do not add a second inference path. See [student onboarding](docs/student-onboarding.md) for a guided first contribution.

## Checks

```sh
npm run lint
npm run typecheck
npm run build
npm test
docker compose config -q
docker compose build
```

GitHub Actions runs these checks on pushes and pull requests to `dev` and `main`. They provide feedback and are not required merge checks.

## Contributing

Work in a `feature/<short-description>` branch and open a pull request into `dev`. Pull requests into `dev` do not need peer approvals. Changes reach `main` through a pull request from `dev`, with the repository's existing approval rule. Read [CONTRIBUTING.md](CONTRIBUTING.md) for setup, review, secrets, and contribution guidance.

## Troubleshooting

- **The page loads but chat reports that a model is not configured:** set `LLM_BASE_URL` and `LLM_MODEL` in `.env`, then restart the backend.
- **Chat fails after model setup:** check the endpoint's Chat Completions compatibility, model or deployment name, API key, and backend logs (`docker compose logs backend`).
- **The frontend cannot reach the API in Docker:** confirm both services are running with `docker compose ps`; the Compose frontend proxies `/api` to the backend service.
- **A port is already in use:** stop the process using port `5173` or `8000`, or change the host-side port in `compose.yaml`.
- **Frontend dependencies changed:** rebuild the frontend image with `docker compose build frontend` before starting it again.
