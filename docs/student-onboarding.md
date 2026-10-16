# Student onboarding

Welcome to Socratic AI Tutor. The current app is a small text-chat prototype. The assistant is called **Tutor** in the frontend; it does not yet provide a complete tutoring method or save student accounts and chat history.

## 1. Start the app

Install Docker Desktop (or Docker Engine with the Compose plugin), then from the repository root run:

```sh
cp .env.example .env
docker compose up --build
```

Visit <http://localhost:5173>. The `/api/health` endpoint at <http://localhost:8000/api/health> should return a healthy status even when no model is configured. Chat needs an OpenAI-compatible Chat Completions endpoint; add its base URL and model name to `.env`, plus an API key if required. Keep those values private.

You can run the app without Docker; see the native development instructions in the README.

## 2. Find your way around

- `frontend/src/` renders the chat interface and sends requests to `/api/chat`.
- `backend/app/main.py` validates the text conversation and streams the response.
- `backend/app/graph.py` defines the LangGraph chat workflow.
- `backend/app/models.py` creates the configured model client for runtime inference.
- `backend/app/observability/` is reserved for future Langfuse instrumentation.
- `training/` is reserved for data preparation, model training, and training evaluation.
- `evals/` is reserved for future evaluations of Tutor's behavior.

The training placeholders do not run inference. Runtime inference remains in the existing backend model and graph modules. More detail is in [the architecture notes](architecture.md).

## 3. Make a first contribution

1. Update your local `dev` branch.
2. Create `feature/<short-description>` from `dev`.
3. Make one focused improvement. Good first changes include a clearer error, an accessibility fix, a documentation correction, or a focused test for current text-chat behavior.
4. Run the relevant checks listed in the README.
5. Open a pull request into `dev`, describe the change, and note what you checked. Feature pull requests do not require peer approvals.

Changes reach `main` through a promotion pull request from `dev`. Main retains its approval rule. CI runs on both branches and provides feedback, but its checks are not required for merging.

Never commit `.env`, model keys, real student data, or private logs. Use made-up examples when writing docs or tests. See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full contribution workflow.
