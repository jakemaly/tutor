# Contributing to Socratic AI Tutor

Thanks for helping improve the student learning prototype. Start with a small, focused change and describe what a student or contributor gains from it.

## Get the project running

The quickest setup uses Docker Compose:

```sh
cp .env.example .env
docker compose up --build
```

Open <http://localhost:5173>. The UI and `/api/health` work without a model. To enable chat, add `LLM_BASE_URL`, `LLM_MODEL`, and an optional `LLM_API_KEY` to `.env`. Never commit `.env` or put model credentials in frontend code.

You can also use the native setup in the README if you have Node.js 22.12+, npm, Python 3.11+, and uv.

## Branch and pull request flow

1. Start from the latest `dev` branch.
2. Create a branch named `feature/<short-description>`.
3. Make a focused change and run the relevant checks.
4. Open a pull request into `dev` and explain the change, how you checked it, and anything a reviewer should know.
5. Changes move from `dev` to `main` through a pull request. `dev` pull requests do not require peer approval; `main` pull requests retain the repository's approval requirement.

Direct pushes to `dev` and `main` are blocked. GitHub Actions runs lint, type checking, the frontend build, backend tests, and Docker Compose checks on pushes and pull requests to these branches. These checks are informational and are not configured as merge requirements.

## Checks

From the repository root:

```sh
npm run lint
npm run typecheck
npm run build
npm test
docker compose config -q
docker compose build
```

Run checks that apply to your change. If something cannot run in your environment, include the command and its output in the pull request.
