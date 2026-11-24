# Socratic AI Tutor
Home for U of T TMI's AI-for-education project development

## Run

With Docker:

```sh
cp .env.example .env
docker compose up --build
```

Or run locally (Node.js 22.12+, Python 3.11+, npm, and uv):

```sh
npm install
uv sync --project backend --all-groups
cp .env.example .env
npm run dev
```

Open <http://localhost:5173>. Health check: <http://localhost:8000/api/health>.

## Model

The UI and health check work without a model. For chat, set `LLM_BASE_URL`, `LLM_MODEL`, and an optional `LLM_API_KEY` in `.env`. These settings stay in the backend; restart the services after changes. For Docker logs, run `docker compose logs backend`.

## Attachments

The chat accepts PDF, JPEG, PNG, and WebP files up to 10 MB. PDFs are text-extracted (up to 100 pages and 200,000 characters); images are sent to the configured model, which must support vision. Cite an attachment with `[[cite:SOURCE_ID:PAGE_NUMBER]]`; the backend replaces only citations that refer to a file attached in the conversation and a valid page. Uploads are held in backend memory and are cleared when the backend restarts.

## Whiteboard

Open **Whiteboard** in the top bar to get a single-user [tldraw](https://tldraw.dev) canvas, saved in the browser. Select shapes, type a question, and choose **Ask tutor**: the selection is sent to the chat as a PNG attachment, with any text from the selected shapes, so the model must support vision. The tutor can only read the board, not change it. tldraw needs a license key in production builds; add `VITE_TLDRAW_LICENSE_KEY` to the root `.env` (and to the build environment when deploying). Local development works without one.

## Checks

To add tutor skills or preview the exact model system prompt, see [Authoring tutor skills](docs/skills.md).

```sh
npm run build
npm run typecheck
npm run lint
npm test
docker compose config -q
docker compose build
```

## Contribute

Use `feature/*` → `dev` → `main`. Pull requests to `dev` need no approval; `main` keeps its approval rule. See [CONTRIBUTING.md](CONTRIBUTING.md).
