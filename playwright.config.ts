import { defineConfig } from "@playwright/test";

const model = "http://127.0.0.1:8101";
const backend = "http://127.0.0.1:8011";
const frontend = "http://127.0.0.1:4173";

export default defineConfig({
  testDir: "./tests/smoke",
  use: { baseURL: frontend },
  webServer: [
    {
      command: "node tests/smoke/model-stub.mjs",
      url: model + "/health",
    },
    {
      command: "uv run --locked --project backend uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8011",
      env: {
        LLM_BASE_URL: model + "/v1",
        LLM_MODEL: "tutor-smoke",
        LLM_API_KEY: "local-fixture-key",
      },
      url: backend + "/api/health",
    },
    {
      command: "npm exec --workspace frontend -- vite preview --host 127.0.0.1 --port 4173 --strictPort",
      env: { BACKEND_URL: backend },
      url: frontend,
    },
  ],
});
