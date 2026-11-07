// Vite provides the dev server/bundler, and the React plugin transforms JSX.
import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

export default defineConfig(({ mode }) => {
  // These values configure only Vite's server; model credentials stay in the backend.
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) },
    },
    // Keep Compose's generated dependency cache out of the host-mounted source tree.
    cacheDir: process.env.DEV_CACHE_DIR || "node_modules/.vite",
    server: {
      host: process.env.HOST || env.HOST || "127.0.0.1",
      proxy: {
        // Compose reaches FastAPI by service name; native development uses loopback.
        "/api": process.env.BACKEND_URL || env.BACKEND_URL || "http://127.0.0.1:8000",
      },
    },
  };
});
