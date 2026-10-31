import { defineConfig } from "vite";

// `npm run dev` serves the app on :5173 and sends /api to a local server
// (see ../README.md).
export default defineConfig({
  server: {
    proxy: { "/api": "http://127.0.0.1:8080" },
  },
});
