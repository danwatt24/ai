import { defineConfig } from "vite";

export default defineConfig({
  build: {
    target: "node24", // or your Node version
    outDir: "dist",
    ssr: true, // important for backend
    sourcemap: true,
    rollupOptions: {
      input: "src/index.ts",
    },
  },
});
