import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    allowedHosts: ["suma.rhoe-api.gr", "sas.rhoe-api.gr"],
    proxy: {
      "/api": {
        target: "http://127.0.0.1:12000",
        changeOrigin: true,
      },
    },
  },
});
