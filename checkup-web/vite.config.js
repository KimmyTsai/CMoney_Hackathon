import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base:'./' 使用相對路徑，部署到 GitHub Pages 任何 repo 名稱都能動
export default defineConfig({
  plugins: [react()],
  base: "./",
});
