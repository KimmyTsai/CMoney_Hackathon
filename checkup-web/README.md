# 存股健診 — CMoney × AWS AI Hackathon

輸入真實持股 → 立即健診（含息總報酬 / 買點分位 / 同學會情緒）→ Bedrock AI 個人化報告。
資料基準日 2025/12/31，涵蓋 300 檔示範籃子。

## 本機開發
```bash
npm install
npm run dev            # 前端 http://localhost:5173
uvicorn server:app --port 8000   # AI 後端（需先 aws configure 並填 server.py 的 MODEL_ID）
```

## 部署 GitHub Pages
1. push 到 GitHub，repo Settings → Pages → Source 選 **GitHub Actions**
2. 每次 push main 會自動建置部署（.github/workflows/deploy.yml）
3. AI 後端上雲後，把 Lambda Function URL 填到 repo Settings → Secrets and variables → Actions → Variables → `VITE_API_URL`
