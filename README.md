# 🌳 AI 投資樹洞 — CMoney × AWS AI Hackathon

## 快速啟動
```bash
pip install -r requirements.txt
aws configure                       # 填入主辦方 credentials

# 查可用的 Claude 模型 ID（注意 us./apac. 前綴）
aws bedrock list-foundation-models --by-provider anthropic \
  --query 'modelSummaries[].modelId' --output table

export BEDROCK_REGION=us-west-2     # 改成主辦方指定 region
export BEDROCK_MODEL_ID=<上面查到的ID>
export CMONEY_DATA_DIR=/path/to/Delivery_Hackathon_DataPackage_20260624

streamlit run app.py
```

## Demo 建議組合（故事最完整）
00919 × 10000股 @23.5｜2886 兆豐金 × 3000股 @42｜1101 台泥 × 2000股 @38
→ 帳面 -10.7% 但 00919 含息 +6%、台泥觸發攤平警示、00919 觸發同溫層警示（多空比 126×）

## 本次整合重點
- 組合層一次診斷（AI 拿得到配置/集中度，一個組合只呼叫一次 Bedrock）
- 含息總報酬 vs 帳面損益（存股族核心敘事）
- 成本買點分位（含 0050 拆分偵測）、30 日社群脈動、除息回訪鉤子
- 修正：OCR/對帳單巢狀按鈕失效、AI 結果快取（rerun 不重打）、失敗醒目報錯不假裝成功
- OCR/對帳單結果以可編輯表格讓使用者確認（信任設計）
