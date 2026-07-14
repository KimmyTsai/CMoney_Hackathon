"""
llm_service.py
--------------
AI 投資樹洞 v5 - 雙引擎 LLM 串接模組
支援：本地 Ollama (qwen2.5:7b-instruct) + AWS Bedrock (Claude 3.5 Haiku)
"""

import json
import os
import requests

# ══════════ 設定 ══════════
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b-instruct"

AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
BEDROCK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# ══════════ Prompt ══════════
PROMPT_TEMPLATE = """【角色設定】
你是「AI 投資樹洞」，一位極具同理心的投資陪伴教練。你只使用繁體中文回覆，語氣溫暖但精準，像一個懂投資的好朋友。

【用戶持股數據（真實資料，請務必引用）】
- 股票：{stock_name}（{stock_id}），產業：{industry}
- 買進成本：{cost} 元 → 2025/12/31 收盤價：{closing_price} 元
- 帳面損益：{pnl_pct}%
- 含息總報酬（加回股利後）：{total_return}%
- 成本殖利率：{cost_yield}%
- 連續配息：{dividend_years} 年，今年現金股利：{cash_dividend} 元
- 年股息現金流：{dividend_income} 元
- 現價分位：{now_pctl}%（0=年度最低，100=年度最高）
- 買進成本分位：{cost_pctl}%
- 同期 0050 年報酬率：36.9%，本股年報酬率：{stock_return}%
- 社群(2025/12/31)：發文 {post_count} 則，看多 {bullish}，看空 {bearish}，多空比 {sentiment_ratio}
- 最近除息日：{ex_date}

【觸發的警示】
{alerts_text}

【診斷要求 — 六維度分析】
請依照以下六個維度，逐一給出 1~2 句精要分析：

1. 現金流體質：先肯定其配息紀錄（連續配息年數、成本殖利率），再點出年股息對生活的實質意義。
2. 報酬對比：誠實區分「帳面損益」與「含息總報酬」的差異，對照同期 0050 表現。
3. 位階判讀：用「成本分位」與「現價分位」說明用戶是買在山頂還是谷底，目前股價的相對位置。
4. 集中度與配置：若有集中度警示則指出風險，沒有則簡單帶過。
5. 社群同溫層：解讀多空比的意涵。若觸發同溫層警示，溫柔提醒「社群熱度不等於投資績效，避免人踩人」。
6. 情緒陪伴：針對用戶目前的損益狀態，給一句走心的安撫或鼓勵。

【結尾引導（必須包含）】
用「即將到來的除息日」或「填息追蹤」作為鉤子，設計一句吸引用戶訂閱「持股防護罩」的文案。

【絕對禁止】
- 不得出現任何簡體字
- 不得給出「買進、賣出、加碼、減碼」的直接投資建議
- 只能使用「值得留意」「可以想想」「你可以觀察」等陪伴框架
- 嚴格遵守投顧法規

【格式要求】
- 繁體中文
- 總字數 250~350 字
- 用數字佐證你的觀點，不要空泛"""


class LLMService:
    """雙引擎 LLM 服務：Ollama + AWS Bedrock"""

    def __init__(self):
        self._bedrock_client = None

    # ────────── 可用性檢查 ──────────
    def is_ollama_available(self) -> bool:
        try:
            resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def is_aws_available(self) -> bool:
        """檢查環境變數中是否有 AWS 認證"""
        return bool(os.environ.get("AWS_ACCESS_KEY_ID"))

    def get_available_engines(self) -> list:
        """回傳目前可用的引擎列表"""
        engines = []
        if self.is_aws_available():
            engines.append("aws")
        if self.is_ollama_available():
            engines.append("ollama")
        return engines

    # ────────── Bedrock Client ──────────
    @property
    def bedrock_client(self):
        if self._bedrock_client is None:
            import boto3
            self._bedrock_client = boto3.client(
                service_name="bedrock-runtime",
                region_name=AWS_REGION,
            )
        return self._bedrock_client

    # ────────── Prompt 構建 ──────────
    def build_prompt(self, context: dict, alerts: list = None) -> str:
        """構建完整 Prompt，帶入實際數據"""
        ratio = context.get("多空比")
        if ratio is None:
            ratio_display = "無資料"
        elif ratio == float("inf"):
            ratio_display = "全看多（無看空發文）"
        else:
            ratio_display = f"{ratio}"

        ex_date_raw = context.get("最近除息日")
        if ex_date_raw and str(ex_date_raw) != "nan":
            try:
                ex_str = str(int(float(ex_date_raw)))
                ex_date = f"{ex_str[:4]}/{ex_str[4:6]}/{ex_str[6:]}"
            except (ValueError, TypeError):
                ex_date = "尚未公告"
        else:
            ex_date = "尚未公告"

        alerts_text = "無觸發警示"
        if alerts:
            relevant = [a for a in alerts if context["股票名稱"] in a.get("stock", "")]
            if relevant:
                alerts_text = "\n".join([f"- {a['message']}" for a in relevant])

        def _v(val, fmt=".2f"):
            if val is None:
                return "N/A"
            return f"{val:{fmt}}"

        return PROMPT_TEMPLATE.format(
            stock_name=context.get("股票名稱", "未知"),
            stock_id=context.get("股票代號", ""),
            industry=context.get("產業", "未知"),
            cost=context.get("買進成本", 0),
            closing_price=_v(context.get("收盤價")),
            pnl_pct=_v(context.get("帳面損益")),
            total_return=_v(context.get("含息總報酬")),
            cost_yield=_v(context.get("成本殖利率")),
            dividend_years=_v(context.get("連續配息年數"), fmt=".0f"),
            cash_dividend=_v(context.get("現金股利")),
            dividend_income=f"{context.get('年股息現金流', 0):,.0f}",
            now_pctl=_v(context.get("現價分位"), fmt=".1f"),
            cost_pctl=_v(context.get("成本分位"), fmt=".1f"),
            stock_return=_v(context.get("年報酬率"), fmt=".1f"),
            post_count=context.get("發文則數", 0),
            bullish=context.get("看多發文", 0),
            bearish=context.get("看空發文", 0),
            sentiment_ratio=ratio_display,
            ex_date=ex_date,
            alerts_text=alerts_text,
        )

    # ────────── Ollama 呼叫 ──────────
    def chat_ollama(self, prompt: str) -> str:
        """呼叫本地 Ollama"""
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.6,
                    "top_p": 0.85,
                    "num_predict": 1200,
                    "repeat_penalty": 1.1,
                },
            }
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120,
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            else:
                return f"⚠️ Ollama 錯誤 (HTTP {resp.status_code})"
        except requests.exceptions.ConnectionError:
            return "⚠️ 無法連線 Ollama，請執行 `ollama serve`"
        except requests.exceptions.Timeout:
            return "⚠️ Ollama 回應超時"
        except Exception as e:
            return f"⚠️ Ollama 錯誤：{str(e)}"

    # ────────── AWS Bedrock 呼叫 ──────────
    def chat_bedrock(self, prompt: str) -> str:
        """呼叫 AWS Bedrock Claude 3.5 Haiku"""
        try:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1200,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.6,
            })

            response = self.bedrock_client.invoke_model(
                modelId=BEDROCK_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )

            result = json.loads(response["body"].read())
            return result["content"][0]["text"].strip()

        except Exception as e:
            error_msg = str(e)
            if "AccessDeniedException" in error_msg:
                return "⚠️ AWS 認證錯誤：無權限存取 Bedrock，請確認 IAM 權限。"
            if "ExpiredTokenException" in error_msg:
                return "⚠️ AWS Session Token 已過期，請重新取得認證。"
            if "ModelNotReadyException" in error_msg:
                return "⚠️ Bedrock 模型尚未就緒，請稍後再試。"
            return f"⚠️ Bedrock 錯誤：{error_msg[:150]}"

    # ────────── 統一診斷入口 ──────────
    def diagnose(self, context: dict, alerts: list = None, engine: str = "auto") -> str:
        """
        完整診斷流程
        engine: "ollama" | "aws" | "auto"
        auto 模式：優先 AWS Bedrock（速度快品質高），備用 Ollama
        """
        prompt = self.build_prompt(context, alerts)

        if engine == "aws":
            return self.chat_bedrock(prompt)
        elif engine == "ollama":
            return self.chat_ollama(prompt)
        else:
            # auto: 優先 AWS
            if self.is_aws_available():
                return self.chat_bedrock(prompt)
            elif self.is_ollama_available():
                return self.chat_ollama(prompt)
            else:
                return "⚠️ 無可用的 AI 引擎。請設定 AWS 環境變數或啟動 Ollama。"

    def get_status(self) -> dict:
        """取得服務狀態"""
        aws_ok = self.is_aws_available()
        ollama_ok = self.is_ollama_available()
        return {
            "aws_available": aws_ok,
            "ollama_available": ollama_ok,
            "aws_region": AWS_REGION,
            "aws_model": BEDROCK_MODEL_ID,
            "ollama_model": OLLAMA_MODEL,
        }


# 單例
_instance = None


def get_llm_service() -> LLMService:
    global _instance
    if _instance is None:
        _instance = LLMService()
    return _instance
