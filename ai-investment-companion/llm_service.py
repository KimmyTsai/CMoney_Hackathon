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
PORTFOLIO_PROMPT_TEMPLATE = """【角色設定】
你是「AI 投資樹洞」，一位極具同理心的投資陪伴教練。你只使用繁體中文回覆，語氣溫暖但精準，像一個懂投資的好朋友。
使用者向你敞開了「完整的持股組合」，請針對整個組合寫「一份」診斷報告，而不是逐檔各寫一段。

【組合總覽（真實資料，請務必引用）】
- 持股檔數：{n_stocks} 檔｜總投入：{total_cost:,.0f} 元 → 總市值：{total_mv:,.0f} 元
- 組合帳面報酬：{pnl_pct:+.1f}%｜含息總報酬：{total_ret_pct:+.1f}%（同期 0050：+36.9%）
- 年股息現金流：{dividend_income:,.0f} 元
- 產業配置：{industry_alloc}
- 市值分佈：{cap_mix}
- 最大單一持股權重：{max_weight:.0f}%

【逐檔明細】（同一檔股票出現多列時，代表使用者「分批買進」的不同批次——請比較各批次的成本分位與買進時點，分析其進場習慣與是否有越攤越平的模式）
{stock_lines}

【觸發的警示】
{alerts_text}

【已排定的未來除息事件（基準日 2025/12/31 之後，僅此為真實資料）】
{upcoming_ex_text}

【診斷要求 — 一份組合報告，四段結構，350~450 字】
0. 投資風格判讀（報告的第一句話）：綜合成本分位、殖利率結構、持有天數與商品性質，用一句話講出這個人的投資風格（例如「你是偏好高息的長期存股者，但其中一筆是短線追高進場」）。這是「懂我」的第一印象，必須具體、不可套模板。
1. 接著講「組合整體」：現金流體質（總股息、加權成本殖利率、連配年數亮點）＋帳面 vs 含息的真相，對照 0050；並用市值分佈說明大型/中小型的風險屬性。
2. 配置檢視：產業集中度與最大單一權重的意涵；若某檔「成本分位遠高於現價分位、帳面重虧且持有天數長」，點出「越攤越平」的行為模式風險（觀察框架，不建議買賣）；點名組合中「最扎實」與「最需要留意」的各一檔（用數據說話，如股利遞增/遞減趨勢、含息報酬、成本分位）。
3. 情緒與同溫層：若有社群多空比極端偏多但績效落後的持股，溫柔提醒；若使用者有重虧持股，給一句走心但不濫情的陪伴。
3-1. 商品性質辨識：若持股中有槓桿型或反向型 ETF（名稱含「正2」「反1」等字樣），必須提醒：這類商品有每日重設與波動耗損特性、不配息，本質上不適合「長期存股」策略，屬於短期交易工具——但依然不得直接建議賣出，用「值得想想它在你存股組合中的角色」的框架。若組合中沒有此類商品，跳過此項不要提及。
4. 結尾引導（必須包含）：若上方「已排定的未來除息事件」有列出事件，引用該事件（股票與日期須完全一致）作鉤子；若列表為「無」，改用「填息追蹤」或「法人與社群異動即時通知」作鉤子，嚴禁自行推測或虛構任何除息日期。

【絕對禁止】
- 不得出現任何簡體字
- 不得逐檔分段各寫一份小報告（要寫成一份融會貫通的組合診斷）
- 不得給出任何「買進/賣出/加碼/減碼」的直接投資建議，一律用「值得想想」「可以留意」的框架
"""

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
    @staticmethod
    def _div_years_txt_impl(v):
        if v is None:
            return "連配N/A年"
        v = float(v)
        if v > 0:
            return f"連配{v:.0f}年"
        if v < 0:
            return f"連續{abs(v):.0f}年未配息"
        return "今年首次配息"

    def build_portfolio_prompt(self, ctxs: list, alerts: list = None) -> str:
        """把逐檔 context 聚合成組合層 prompt（一個組合一次呼叫）"""
        total_cost = sum((c.get("買進成本") or 0) * (c.get("持有股數") or 0) for c in ctxs)
        total_mv = sum((c.get("收盤價") or 0) * (c.get("持有股數") or 0) for c in ctxs)
        dividend_income = sum(c.get("年股息現金流") or 0 for c in ctxs)
        pnl_pct = (total_mv / total_cost - 1) * 100 if total_cost else 0
        total_ret_pct = ((total_mv + dividend_income) / total_cost - 1) * 100 if total_cost else 0

        ind = {}
        for c in ctxs:
            mv = (c.get("收盤價") or 0) * (c.get("持有股數") or 0)
            ind[c.get("產業", "未知")] = ind.get(c.get("產業", "未知"), 0) + mv
        industry_alloc = "、".join(f"{k} {v/total_mv*100:.0f}%" for k, v in ind.items()) if total_mv else "—"
        max_weight = max(((c.get("收盤價") or 0) * (c.get("持有股數") or 0)) / total_mv * 100
                         for c in ctxs) if total_mv else 0

        etf_mv = large_mv = small_mv = 0
        for c in ctxs:
            mv = (c.get("收盤價") or 0) * (c.get("持有股數") or 0)
            if "ETF" in str(c.get("產業", "")):
                etf_mv += mv
            elif (c.get("總市值億") or 0) > 1000:
                large_mv += mv
            else:
                small_mv += mv
        parts = []
        if etf_mv: parts.append(f"ETF {etf_mv/total_mv*100:.0f}%")
        if large_mv: parts.append(f"大型股(市值>1000億) {large_mv/total_mv*100:.0f}%")
        if small_mv: parts.append(f"中小型股 {small_mv/total_mv*100:.0f}%")
        cap_mix = "、".join(parts) if total_mv and parts else "—"

        def _v(val, fmt=".1f"):
            return "N/A" if val is None else f"{val:{fmt}}"

        lines = []
        for c in ctxs:
            mv = (c.get("收盤價") or 0) * (c.get("持有股數") or 0)
            w = mv / total_mv * 100 if total_mv else 0
            lev_tag = ""
            sname = str(c.get("股票名稱", ""))
            if any(k in sname for k in ("正2", "反1", "正2X", "2X")):
                lev_tag = "｜⚠️槓桿/反向型ETF(不配息,非存股標的)"
            trend = c.get("股利連N年遞增")
            trend_txt = (f"股利連{trend:.0f}年遞增" if trend and trend >= 2
                         else f"股利連{abs(trend):.0f}年遞減" if trend and trend <= -2 else "")
            hd = c.get("持有天數")
            hd_txt = f"持有{hd}天" if hd is not None else "持有天數N/A"
            inst20 = c.get("近20日法人買賣超")
            inst_txt = f"近20日法人買賣超{inst20:+,.0f}" if inst20 is not None else "法人動向N/A"
            lines.append(
                f"- {c.get('股票名稱')}({c.get('股票代號')}) 權重{w:.0f}%｜"
                f"帳面{_v(c.get('帳面損益'))}%｜含息{_v(c.get('含息總報酬'))}%｜"
                f"成本殖利率{_v(c.get('成本殖利率'))}%｜成本分位{_v(c.get('成本分位'), '.0f')}｜現價分位{_v(c.get('現價分位'), '.0f')}｜"
                f"{hd_txt}｜本益比{_v(c.get('本益比'))}｜{inst_txt}｜"
                f"{_div_years_txt(c.get('連續配息年數'))}{('｜' + trend_txt) if trend_txt else ''}｜"
                f"該股年報酬{_v(c.get('年報酬率'))}%（大盤+25.7%）｜"
                f"社群看多{c.get('看多發文', 0)}/看空{c.get('看空發文', 0)}{lev_tag}")

        alerts_text = "無觸發警示"
        if alerts:
            msgs = [f"- {a.get('message', a)}" for a in alerts]
            if msgs:
                alerts_text = "\n".join(msgs)

        upcoming = []
        for c in ctxs:
            raw = c.get("最近除息日")
            try:
                v = int(float(raw))
                if v > 20251231:
                    s = str(v)
                    upcoming.append(f"- {c.get('股票名稱')}({c.get('股票代號')})：{s[:4]}/{s[4:6]}/{s[6:]} 除息")
            except (TypeError, ValueError):
                pass
        upcoming_ex_text = "\n".join(upcoming) if upcoming else "無"

        return PORTFOLIO_PROMPT_TEMPLATE.format(
            upcoming_ex_text=upcoming_ex_text, cap_mix=cap_mix,
            n_stocks=len(ctxs), total_cost=total_cost, total_mv=total_mv,
            pnl_pct=pnl_pct, total_ret_pct=total_ret_pct, dividend_income=dividend_income,
            industry_alloc=industry_alloc, max_weight=max_weight,
            stock_lines="\n".join(lines), alerts_text=alerts_text)

    VISION_EXTRACT_PROMPT = (
        "這是一張台股看盤/下單 App 的截圖（可能是庫存頁、對帳單或交易紀錄）。"
        "請抽取其中的股票資訊，只輸出 JSON 陣列、不要任何其他文字或 markdown 圍欄，格式：\n"
        '[{"name":"股票名稱","stock_id":"代號或null","shares":股數整數或null,"cost":買進成本或均價浮點數或null,"date":"買進/成交日期YYYY-MM-DD或null"}]\n'
        "規則：1) 名稱一定要填；代號看不到就填 null，不要憑記憶猜代號。"
        "2) shares 是持有股數（「134股」→134；「1,000」→1000；台股一張=1000股，若欄位單位是張請乘以1000）。"
        "3) cost 是買進均價/成本價；若截圖只有現價沒有成本，cost 填 null，不要拿現價、市值或損益充當成本。"
        "4) 同一檔股票出現多筆交易紀錄時，請「逐筆分開輸出」，每一列交易就是一個 JSON 物件——"
        "不要合併、不要加總、不要自行計算平均（彙總由系統處理，你只負責忠實抄錄每一列）。"
        "5) date 填該列的成交/買進日期並轉為 YYYY-MM-DD 格式；截圖上沒有日期就填 null。"
        "6) 忽略非股票列（表頭、合計列）。看不出任何股票就輸出 []。"
    )

    def extract_holdings_from_image(self, image_bytes: bytes, fmt: str) -> list:
        """用 Bedrock 視覺能力抽取截圖中的持股（fmt: 'png' 或 'jpeg'）"""
        import re as _re
        resp = self.bedrock_client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[{"role": "user", "content": [
                {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
                {"text": self.VISION_EXTRACT_PROMPT},
            ]}],
            inferenceConfig={"maxTokens": 1500, "temperature": 0},
        )
        text = resp["output"]["message"]["content"][0]["text"].strip()
        text = _re.sub(r"^```(?:json)?|```$", "", text, flags=_re.MULTILINE).strip()
        m = _re.search(r"\[.*\]", text, _re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(0))
        out = []
        for item in data:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            date = None
            if item.get("date"):
                d = str(item["date"]).replace("/", "-").strip()[:10]
                if _re.match(r"^\d{4}-\d{2}-\d{2}$", d):
                    date = d
            out.append({
                "name": str(item.get("name", "")).strip(),
                "stock_id": str(item["stock_id"]).strip() if item.get("stock_id") else None,
                "shares": int(item["shares"]) if item.get("shares") else None,
                "cost": float(item["cost"]) if item.get("cost") else None,
                "buy_date": date,
            })
        return out

    def diagnose_portfolio(self, ctxs: list, alerts: list = None, engine: str = "auto") -> str:
        """組合層診斷：整個組合一次呼叫"""
        prompt = self.build_portfolio_prompt(ctxs, alerts)
        if engine == "aws":
            return self.chat_bedrock(prompt)
        if engine == "ollama":
            return self.chat_ollama(prompt)
        if self.is_aws_available():
            return self.chat_bedrock(prompt)
        return self.chat_ollama(prompt)

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


def _div_years_txt(v):
    return LLMService._div_years_txt_impl(v)


def get_llm_service() -> LLMService:
    global _instance
    if _instance is None:
        _instance = LLMService()
    return _instance
