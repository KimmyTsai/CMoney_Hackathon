"""
data_processor.py
-----------------
AI 投資樹洞 v3 - 核心數據處理模組
含息總報酬、成本殖利率、買點/現價分位、同溫層警示
"""

import pandas as pd
from pathlib import Path

# 資料路徑
DATA_DIR = Path(__file__).parent.parent / "cmoney-aws-summit-hackathon" / "Delivery_Hackathon_DataPackage_20260624"

# 系統時間基準
SYSTEM_DATE = "2025-12-31"
SYSTEM_DATE_NO_DASH = "20251231"

# 大盤基準 (0050 年報酬率)
BENCHMARK_RETURN = 36.9


class DataProcessor:
    """CMoney 資料處理核心"""

    def __init__(self):
        self._wide_table = None
        self._forum_data = None
        self._price_data = None
        self._year_high_low_cache = None
        self._div_trend = None

    @property
    def wide_table(self) -> pd.DataFrame:
        if self._wide_table is None:
            fp = DATA_DIR / "09_Wide_Table_Summary_One_Row_Per_Stock_2025.csv"
            self._wide_table = pd.read_csv(fp, dtype={"股票代號": str})
        return self._wide_table

    @property
    def forum_data(self) -> pd.DataFrame:
        if self._forum_data is None:
            fp = DATA_DIR / "10_Forum_Posts_Replies_Daily_Stats_2025.csv"
            self._forum_data = pd.read_csv(fp, dtype={"股票代號": str})
        return self._forum_data

    @property
    def price_data(self) -> pd.DataFrame:
        if self._price_data is None:
            fp = DATA_DIR / "01_Price_Valuation_2025.csv"
            self._price_data = pd.read_csv(fp, dtype={"股票代號": str, "日期": str})
        return self._price_data

    def _get_year_high_low(self, stock_id: str) -> tuple:
        """
        取得年度最高價與最低價
        個股：從 09 寬表的「今年新高」「今年新低」取
        ETF/無資料：從 01 逐日行情計算
        """
        row = self.wide_table[self.wide_table["股票代號"] == stock_id]
        if not row.empty:
            yh = row.iloc[0].get("今年新高")
            yl = row.iloc[0].get("今年新低")
            if pd.notna(yh) and pd.notna(yl):
                return float(yh), float(yl)

        # Fallback: 從逐日行情算（含拆分偵測：收盤單日跳動 >30% 視為斷點，只取最後一段，
        # 否則像 0050 這種年中 1 拆 4 的標的，年區間會橫跨拆分前後導致分位嚴重失真）
        df = self.price_data[self.price_data["股票代號"] == stock_id].sort_values("日期")
        if df.empty:
            return None, None
        closes = df["收盤價"].dropna().tolist()
        start = 0
        for i in range(1, len(closes)):
            if closes[i - 1] > 0 and abs(closes[i] / closes[i - 1] - 1) > 0.30:
                start = i
        seg = df.iloc[start:]
        return float(seg["最高價"].max()), float(seg["最低價"].min())

    @property
    def div_trend(self) -> dict:
        """現金股利連N年遞增（正=連增N年，負=連減N年），合併個股(06)與ETF(06b)"""
        if self._div_trend is None:
            frames = []
            for fname in ["06_Consecutive_Dividend_Stocks_2025.csv",
                          "06b_Consecutive_Dividend_ETF_2025.csv"]:
                fp = DATA_DIR / fname
                if fp.exists():
                    df = pd.read_csv(fp, dtype={"股票代號": str})
                    frames.append(df[["股票代號", "現金股利連N年遞增"]])
            merged = pd.concat(frames).dropna() if frames else pd.DataFrame(columns=["股票代號", "現金股利連N年遞增"])
            self._div_trend = dict(zip(merged["股票代號"], pd.to_numeric(merged["現金股利連N年遞增"], errors="coerce")))
        return self._div_trend

    def get_year_avg_price(self, stock_id: str):
        """2025 年均收盤價（成本選填時的估算值；有拆分者只取拆分後段，與分位邏輯一致）"""
        df = self.price_data[self.price_data["股票代號"] == str(stock_id).strip()].sort_values("日期")
        closes = df["收盤價"].dropna().tolist()
        if not closes:
            return None
        start = 0
        for i in range(1, len(closes)):
            if closes[i - 1] > 0 and abs(closes[i] / closes[i - 1] - 1) > 0.30:
                start = i
        seg = closes[start:]
        return round(sum(seg) / len(seg), 2)

    def get_forum_year_stats(self, stock_id: str) -> dict:
        """同學會 2025 全年統計（匿名同儕對照用）"""
        f = self.forum_data[self.forum_data["股票代號"] == str(stock_id).strip()]
        if f.empty:
            return None
        return {"全年發文": int(f["發文則數"].sum()),
                "全年回文": int(f["回文則數"].sum()) if "回文則數" in f.columns else 0}

    def get_available_stocks(self) -> pd.DataFrame:
        df = self.wide_table[["股票代號", "股票名稱", "產業"]].copy()
        df["顯示"] = df["股票代號"] + " " + df["股票名稱"]
        return df.sort_values("股票代號").reset_index(drop=True)

    def get_stock_summary(self, stock_id: str) -> dict:
        """從寬表取得靜態彙總"""
        stock_id = str(stock_id).strip()
        row = self.wide_table[self.wide_table["股票代號"] == stock_id]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "股票代號": stock_id,
            "股票名稱": r.get("股票名稱", ""),
            "產業": r.get("產業", "未知"),
            "收盤價": self._sf(r.get("收盤價")),
            "連續配息年數": self._sf(r.get("連續配息年數")),
            "股利連N年遞增": self._sf(self.div_trend.get(stock_id)),
            "買點分位_系統": self._sf(r.get("買點分位(%)")),
            "殖利率": self._sf(r.get("殖利率(%)")),
            "本益比": self._sf(r.get("本益比(近四季)")),
            "年報酬率": self._sf(r.get("年報酬率(%)")),
            "與大盤比": self._sf(r.get("與大盤比年報酬(%)")),
            "現金股利": self._sf(r.get("最新年度現金股利")),
            "股利發放率": self._sf(r.get("股利發放率(%)")),
            "最近除息日": r.get("最近除息日"),
            "外資持股率": self._sf(r.get("外資持股率(%)")),
            "總市值億": self._sf(r.get("總市值(億)")),
            "近20日法人買賣超": self._sf(r.get("近20日法人買賣超")),
        }

    def get_forum_sentiment(self, stock_id: str) -> dict:
        """取得 2025/12/31 社群情緒"""
        stock_id = str(stock_id).strip()
        row = self.forum_data[
            (self.forum_data["股票代號"] == stock_id) &
            (self.forum_data["日期"] == SYSTEM_DATE)
        ]
        if row.empty:
            return {"發文則數": 0, "看多發文": 0, "看空發文": 0, "多空比": None, "回文則數": 0}

        r = row.iloc[0]
        bullish = int(r.get("看多發文", 0))
        bearish = int(r.get("看空發文", 0))

        if bearish > 0:
            ratio = round(bullish / bearish, 2)
        elif bullish > 0:
            ratio = float("inf")
        else:
            ratio = None

        return {
            "發文則數": int(r.get("發文則數", 0)),
            "看多發文": bullish,
            "看空發文": bearish,
            "多空比": ratio,
            "回文則數": int(r.get("回文則數", 0)),
        }

    def calculate_percentile(self, price: float, stock_id: str) -> float:
        """
        計算某個價格在年高低區間的分位
        公式: (price - year_low) / (year_high - year_low) * 100
        """
        year_high, year_low = self._get_year_high_low(stock_id)
        if year_high is None or year_low is None:
            return None
        if year_high == year_low:
            return 50.0
        pctl = (price - year_low) / (year_high - year_low) * 100
        return round(max(0, min(100, pctl)), 1)

    def build_ai_context(self, stock_id: str, cost: float, shares: int, buy_date: str = "") -> dict:
        """整合所有數據，打包完整 AI 上下文"""
        summary = self.get_stock_summary(stock_id)
        if summary is None:
            return None

        sentiment = self.get_forum_sentiment(stock_id)
        closing_price = summary["收盤價"]
        cash_dividend = summary["現金股利"] or 0

        # --- 核心運算 ---
        # 帳面損益
        pnl_pct = None
        if closing_price and cost > 0:
            pnl_pct = round(((closing_price - cost) / cost) * 100, 2)

        # 含息總報酬
        total_return = None
        if closing_price and cost > 0:
            total_return = round(((closing_price + cash_dividend) / cost - 1) * 100, 2)

        # 成本殖利率
        cost_yield = None
        if cost > 0 and cash_dividend > 0:
            cost_yield = round((cash_dividend / cost) * 100, 2)

        # 年股息現金流
        annual_dividend_income = round(cash_dividend * shares, 0) if cash_dividend else 0

        # 市值
        market_value = round(closing_price * shares, 0) if closing_price else 0

        # 現價分位（收盤價在年高低的位置）
        now_pctl = self.calculate_percentile(closing_price, stock_id) if closing_price else None

        # 用戶買進成本分位
        cost_pctl = self.calculate_percentile(cost, stock_id) if cost > 0 else None

        # 年高年低
        year_high, year_low = self._get_year_high_low(stock_id)

        # 持有天數（基準日 2025/12/31 − 買進日期）
        holding_days = None
        if buy_date:
            try:
                from datetime import datetime as _dt
                holding_days = (_dt(2025, 12, 31) - _dt.strptime(str(buy_date)[:10], "%Y-%m-%d")).days
            except (ValueError, TypeError):
                pass

        context = {
            "持有天數": holding_days,
            "本益比": summary.get("本益比"),
            "近20日法人買賣超": summary.get("近20日法人買賣超"),
            "總市值億": summary.get("總市值億"),
            "股票代號": stock_id,
            "股票名稱": summary["股票名稱"],
            "產業": summary["產業"],
            "買進成本": cost,
            "持有股數": shares,
            "買進日期": buy_date,
            "收盤價": closing_price,
            "帳面損益": pnl_pct,
            "含息總報酬": total_return,
            "成本殖利率": cost_yield,
            "現金股利": cash_dividend,
            "年股息現金流": annual_dividend_income,
            "市值": market_value,
            "連續配息年數": summary["連續配息年數"],
            "股利連N年遞增": summary["股利連N年遞增"],  # 正=股利金額連增N年, 負=連減N年
            "殖利率": summary["殖利率"],
            "年報酬率": summary["年報酬率"],
            "與大盤比": summary["與大盤比"],
            "最近除息日": summary["最近除息日"],
            "年高": year_high,
            "年低": year_low,
            "現價分位": now_pctl,
            "成本分位": cost_pctl,
            "買點分位_系統": summary["買點分位_系統"],
            "發文則數": sentiment["發文則數"],
            "看多發文": sentiment["看多發文"],
            "看空發文": sentiment["看空發文"],
            "多空比": sentiment["多空比"],
            "回文則數": sentiment["回文則數"],
        }
        return context

    def compute_portfolio_alerts(self, contexts: list) -> list:
        """
        計算同溫層警示
        回傳警示清單 [{type, stock, message}]
        """
        alerts = []
        total_value = sum(c["市值"] for c in contexts if c["市值"])

        for ctx in contexts:
            stock_label = f"{ctx['股票名稱']}({ctx['股票代號']})"

            # 1. 集中度警示：單一持股 > 50%
            if total_value > 0 and ctx["市值"]:
                weight = ctx["市值"] / total_value * 100
                ctx["持股權重"] = round(weight, 1)
                if weight > 50:
                    alerts.append({
                        "type": "concentration",
                        "stock": stock_label,
                        "message": f"⚠️ {stock_label} 佔你總持股的 {weight:.0f}%，集中度偏高。雞蛋放同一籃子，波動會被放大。",
                    })
            else:
                ctx["持股權重"] = 0

            # 2. 同溫層警示：多空比 > 20 且年報酬落後大盤 > 10%
            ratio = ctx.get("多空比")
            vs_benchmark = ctx.get("與大盤比")
            if ratio and ratio != float("inf") and ratio > 20:
                if vs_benchmark is not None and vs_benchmark < -10:
                    alerts.append({
                        "type": "echo_chamber",
                        "stock": stock_label,
                        "message": (
                            f"🫧 {stock_label} 社群多空比高達 {ratio}（極度樂觀），"
                            f"但該股年報酬落後大盤 {abs(vs_benchmark):.1f}%。"
                            f"留意同溫層效應，社群熱度不等於獲利。"
                        ),
                    })

            # 3. 虧損過大警示
            pnl = ctx.get("帳面損益")
            if pnl is not None and pnl < -20:
                alerts.append({
                    "type": "deep_loss",
                    "stock": stock_label,
                    "message": (
                        f"📉 {stock_label} 帳面虧損 {pnl:.1f}%，已超過 20% 警戒線。"
                        f"若考慮攤平，請先想想資金效率與機會成本。"
                    ),
                })

        return alerts

    @staticmethod
    def _sf(value) -> float:
        """Safe float conversion"""
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (ValueError, TypeError):
            return None


# 單例
_instance = None


def get_processor() -> DataProcessor:
    global _instance
    if _instance is None:
        _instance = DataProcessor()
    return _instance


# ══════════ 持股防護罩：警示引擎 ══════════
# 四種警示，全部由真實資料觸發（回放 2025 全年 + 未來已排定事件）：
#   🏦 法人動向(02)：外資/投信連續 >=5 日買超或賣超
#   📈 價格動能(01)：創今年新高/新低（含拆分斷點與年初暖機處理）
#   💬 社群情緒(10)：單日聲量爆量(>=3倍30日均且>=50則)、情緒轉空(看空>看多且>=30則)
#   📅 除息事件(05)：除息日提醒

class ShieldEngine:
    def __init__(self, processor: "DataProcessor"):
        self.p = processor
        self._inst = None
        self._div = None

    @property
    def inst_data(self) -> pd.DataFrame:
        if self._inst is None:
            self._inst = pd.read_csv(DATA_DIR / "02_Institutional_Trading_2025.csv",
                                     dtype={"股票代號": str, "日期": str}).sort_values(["股票代號", "日期"])
        return self._inst

    @property
    def dividend_data(self) -> pd.DataFrame:
        if self._div is None:
            self._div = pd.read_csv(DATA_DIR / "05_Dividend_Ex_Dividend_2025.csv",
                                    dtype={"股票代號": str})
        return self._div

    @staticmethod
    def _fmt(d) -> str:
        import re as _re
        s = _re.sub(r"\D", "", str(d))[:8]  # 相容 20251230 與 2025-12-30 兩種格式
        return f"{s[:4]}/{s[4:6]}/{s[6:]}" if len(s) == 8 else str(d)

    def _stock_name(self, sid: str) -> str:
        row = self.p.wide_table[self.p.wide_table["股票代號"] == sid]
        return row.iloc[0]["股票名稱"] if len(row) else sid

    def _institutional_alerts(self, sid, name):
        alerts = []
        g = self.inst_data[self.inst_data["股票代號"] == sid]
        for col, who in [("外資買賣超", "外資"), ("投信買賣超", "投信")]:
            streak, direction = 0, 0
            for _, row in g.iterrows():
                v = row[col]
                if pd.isna(v) or v == 0:
                    streak, direction = 0, 0
                    continue
                d = 1 if v > 0 else -1
                streak = streak + 1 if d == direction else 1
                direction = d
                if streak == 5:  # 每一波只在第 5 天觸發一次
                    act = "買超" if d > 0 else "賣超"
                    alerts.append({"日期": self._fmt(row["日期"]), "類型": "🏦 法人動向",
                                   "訊息": f"{who}已連續 5 日{act}{name}，可留意籌碼變化"})
        return alerts

    def _price_alerts(self, sid, name):
        alerts = []
        g = self.p.price_data[self.p.price_data["股票代號"] == sid].sort_values("日期")[["日期", "收盤價"]].dropna()
        run_hi = run_lo = prev = None
        warmup = 20  # 年初前 20 個交易日不告警
        for i, (_, row) in enumerate(g.iterrows()):
            c = row["收盤價"]
            if prev is not None and prev > 0 and abs(c / prev - 1) > 0.30:
                run_hi, run_lo = None, None  # 拆分斷點：重置並重新暖機
                warmup = i + 20
            prev = c
            new_hi = run_hi is None or c > run_hi
            new_lo = run_lo is None or c < run_lo
            run_hi = c if new_hi else run_hi
            run_lo = c if new_lo else run_lo
            if i < warmup:
                continue
            if new_hi:
                alerts.append({"日期": self._fmt(row["日期"]), "類型": "📈 價格動能",
                               "訊息": f"{name}收盤 {c} 創今年新高"})
            elif new_lo:
                alerts.append({"日期": self._fmt(row["日期"]), "類型": "📉 價格動能",
                               "訊息": f"{name}收盤 {c} 創今年新低"})
        # 同方向連續創高/低，同一個月只留第一則，避免洗版
        dedup, last_type, last_month = [], None, None
        for a in alerts:
            if a["類型"] != last_type or a["日期"][:7] != last_month:
                dedup.append(a)
            last_type, last_month = a["類型"], a["日期"][:7]
        return dedup

    def _forum_alerts(self, sid, name):
        alerts = []
        g = self.p.forum_data[self.p.forum_data["股票代號"] == sid].sort_values("日期").reset_index(drop=True)
        for i, row in g.iterrows():
            posts = row["發文則數"]
            base = g["發文則數"].iloc[max(0, i - 30):i]
            avg = base.mean() if len(base) >= 10 else None
            if avg and posts >= max(50, avg * 3):
                prev_posts = g["發文則數"].iloc[i - 1] if i > 0 else 0
                if prev_posts < max(50, avg * 3):
                    alerts.append({"日期": self._fmt(row["日期"]), "類型": "💬 社群情緒",
                                   "訊息": f"{name}同學會單日 {int(posts)} 則發文，達 30 日均量 {posts/avg:.0f} 倍，討論突然升溫"})
            if row["看空發文"] > row["看多發文"] and posts >= 30:
                alerts.append({"日期": self._fmt(row["日期"]), "類型": "💬 社群情緒",
                               "訊息": f"{name}同學會當日看空({int(row['看空發文'])}) 超過看多({int(row['看多發文'])})，情緒轉空"})
        return alerts

    def _dividend_alerts(self, sid, name):
        past, future = [], []
        d = self.dividend_data[self.dividend_data["股票代號"] == sid]
        for _, row in d.iterrows():
            if pd.isna(row["除息日"]):
                continue
            v = int(row["除息日"])
            cash = row.get("現金股利")
            cash_txt = f"，現金股利 {cash} 元" if pd.notna(cash) and str(cash).strip() else ""
            item = {"日期": self._fmt(v), "類型": "📅 除息事件",
                    "訊息": f"{name}於 {self._fmt(v)} 除息{cash_txt}，留意參與除息的資格日"}
            (future if v > 20251231 else past).append(item)
        return past, future

    def generate(self, holdings: list) -> dict:
        replay, upcoming = [], []
        for h in holdings:
            sid = h["stock_id"]
            name = self._stock_name(sid)
            batch = (self._institutional_alerts(sid, name) + self._price_alerts(sid, name)
                     + self._forum_alerts(sid, name))
            past_div, future_div = self._dividend_alerts(sid, name)
            batch += past_div
            for a in batch + future_div:
                a["名稱"] = name
            replay += batch
            upcoming += future_div
        replay.sort(key=lambda a: a["日期"], reverse=True)
        upcoming.sort(key=lambda a: a["日期"])
        by_type = {}
        for a in replay:
            by_type[a["類型"]] = by_type.get(a["類型"], 0) + 1
        return {"回放": replay, "即將到來": upcoming, "統計": by_type, "總數": len(replay)}


def get_shield_engine() -> ShieldEngine:
    return ShieldEngine(get_processor())
