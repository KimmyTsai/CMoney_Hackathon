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

        # Fallback: 從逐日行情算
        df = self.price_data[self.price_data["股票代號"] == stock_id]
        if df.empty:
            return None, None
        return float(df["最高價"].max()), float(df["最低價"].min())

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

        context = {
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
