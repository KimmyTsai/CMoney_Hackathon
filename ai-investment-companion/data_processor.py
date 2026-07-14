"""
data_processor.py
-----------------
AI 投資樹洞 - 核心數據處理模組
單檔查詢 + 組合層診斷（含息總報酬 / 成本買點分位 / 30日社群脈動 / 風險警示）
"""

import os
import pandas as pd
from pathlib import Path

# 資料路徑：優先讀環境變數 CMONEY_DATA_DIR，否則嘗試常見相對位置
def _resolve_data_dir() -> Path:
    env = os.environ.get("CMONEY_DATA_DIR")
    if env:
        return Path(env)
    here = Path(__file__).parent
    candidates = [
        here.parent / "cmoney-aws-summit-hackathon" / "Delivery_Hackathon_DataPackage_20260624",
        here / "Delivery_Hackathon_DataPackage_20260624",
        here.parent / "Delivery_Hackathon_DataPackage_20260624",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]

DATA_DIR = _resolve_data_dir()

SYSTEM_DATE = "2025-12-31"
SYSTEM_DATE_NO_DASH = "20251231"
MKT_0050_RETURN = 36.9  # 0050 2025 年報酬（對照基準，取自 09 寬表）


class DataProcessor:
    """CMoney 資料處理核心類別"""

    def __init__(self):
        self._wide_table = None
        self._forum_data = None
        self._price_data = None
        self._dividend_data = None
        self._year_range_cache = {}

    # ---------- 資料表 lazy load ----------
    @property
    def wide_table(self) -> pd.DataFrame:
        if self._wide_table is None:
            self._wide_table = pd.read_csv(
                DATA_DIR / "09_Wide_Table_Summary_One_Row_Per_Stock_2025.csv",
                dtype={"股票代號": str})
        return self._wide_table

    @property
    def forum_data(self) -> pd.DataFrame:
        if self._forum_data is None:
            self._forum_data = pd.read_csv(
                DATA_DIR / "10_Forum_Posts_Replies_Daily_Stats_2025.csv",
                dtype={"股票代號": str}).sort_values("日期")
        return self._forum_data

    @property
    def price_data(self) -> pd.DataFrame:
        if self._price_data is None:
            self._price_data = pd.read_csv(
                DATA_DIR / "01_Price_Valuation_2025.csv",
                dtype={"股票代號": str, "日期": str}).sort_values(["股票代號", "日期"])
        return self._price_data

    @property
    def dividend_data(self) -> pd.DataFrame:
        if self._dividend_data is None:
            self._dividend_data = pd.read_csv(
                DATA_DIR / "05_Dividend_Ex_Dividend_2025.csv",
                dtype={"股票代號": str})
        return self._dividend_data

    # ---------- 基礎查詢 ----------
    def get_available_stocks(self) -> pd.DataFrame:
        df = self.wide_table[["股票代號", "股票名稱", "產業"]].copy()
        df["顯示"] = df["股票代號"] + " " + df["股票名稱"]
        return df.sort_values("股票代號").reset_index(drop=True)

    def get_stock_row(self, stock_id: str):
        row = self.wide_table[self.wide_table["股票代號"] == str(stock_id).strip()]
        return None if row.empty else row.iloc[0]

    def year_range(self, stock_id: str):
        """年內高低（由 01 檔日收盤計算，含拆分偵測：單日跳動>30%視為斷點取最後一段）"""
        stock_id = str(stock_id).strip()
        if stock_id in self._year_range_cache:
            return self._year_range_cache[stock_id]
        s = self.price_data.loc[self.price_data["股票代號"] == stock_id, "收盤價"].dropna().tolist()
        if not s:
            result = (None, None, False)
        else:
            start = 0
            for i in range(1, len(s)):
                if s[i - 1] > 0 and abs(s[i] / s[i - 1] - 1) > 0.30:
                    start = i
            seg = s[start:]
            result = (round(max(seg), 2), round(min(seg), 2), start > 0)
        self._year_range_cache[stock_id] = result
        return result

    def forum_pulse(self, stock_id: str, window: int = 30) -> dict:
        """近 window 日社群脈動 vs 前 window 日（比單日穩定）"""
        f = self.forum_data[self.forum_data["股票代號"] == str(stock_id).strip()]
        if f.empty:
            return None
        recent, prev = f.tail(window), f.tail(window * 2).head(window)
        bull, bear = int(recent["看多發文"].sum()), int(recent["看空發文"].sum())
        ratio = round(bull / bear, 1) if bear > 0 else ("全看多" if bull > 0 else None)
        return {
            "近30日發文": int(recent["發文則數"].sum()),
            "聲量變化(%)": round((recent["發文則數"].sum() / max(prev["發文則數"].sum(), 1) - 1) * 100, 1),
            "看多": bull, "看空": bear, "多空比": ratio,
        }

    def next_ex_date(self, stock_id: str):
        """下次除息日：只取基準日(2025/12/31)之後的"""
        d = self.dividend_data[self.dividend_data["股票代號"] == str(stock_id).strip()]
        if d.empty or pd.isna(d["除息日"].iloc[0]):
            return None
        v = int(d["除息日"].iloc[0])
        if v <= 20251231:
            return None
        s = str(v)
        return f"{s[:4]}/{s[4:6]}/{s[6:]}"

    # ---------- 組合層診斷（核心引擎）----------
    def build_portfolio_context(self, holdings: list) -> dict:
        """
        holdings: [{"stock_id", "cost", "shares", "buy_date"}]
        回傳組合層 + 逐檔診斷，直接餵給 AI
        """
        rows = []
        for h in holdings:
            w = self.get_stock_row(h["stock_id"])
            if w is None:
                continue
            close = self._safe_float(w.get("收盤價"))
            if not close or h["cost"] <= 0:
                continue
            cash_div = self._safe_float(w.get("最新年度現金股利")) or 0.0
            hi, lo, split = self.year_range(h["stock_id"])
            cost_pctl, above_high = None, False
            if hi is not None and hi != lo:
                raw = (h["cost"] - lo) / (hi - lo) * 100
                above_high = raw > 100
                cost_pctl = round(max(0, min(100, raw)))
            rows.append({
                "代號": h["stock_id"], "名稱": w.get("股票名稱", ""), "產業": w.get("產業", "未知"),
                "股數": h["shares"], "成本": h["cost"], "收盤價": close,
                "買進日期": h.get("buy_date", ""),
                "市值": round(close * h["shares"]),
                "帳面損益(%)": round((close / h["cost"] - 1) * 100, 1),
                "含息總報酬(%)": round((close + cash_div) / h["cost"] * 100 - 100, 1),
                "成本殖利率(%)": round(cash_div / h["cost"] * 100, 2) if cash_div else 0,
                "年度現金股利": cash_div,
                "成本買點分位(%)": "高於今年高點(更早期買進)" if above_high else cost_pctl,
                "年內高低": f"{hi}/{lo}" if hi else None,
                "含拆分調整": split,
                "該股年報酬(%)": self._safe_float(w.get("年報酬率(%)")),
                "與大盤比(%)": self._safe_float(w.get("與大盤比年報酬(%)")),
                "連續配息年數": int(w["連續配息年數"]) if pd.notna(w.get("連續配息年數")) else None,
                "下次除息日": self.next_ex_date(h["stock_id"]),
                "社群30日": self.forum_pulse(h["stock_id"]),
            })

        if not rows:
            return None

        total_mv = sum(r["市值"] for r in rows)
        total_cost = sum(r["成本"] * r["股數"] for r in rows)
        annual_div = sum(r["年度現金股利"] * r["股數"] for r in rows)
        for r in rows:
            r["權重(%)"] = round(r["市值"] / total_mv * 100, 1)

        industry = {}
        for r in rows:
            industry[r["產業"]] = round(industry.get(r["產業"], 0) + r["權重(%)"], 1)
        max_weight = max(r["權重(%)"] for r in rows)

        warnings = []
        if len(rows) > 1 and max_weight > 50:
            warnings.append(f"單一持股佔比 {max_weight:.0f}%，集中度偏高")
        for r in rows:
            p = r["社群30日"]
            ratio = p and p["多空比"]
            extreme = ratio == "全看多" or (isinstance(ratio, (int, float)) and ratio > 20)
            if extreme and r["與大盤比(%)"] is not None and r["與大盤比(%)"] < -10:
                warnings.append(
                    f"{r['名稱']}：同學會近30日多空比極端偏多（{ratio}），"
                    f"但全年落後大盤 {abs(r['與大盤比(%)']):.0f}%——留意同溫層效應")
            if r["帳面損益(%)"] < -20:
                warnings.append(f"{r['名稱']}：帳面 {r['帳面損益(%)']:.0f}%，持續加碼前值得想想資金效率")

        return {
            "基準日": SYSTEM_DATE,
            "組合總覽": {
                "總市值": round(total_mv),
                "總投入": round(total_cost),
                "帳面報酬(%)": round((total_mv / total_cost - 1) * 100, 1),
                "含息總報酬(%)": round((total_mv + annual_div) / total_cost * 100 - 100, 1),
                "年股息現金流": round(annual_div),
                "對照_0050年報酬(%)": MKT_0050_RETURN,
                "產業配置(%)": industry,
                "最大單一持股權重(%)": round(max_weight),
            },
            "系統警示": warnings,
            "持股明細": rows,
        }

    @staticmethod
    def _safe_float(value):
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (ValueError, TypeError):
            return None


_processor_instance = None


def get_processor() -> DataProcessor:
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = DataProcessor()
    return _processor_instance
