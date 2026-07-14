# -*- coding: utf-8 -*-
"""
存股健診原型（AWS x CMoney Hackathon — TA: 定期定額存股族）
輸入：使用者持股（代號 / 股數 / 平均成本）
輸出：1) 診斷指標 JSON  2) 可直接餵給 Amazon Bedrock 的 prompt
時間基準：以 2025/12/31 為「今天」
"""
import json
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "cmoney-aws-summit-hackathon" / "Delivery_Hackathon_DataPackage_20260624"
TODAY = "2025/12/31"

# ---------- 載入資料 ----------
wide = pd.read_csv(DATA / "09_Wide_Table_Summary_One_Row_Per_Stock_2025.csv", dtype={"股票代號": str})
price = pd.read_csv(DATA / "01_Price_Valuation_2025.csv", dtype={"股票代號": str})
forum = pd.read_csv(DATA / "10_Forum_Posts_Replies_Daily_Stats_2025.csv", dtype={"股票代號": str})
div = pd.read_csv(DATA / "05_Dividend_Ex_Dividend_2025.csv", dtype={"股票代號": str})

MARKET_YOY = 25.7  # 與大盤比反推的大盤年報酬基準（demo 用 0050 年報酬 36.9 - 超額 11.1）


def year_range(code: str):
    """由 01 檔日收盤計算年內高低（ETF 不在 04 檔，統一用這招最穩）"""
    s = price.loc[price["股票代號"] == code, "收盤價"]
    return (float(s.max()), float(s.min())) if len(s) else (None, None)


def forum_pulse(code: str, window: int = 30):
    """近 window 日 vs 前 window 日的社群聲量與多空"""
    f = forum[forum["股票代號"] == code].sort_values("日期")
    if f.empty:
        return None
    recent, prev = f.tail(window), f.tail(window * 2).head(window)
    bull, bear = int(recent["看多發文"].sum()), int(recent["看空發文"].sum())
    return {
        "近30日發文": int(recent["發文則數"].sum()),
        "聲量變化(%)": round((recent["發文則數"].sum() / max(prev["發文則數"].sum(), 1) - 1) * 100, 1),
        "看多": bull, "看空": bear,
        "多空比": round(bull / max(bear, 1), 1),
    }


def checkup(holdings: list[dict]) -> dict:
    rows, total_mv, total_cost = [], 0.0, 0.0
    for h in holdings:
        code = h["code"]
        w = wide[wide["股票代號"] == code]
        if w.empty:
            raise ValueError(f"{code} 不在 300 檔示範籃子內")
        w = w.iloc[0]
        close = float(w["收盤價"])
        mv = close * h["shares"]
        cost_total = h["cost"] * h["shares"]
        total_mv += mv
        total_cost += cost_total

        hi, lo = year_range(code)
        cost_pct = round((h["cost"] - lo) / (hi - lo) * 100, 0) if hi and hi != lo else None
        cash_div = float(w["最新年度現金股利"]) if pd.notna(w["最新年度現金股利"]) else 0.0
        d = div[div["股票代號"] == code]

        rows.append({
            "代號": code, "名稱": w["股票名稱"], "產業": w["產業"],
            "股數": h["shares"], "平均成本": h["cost"], "現價": close,
            "市值": round(mv), 
            "帳面損益(%)": round((close / h["cost"] - 1) * 100, 1),
            "含息總報酬(%)": round((close + cash_div) / h["cost"] * 100 - 100, 1),
            "成本殖利率(%)": round(cash_div / h["cost"] * 100, 2) if cash_div else 0,
            "成本買點分位(%)": cost_pct,  # 0=買在年低, 100=買在年高
            "年內高/低": f"{hi}/{lo}",
            "該股年報酬(%)": float(w["年報酬率(%)"]) if pd.notna(w["年報酬率(%)"]) else None,
            "與大盤比(%)": float(w["與大盤比年報酬(%)"]) if pd.notna(w["與大盤比年報酬(%)"]) else None,
            "連續配息年數": int(w["連續配息年數"]) if pd.notna(w["連續配息年數"]) else None,
            "下次除息日": str(int(d["除息日"].iloc[0])) if len(d) and pd.notna(d["除息日"].iloc[0]) else None,
            "社群": forum_pulse(code),
        })

    for r in rows:
        r["權重(%)"] = round(r["市值"] / total_mv * 100, 1)

    ind = {}
    for r in rows:
        ind[r["產業"]] = ind.get(r["產業"], 0) + r["權重(%)"]

    port_ret = round((total_mv / total_cost - 1) * 100, 1)
    return {
        "基準日": TODAY,
        "組合總覽": {
            "總市值": round(total_mv), "總投入": round(total_cost),
            "帳面報酬(%)": port_ret,
            "對照_0050年報酬(%)": 36.9,
            "產業配置(%)": {k: round(v, 1) for k, v in ind.items()},
            "最大單一持股權重(%)": max(r["權重(%)"] for r in rows),
        },
        "持股明細": rows,
    }


SYSTEM_PROMPT = """你是 CMoney「存股健診」AI，服務對象是定期定額存股族。
根據使用者真實持股的診斷數據，用溫暖但誠實的口吻產出健診報告，必須：
1. 先給存股族在乎的正向肯定（成本殖利率、連續配息年數、股息現金流）
2. 誠實面對 2025 年高股息落後大盤的事實，區分「帳面損益」與「含息總報酬」
3. 指出集中度風險與成本買點分位的意涵
4. 對照社群情緒與實際數據，提醒同溫層效應（多空比極端偏多但績效落後時）
5. 結尾給 1-2 個「下次回來看什麼」的鉤子（如即將到來的除息日）
禁止給出買賣特定股票的直接投資建議（法遵要求），以「值得留意/可以思考」的框架呈現。
輸出使用繁體中文，200-300 字，口語、像懂投資的朋友。"""


if __name__ == "__main__":
    # Demo 組合：典型存股族（高股息 ETF + 金融股 + 老牌配息股）
    demo = [
        {"code": "00919", "shares": 10000, "cost": 23.5},
        {"code": "2886", "shares": 3000, "cost": 42.0},
        {"code": "1101", "shares": 2000, "cost": 38.0},
    ]
    result = checkup(demo)
    print("=" * 60)
    print("診斷指標 JSON（餵給 Bedrock 的資料）")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=1, default=str))
    print()
    print("=" * 60)
    print("Bedrock API 呼叫範例（Claude on Bedrock）")
    print("=" * 60)
    print(f"""
import boto3, json
client = boto3.client("bedrock-runtime", region_name="us-west-2")
resp = client.converse(
    modelId="anthropic.claude-sonnet-4-5",
    system=[{{"text": SYSTEM_PROMPT}}],
    messages=[{{"role": "user", "content": [{{"text": "我的持股健診數據：" + json.dumps(result, ensure_ascii=False)}}]}}],
)
print(resp["output"]["message"]["content"][0]["text"])
""")