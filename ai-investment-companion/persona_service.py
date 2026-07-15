"""
persona_service.py — 投資人格卡 (Investment Persona Card)

把使用者的持股組合，收斂成一張「可炫耀、可自嘲」的 9:16 社群分享卡。

設計理念（對照企劃）：
  觸發 → 著陸 → 匯入 → 揭曉 → 分享 的病毒迴圈中，本模組負責「揭曉」與「分享」：
    - determine_persona(): 由四項關鍵數據（ETF 佔比、產業集中度、未實現損益、波動度代理）
      判定人格主標籤，並依複合比例機制混出「70% A + 30% B」。
    - build_card_svg(): 產生抽卡風格的 9:16 卡片（Instagram 限時動態 / Threads / Dcard）。

資料誠實度：
  本次示範資料沒有真正的 Beta 值，故以「年振幅 (年高−年低)/收盤價」作為波動度代理，
  在程式碼與卡片文案上都不宣稱是精算 Beta。PR 同儕分位為示範用的穩定映射（非真實母體）。
"""
from __future__ import annotations

import hashlib

# ─────────────────────────────────────────────────────────────────────
#  產業歸類：判定「AI／科技股佔比」與 ETF 佔比
# ─────────────────────────────────────────────────────────────────────
_TECH_INDUSTRIES = {
    "電子–光電", "電子–其他電子", "電子–半導體", "電子–資訊服務",
    "電子–通信網路", "電子–電子通路", "電子–電子零組件",
    "電子–電腦及週邊設備", "數位雲端", "農業科技",
}
_ETF_INDUSTRIES = {"ETF/其他", "存託憑證"}


def _is_etf(ctx: dict) -> bool:
    sid = str(ctx.get("股票代號", ""))
    return ctx.get("產業") in _ETF_INDUSTRIES or sid.startswith("00")


def _is_tech(ctx: dict) -> bool:
    return ctx.get("產業") in _TECH_INDUSTRIES


def _mv(ctx: dict) -> float:
    return (ctx.get("收盤價") or 0) * (ctx.get("持有股數") or 0)


def _volatility_proxy(ctx: dict) -> float:
    """年振幅 (年高−年低)/收盤價，作為單檔波動度（Beta）的代理。回傳 0~1+。"""
    hi, lo, px = ctx.get("年高"), ctx.get("年低"), ctx.get("收盤價")
    if not px or not hi or not lo or px <= 0:
        return 0.0
    return max(0.0, (hi - lo) / px)


# ─────────────────────────────────────────────────────────────────────
#  真實市場參照分布：用全市場個股當母體，算「贏過多少檔個股」
#  （取代先前的固定映射；PR 改為真實橫斷面百分位）
# ─────────────────────────────────────────────────────────────────────
def build_market_reference(processor=None) -> dict:
    """
    掃全市場 ~300 檔，建立兩條分布：
      vol：每檔 (年高−年低)/收盤價（與使用者持股用的同一種算法，含 ETF 逐日行情回退）
      ret：每檔 2025 年報酬率(%)
    回傳已排序（遞增）的 list，供 _percentile() 查百分位。建議由呼叫端快取。
    """
    if processor is None:
        from data_processor import get_processor
        processor = get_processor()
    import pandas as pd

    w = processor.wide_table
    vols, rets = [], []
    for sid, close in zip(w["股票代號"], pd.to_numeric(w["收盤價"], errors="coerce")):
        if not close or close <= 0:
            continue
        try:
            hi, lo = processor._get_year_high_low(sid)
        except Exception:
            hi, lo = None, None
        if hi and lo and hi >= lo:
            vols.append((hi - lo) / close)
    rets = pd.to_numeric(w.get("年報酬率(%)"), errors="coerce").dropna().tolist()
    return {"vol": sorted(vols), "ret": sorted(rets),
            "n_vol": len(vols), "n_ret": len(rets)}


def _percentile(value: float, arr: list, higher_is_better: bool) -> int:
    """
    value 在 arr（遞增排序的母體）中「贏過」多少 %。
      higher_is_better=True ：值越大越好 → 贏過比你小的那些
      higher_is_better=False：值越小越好 → 贏過比你大的那些
    夾在 1~99，避免出現 0% 或 100% 這種不可信的極值。
    """
    n = len(arr)
    if n == 0:
        return 60
    if higher_is_better:
        beat = sum(1 for x in arr if x < value)
    else:
        beat = sum(1 for x in arr if x > value)
    pct = round(beat / n * 100)
    return max(1, min(99, pct))


# ─────────────────────────────────────────────────────────────────────
#  組合特徵萃取
# ─────────────────────────────────────────────────────────────────────
def compute_features(ctxs: list) -> dict:
    """把逐檔 context 聚合成組合層特徵（皆以市值加權）。"""
    total_mv = sum(_mv(c) for c in ctxs) or 1.0

    etf_mv = sum(_mv(c) for c in ctxs if _is_etf(c))
    tech_mv = sum(_mv(c) for c in ctxs if _is_tech(c))

    # 產業集中度：最大單一產業佔比
    ind_mv: dict = {}
    for c in ctxs:
        ind_mv[c.get("產業", "未知")] = ind_mv.get(c.get("產業", "未知"), 0) + _mv(c)
    top_industry, top_industry_mv = max(ind_mv.items(), key=lambda x: x[1]) if ind_mv else ("未知", 0)

    # 單一持股集中度
    top_stock_mv = max((_mv(c) for c in ctxs), default=0)

    # 市值加權：未實現損益、波動度
    w_pnl = sum((c.get("帳面損益") or 0) * _mv(c) for c in ctxs) / total_mv
    w_vol = sum(_volatility_proxy(c) * _mv(c) for c in ctxs) / total_mv

    return {
        "檔數": len(ctxs),
        "etf_pct": etf_mv / total_mv * 100,
        "tech_pct": tech_mv / total_mv * 100,
        "top_industry": top_industry,
        "top_industry_pct": top_industry_mv / total_mv * 100,
        "top_stock_pct": top_stock_mv / total_mv * 100,
        "weighted_pnl": w_pnl,
        "volatility": w_vol,            # 0.3≈溫和、0.6≈劇烈、1.0+≈雲霄飛車
        "total_mv": total_mv,
    }


# ─────────────────────────────────────────────────────────────────────
#  人格文案庫（完全對照企劃的 Meme 風格與吐槽語氣）
#  每個 key 對應：label / emoji / accent / tag / pr_label / roast
# ─────────────────────────────────────────────────────────────────────
PERSONAS = {
    "buddha": {
        "label": "佛系存股仙人", "emoji": "🧘", "short": "仙人",
        "accent": "#10b981", "accent2": "#065f46", "rarity": "SR",
        "pr_label": "心電圖平穩指數",
        "roast": "股市崩盤與我無關，我只在乎下個月的配息——你的心電圖，穩得像睡著的貓。",
    },
    "surfer": {
        "label": "心電圖衝浪客", "emoji": "🎢", "short": "衝浪客",
        "accent": "#ef4444", "accent2": "#7f1d1d", "rarity": "SSR",
        "pr_label": "心臟大顆指數",
        "roast": "不是在漲停板上狂歡，就是在跌停板上懷疑人生。你的心臟，是市場認證的大顆。",
    },
    "knife": {
        "label": "職業接刀手", "emoji": "🔪", "short": "接刀手",
        "accent": "#3b82f6", "accent2": "#1e3a8a", "rarity": "R",
        "pr_label": "無私流動性指數",
        "roast": "別人恐懼我貪婪，結果我不小心破產。沒關係，你為市場注入了無私的流動性。",
    },
    "silicon": {
        "label": "矽谷狂熱信徒", "emoji": "🤖", "short": "信徒",
        "accent": "#8b5cf6", "accent2": "#4c1d95", "rarity": "SSR",
        "pr_label": "庫存含輝量",
        "roast": "沒有 AI 的世界是不完整的！你的庫存含輝量極高，是見證科技革命的先驅。",
    },
    "prodigy": {
        "label": "隱藏版少年股神", "emoji": "👑", "short": "股神",
        "accent": "#f59e0b", "accent2": "#78350f", "rarity": "UR",
        "pr_label": "股神潛力值",
        "roast": "巴菲特看了都想請你吃午餐——這績效，低調都嫌浪費。",
    },
    # 混合機制常用的第二人格（對照企劃「睡得好投資家 / 賭徒綜合症」）
    "sleeper": {
        "label": "睡得好投資家", "emoji": "😴", "short": "睡得好",
        "accent": "#14b8a6", "accent2": "#134e4a", "rarity": "SR",
        "pr_label": "安穩睡眠指數",
        "roast": "帳戶漲跌都能安心睡覺，是投資界少見的低血壓體質。",
    },
    "gambler": {
        "label": "賭徒綜合症", "emoji": "🎰", "short": "賭徒",
        "accent": "#e11d48", "accent2": "#881337", "rarity": "SSR",
        "pr_label": "腎上腺素指數",
        "roast": "梭哈才是人生，分散是弱者的藉口。你的腎上腺素，隨著 K 線一起噴發。",
    },
    "balanced": {
        "label": "溫拿佛系散戶", "emoji": "🌱", "short": "均衡派",
        "accent": "#22c55e", "accent2": "#166534", "rarity": "N",
        "pr_label": "組合健康度",
        "roast": "不極端、不梭哈、不接刀——你是市場裡最稀有的正常人。",
    },
}


# ─────────────────────────────────────────────────────────────────────
#  人格判定 + 複合比例機制
# ─────────────────────────────────────────────────────────────────────
def _pr_fields(top_key: str, features: dict, market_ref: dict) -> dict:
    """
    依主人格的真實維度，對照全市場個股分布算出百分位。
    回傳 pr(int)、pr_prefix、pr_ref（PR 條下方的母體說明）。
    silicon 用「科技股市值佔比」直接呈現（非橫斷面百分位）。
    """
    vol = features["volatility"]          # 組合市值加權波動度
    pnl = features["weighted_pnl"]        # 組合市值加權帳面損益
    tech = features["tech_pct"]

    if top_key in ("buddha", "sleeper", "balanced"):
        pr = _percentile(vol, market_ref["vol"], higher_is_better=False)
        return {"pr": pr, "pr_prefix": "平穩勝過", "pr_ref": f"全市場 {market_ref['n_vol']} 檔個股波動"}
    if top_key in ("surfer", "gambler"):
        pr = _percentile(vol, market_ref["vol"], higher_is_better=True)
        return {"pr": pr, "pr_prefix": "刺激勝過", "pr_ref": f"全市場 {market_ref['n_vol']} 檔個股波動"}
    if top_key == "prodigy":
        pr = _percentile(pnl, market_ref["ret"], higher_is_better=True)
        return {"pr": pr, "pr_prefix": "績效贏過", "pr_ref": f"全市場 {market_ref['n_ret']} 檔個股年報酬"}
    if top_key == "knife":
        pr = _percentile(pnl, market_ref["ret"], higher_is_better=True)
        return {"pr": pr, "pr_prefix": "績效贏過", "pr_ref": f"全市場 {market_ref['n_ret']} 檔個股年報酬"}
    if top_key == "silicon":
        return {"pr": max(1, min(100, round(tech))), "pr_prefix": "含輝量",
                "pr_ref": "科技股市值佔比"}
    # 後備
    pr = _percentile(vol, market_ref["vol"], higher_is_better=True)
    return {"pr": pr, "pr_prefix": "勝過", "pr_ref": f"全市場 {market_ref['n_vol']} 檔個股"}


def determine_persona(features: dict, market_ref: dict = None) -> dict:
    """
    回傳：
      {
        primary, secondary(可為 None), ratio (primary 佔比 0~100),
        pr, pr_prefix, pr_ref, pr_label, roast, mix_text,
        accent, accent2, emoji, label, rarity
      }
    PR 以真實全市場個股分布計算（market_ref；未提供則自動建立並可由呼叫端快取）。
    複合比例機制：對每個候選人格算「強度分數」，取前二名，正規化成整數比例。
    """
    if market_ref is None:
        market_ref = build_market_reference()

    etf = features["etf_pct"]
    tech = features["tech_pct"]
    conc = max(features["top_stock_pct"], features["top_industry_pct"])
    pnl = features["weighted_pnl"]
    vol = features["volatility"]

    # 各人格強度分數（0~100 量級）——門檻對照企劃觸發條件
    scores = {
        "buddha": max(0.0, etf - 40) * 1.4,                 # ETF 佔比高 → 佛系
        "sleeper": max(0.0, etf - 30) * 0.9,                # 高股息底倉 → 睡得好
        "silicon": max(0.0, tech - 40) * 1.5,               # 科技/AI 佔比高
        "knife": max(0.0, -pnl - 8) * 4.2,                  # 深度虧損 → 接刀手（企劃：損益<-20% 主導）
        "prodigy": max(0.0, pnl - 12) * 3.4,                # 高績效 → 少年股神
        "surfer": max(0.0, conc - 45) * 0.7 + vol * 34,     # 集中 + 高波動 → 心電圖
        "gambler": max(0.0, conc - 55) * 0.8 + vol * 26,    # 極集中/高波動 → 賭徒
    }
    # 極端損益是企劃裡最鮮明的標籤，給明確加成確保它奪下主人格
    if pnl <= -20:
        scores["knife"] += 22
    if pnl >= 30:
        scores["prodigy"] += 22

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top_key, top_score = ranked[0]
    sec_key, sec_score = ranked[1]

    # 全部人格都沒被明顯觸發 → 均衡派
    if top_score < 8:
        p = PERSONAS["balanced"]
        prf = _pr_fields("balanced", features, market_ref)
        return {
            "primary": "balanced", "secondary": None, "ratio": 100,
            "pr_label": p["pr_label"], "roast": p["roast"],
            "mix_text": "100% 溫拿佛系散戶",
            "accent": p["accent"], "accent2": p["accent2"], "emoji": p["emoji"],
            "label": p["label"], "rarity": p["rarity"], **prf,
        }

    # 決定是否混第二人格：第二名要有一定強度
    blend = sec_score >= max(8.0, top_score * 0.35)
    if blend:
        a = top_score / (top_score + sec_score)
        ratio = int(round(a * 20)) * 5          # 湊成 5 的倍數
        ratio = min(90, max(55, ratio))         # 主人格介於 55~90%
        sec_ratio = 100 - ratio
    else:
        ratio, sec_ratio, sec_key = 100, 0, None

    p = PERSONAS[top_key]
    prf = _pr_fields(top_key, features, market_ref)

    if sec_key:
        mix_text = f"{ratio}% {PERSONAS[top_key]['short']} + {sec_ratio}% {PERSONAS[sec_key]['short']}"
    else:
        mix_text = f"100% {PERSONAS[top_key]['short']}"

    return {
        "primary": top_key, "secondary": sec_key, "ratio": ratio,
        "pr_label": p["pr_label"], "roast": p["roast"], "mix_text": mix_text,
        "accent": p["accent"], "accent2": p["accent2"], "emoji": p["emoji"],
        "label": p["label"], "rarity": p["rarity"], **prf,
    }


# ─────────────────────────────────────────────────────────────────────
#  成分比例（給卡片畫「成分條」用）：以市值把持股歸到人格傾向
# ─────────────────────────────────────────────────────────────────────
def composition_breakdown(ctxs: list, persona: dict) -> list:
    """回傳 [(名稱, 佔比%, 是否打碼顯示產業)]，供卡片列出前幾大成分。"""
    total = sum(_mv(c) for c in ctxs) or 1.0
    rows = []
    for c in sorted(ctxs, key=lambda x: -_mv(x)):
        pct = _mv(c) / total * 100
        if pct < 1:
            continue
        rows.append({
            "name": c.get("股票名稱", ""),
            "code": str(c.get("股票代號", "")),
            "industry": c.get("產業", "未知"),
            "pct": pct,
        })
    return rows


# ─────────────────────────────────────────────────────────────────────
#  卡片渲染：9:16 抽卡風格 SVG
# ─────────────────────────────────────────────────────────────────────
def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _wrap(text: str, width: int) -> list:
    """中文按字數硬換行（每行約 width 個字）。"""
    text = str(text)
    return [text[i:i + width] for i in range(0, len(text), width)] or [""]


def build_card_svg(
    persona: dict,
    features: dict,
    rows: list,
    mask_amount: bool = True,
    mask_code: bool = True,
    product_name: str = "AI 投資樹洞",
) -> str:
    """
    產生 9:16（360×640）抽卡風格卡片。
      mask_amount：不顯示任何金額（本卡本來就不放金額，此旗標保留為承諾展示）
      mask_code  ：把股票代號替換為產業類別
    """
    W, H = 360, 640
    acc, acc2 = persona["accent"], persona["accent2"]
    emoji = persona["emoji"]
    label = persona["label"]
    rarity = persona["rarity"]
    pr = persona["pr"]
    pr_label = persona["pr_label"]
    pr_prefix = persona.get("pr_prefix", "勝過")
    pr_ref = persona.get("pr_ref", "全市場個股")
    mix = persona["mix_text"]
    roast = persona["roast"]

    bar_w = W - 48
    pal = [acc, acc2, "#94a3b8"]

    # 成分條（取前 3 大）
    comp = rows[:3]
    comp_bar = ""
    comp_legend = ""
    x = 24
    acc_total = sum(r["pct"] for r in comp) or 1.0
    for i, r in enumerate(comp):
        seg = r["pct"] / acc_total * bar_w
        comp_bar += (f'<rect x="{x:.1f}" y="392" width="{max(0,seg-2):.1f}" height="12" '
                     f'rx="3" fill="{pal[i % 3]}"/>')
        x += seg
        shown = r["industry"] if mask_code else f'{r["name"]}'
        comp_legend += (
            f'<circle cx="26" cy="{422 + i*18}" r="4" fill="{pal[i % 3]}"/>'
            f'<text x="38" y="{426 + i*18}" fill="#cbd5e1" font-size="12">'
            f'{_esc(shown)} · {r["pct"]:.0f}%</text>'
        )

    # 吐槽文案：最多 3 行，文字框錨定固定頂端往下長，避免與成分圖例重疊
    roast_lines = _wrap(roast, 16)[:3]
    roast_top = 486
    roast_box_h = 20 + len(roast_lines) * 20
    roast_tspans = "".join(
        f'<tspan x="180" dy="{0 if i == 0 else 20}">{_esc(l)}</tspan>'
        for i, l in enumerate(roast_lines)
    )

    shield = (
        '<g transform="translate(24,598)">'
        '<text x="0" y="0" font-size="11" fill="#64748b">🛡️ 閱後即焚 · 僅提取代碼作測驗 · 不儲存金額</text>'
        '</g>'
    ) if mask_amount else ""

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="'Noto Sans TC',sans-serif">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#0b1220"/>
      <stop offset="1" stop-color="{acc2}"/>
    </linearGradient>
    <radialGradient id="halo" cx="0.5" cy="0.32" r="0.5">
      <stop offset="0" stop-color="{acc}" stop-opacity="0.55"/>
      <stop offset="1" stop-color="{acc}" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="prg" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="{acc}"/>
      <stop offset="1" stop-color="#fde68a"/>
    </linearGradient>
    <filter id="soft"><feDropShadow dx="0" dy="4" stdDeviation="8" flood-color="#000" flood-opacity="0.5"/></filter>
  </defs>

  <rect width="{W}" height="{H}" rx="26" fill="url(#bg)"/>
  <rect x="6" y="6" width="{W-12}" height="{H-12}" rx="22" fill="none" stroke="{acc}" stroke-opacity="0.55" stroke-width="1.5"/>
  <rect width="{W}" height="300" fill="url(#halo)"/>

  <!-- 稀有度徽章 -->
  <g transform="translate(24,34)">
    <rect x="0" y="-18" width="54" height="26" rx="13" fill="{acc}"/>
    <text x="27" y="0" text-anchor="middle" fill="#0b1220" font-size="14" font-weight="700">{_esc(rarity)}</text>
  </g>
  <text x="{W-24}" y="40" text-anchor="end" fill="#94a3b8" font-size="12" letter-spacing="2">INVESTMENT PERSONA</text>

  <!-- 主視覺 emoji 徽章 -->
  <circle cx="180" cy="138" r="70" fill="#0b1220" stroke="{acc}" stroke-width="3" filter="url(#soft)"/>
  <text x="180" y="168" text-anchor="middle" font-size="80">{emoji}</text>

  <!-- 人格標籤（最大字級）-->
  <text x="180" y="250" text-anchor="middle" fill="#f8fafc" font-size="32" font-weight="700">{_esc(label)}</text>
  <text x="180" y="276" text-anchor="middle" fill="{acc}" font-size="14" letter-spacing="3">{_esc(mix)}</text>

  <!-- PR 真實市場百分位（次大字級）-->
  <g transform="translate(24,292)">
    <text x="0" y="0" fill="#94a3b8" font-size="13">{_esc(pr_label)}</text>
    <rect x="0" y="10" width="{bar_w}" height="15" rx="7" fill="#1e293b"/>
    <rect x="0" y="10" width="{bar_w * pr / 100:.1f}" height="15" rx="7" fill="url(#prg)"/>
    <text x="0" y="56" fill="#f8fafc" font-size="26" font-weight="700">{_esc(pr_prefix)} {pr}%</text>
    <text x="0" y="76" fill="#94a3b8" font-size="12">對照 {_esc(pr_ref)}</text>
  </g>

  <!-- 成分比例 -->
  <text x="24" y="380" fill="#94a3b8" font-size="13">成分比例</text>
  {comp_bar}
  {comp_legend}

  <!-- 吐槽文案 -->
  <rect x="16" y="{roast_top}" width="{W-32}" height="{roast_box_h}" rx="12" fill="#0b1220" fill-opacity="0.6" stroke="{acc}" stroke-opacity="0.35"/>
  <text y="{roast_top + 26}" text-anchor="middle" fill="#e2e8f0" font-size="13" font-style="italic">{roast_tspans}</text>

  {shield}

  <!-- 病毒浮水印 CTA -->
  <line x1="24" y1="612" x2="{W-24}" y2="612" stroke="{acc}" stroke-opacity="0.3"/>
  <text x="180" y="632" text-anchor="middle" fill="#f8fafc" font-size="13" font-weight="500">你的庫存是什麼形狀？搜尋【{_esc(product_name)}】測測看</text>
</svg>'''


def build_share_text(persona: dict, product_name: str = "AI 投資樹洞") -> str:
    """一鍵複製的分享文案（Threads / Dcard 用）。"""
    return (
        f"我的投資人格是【{persona['label']}】🃏（{persona['mix_text']}）\n"
        f"{persona['pr_label']}：{persona.get('pr_prefix','勝過')} {persona['pr']}%（{persona.get('pr_ref','全市場個股')}）\n"
        f"{persona['roast']}\n"
        f"—— 你的庫存是什麼形狀？搜尋【{product_name}】30 秒測測看 👇"
    )


# ─────────────────────────────────────────────────────────────────────
#  選用：讓 LLM 加碼一句客製吐槽（失敗則回退靜態文案庫）
# ─────────────────────────────────────────────────────────────────────
def generate_ai_flavor(persona: dict, features: dict, engine: str = "auto") -> str:
    """呼叫既有 LLM 服務，產生一句更貼合此組合的專屬吐槽。失敗回退空字串。"""
    try:
        from llm_service import get_llm_service
        llm = get_llm_service()
        prompt = (
            "你是台灣股市社群的毒舌但幽默的 meme 寫手。"
            "根據以下投資人格與數據，寫『一句』繁體中文吐槽（30 字內，自嘲、好笑、可炫耀），"
            "不要加引號、不要說明、只輸出那一句：\n"
            f"人格：{persona['label']}（{persona['mix_text']}）\n"
            f"ETF佔比 {features['etf_pct']:.0f}%、科技佔比 {features['tech_pct']:.0f}%、"
            f"最大集中度 {max(features['top_stock_pct'], features['top_industry_pct']):.0f}%、"
            f"加權損益 {features['weighted_pnl']:+.0f}%、波動度 {features['volatility']:.2f}"
        )
        eng = engine
        if eng == "auto":
            eng = "aws" if llm.is_aws_available() else ("ollama" if llm.is_ollama_available() else None)
        if eng == "aws":
            out = llm.chat_bedrock(prompt)
        elif eng == "ollama":
            out = llm.chat_ollama(prompt)
        else:
            return ""
        out = (out or "").strip().strip("「」\"'。").splitlines()[0]
        return out if out and not out.startswith("⚠️") else ""
    except Exception:
        return ""
