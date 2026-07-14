"""
app.py
------
AI 投資樹洞 (AI Investment Companion) - Streamlit 主應用
三步驟：持股導入 → 組合診斷 → AI 陪伴報告
"""

import streamlit as st
import pandas as pd
import re
from datetime import datetime
from data_processor import get_processor, MKT_0050_RETURN
from bedrock_service import get_bedrock_service

st.set_page_config(
    page_title="AI 投資樹洞 | AI Investment Companion",
    page_icon="🌳", layout="wide", initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main-header { text-align: center; padding: 1rem 0; }
    .step-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem 1.5rem; border-radius: 10px; color: white; margin-bottom: 1rem; }
    .warn-line { border-left: 4px solid #d5493c; padding: 4px 12px; margin: 6px 0;
        background: #fdf3f2; border-radius: 4px; color: black; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    defaults = {
        "current_step": 1,
        "holdings": [],          # [{stock_id, cost, shares, buy_date}]
        "portfolio_context": None,
        "ai_report": None,       # AI 診斷快取（避免 rerun 重打 Bedrock）
        "ocr_result": None,      # OCR 結果暫存（修正巢狀按鈕 bug）
        "statement_result": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


def invalidate_analysis():
    """持股變動時清除舊分析與 AI 報告"""
    st.session_state.portfolio_context = None
    st.session_state.ai_report = None


# ===== OCR 模組 =====
def ocr_parse(uploaded_image) -> list:
    """OCR 解析看盤 App 截圖（未安裝 easyocr 時回傳模擬結果）"""
    try:
        import easyocr
        import numpy as np
        from PIL import Image

        reader = easyocr.Reader(["ch_tra", "en"])
        img_array = np.array(Image.open(uploaded_image))
        results = reader.readtext(img_array)
        full_text = " ".join(r[1] for r in results)

        processor = get_processor()
        valid_ids = set(processor.get_available_stocks()["股票代號"])
        # 4~6 位代號（涵蓋 00919 等 ETF），且必須在 300 檔資料庫內
        found = [c for c in re.findall(r"\b(\d{4,6})\b", full_text) if c in valid_ids]
        prices = re.findall(r"(\d+\.\d+)", full_text)
        parsed = []
        for i, code in enumerate(dict.fromkeys(found)):  # 去重保序
            parsed.append({
                "stock_id": code,
                "cost": float(prices[i]) if i < len(prices) else 0.0,
                "shares": 1000,
            })
        return parsed
    except ImportError:
        st.info("📌 EasyOCR 未安裝，使用模擬 OCR 結果展示功能")
        return [
            {"stock_id": "00919", "cost": 23.5, "shares": 10000},
            {"stock_id": "2886", "cost": 42.0, "shares": 3000},
        ]


# ===== 電子對帳單解析 =====
def parse_statement(uploaded_file) -> list:
    parsed = []
    if uploaded_file is None:
        return parsed
    file_type = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_type == "pdf":
            try:
                import pdfplumber
                with pdfplumber.open(uploaded_file) as pdf:
                    full_text = "".join(page.extract_text() or "" for page in pdf.pages)
                for line in full_text.split("\n"):
                    m = re.search(r"(\d{4,6})\s+\S+\s+(\d[\d,]*)\s+([\d.]+)", line)
                    if m:
                        parsed.append({"stock_id": m.group(1),
                                       "shares": int(m.group(2).replace(",", "")),
                                       "cost": float(m.group(3))})
            except ImportError:
                st.info("📌 pdfplumber 未安裝，使用模擬解析結果")
                parsed = _mock_statement_result()
        elif file_type in ("html", "htm"):
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            rows = re.findall(
                r"<td[^>]*>(\d{4,6})</td>\s*<td[^>]*>([^<]+)</td>\s*<td[^>]*>([\d,]+)</td>\s*<td[^>]*>([\d.]+)</td>",
                content)
            parsed = [{"stock_id": r[0], "shares": int(r[2].replace(",", "")), "cost": float(r[3])}
                      for r in rows]
            if not parsed:
                st.info("📌 無法從 HTML 解析到持股，使用模擬結果")
                parsed = _mock_statement_result()
        else:
            df = pd.read_csv(uploaded_file)
            for _, row in df.iterrows():
                sid = str(row.iloc[0]).strip()
                if re.match(r"^\d{4,6}$", sid):
                    parsed.append({"stock_id": sid,
                                   "shares": int(row.iloc[2]) if len(row) > 2 else 1000,
                                   "cost": float(row.iloc[3]) if len(row) > 3 else 0})
    except Exception as e:
        st.warning(f"解析發生錯誤：{e}，使用模擬資料")
        parsed = _mock_statement_result()
    return parsed


def _mock_statement_result() -> list:
    return [
        {"stock_id": "0056", "cost": 35.5, "shares": 10000},
        {"stock_id": "00878", "cost": 20.0, "shares": 15000},
        {"stock_id": "1101", "cost": 38.0, "shares": 2000},
    ]


def add_holdings(items, default_date):
    valid_ids = set(get_processor().get_available_stocks()["股票代號"])
    added, skipped = 0, []
    for item in items:
        if item["stock_id"] in valid_ids:
            st.session_state.holdings = [h for h in st.session_state.holdings
                                         if h["stock_id"] != item["stock_id"]]
            st.session_state.holdings.append({
                "stock_id": item["stock_id"], "cost": item["cost"],
                "shares": item["shares"], "buy_date": default_date})
            added += 1
        else:
            skipped.append(item["stock_id"])
    invalidate_analysis()
    return added, skipped


# ===== 主介面 =====
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🌳 AI 投資樹洞</h1>
        <p style="font-size: 1.1rem; color: #666;">AI Investment Companion — 你的專屬投資陪伴教練</p>
        <p style="font-size: 0.9rem; color: #999;">📅 系統時間基準：2025/12/31 ｜ 資料來源：CMoney（300 檔示範籃子）</p>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns(3)
    steps = ["📥 持股導入", "🔬 組合診斷", "💬 AI 陪伴報告"]
    for i, (col, step) in enumerate(zip([col1, col2, col3], steps), 1):
        with col:
            if i == st.session_state.current_step:
                st.markdown(f"**➤ 步驟 {i}：{step}**")
            elif i < st.session_state.current_step:
                st.markdown(f"✅ 步驟 {i}：{step}")
            else:
                st.markdown(f"⬜ 步驟 {i}：{step}")
    st.divider()

    if st.session_state.current_step == 1:
        render_step1_input()
    elif st.session_state.current_step == 2:
        render_step2_analysis()
    else:
        render_step3_output()


# ===== 步驟 1：持股導入 =====
def render_step1_input():
    st.markdown('<div class="step-header"><h3>📥 步驟 1：持股導入與健診入口</h3></div>', unsafe_allow_html=True)
    st.markdown("選一種方式導入你的真實持股。**輸入一檔就能健診**，不用一次填完。")

    tab1, tab2, tab3 = st.tabs(["✍️ 手動輸入", "📸 截圖 OCR 匯入", "📄 電子對帳單解析"])

    with tab1:
        processor = get_processor()
        available_stocks = processor.get_available_stocks()
# 「代號　名稱（產業）」選項清單：selectbox 內建打字即時過濾（輸入 00 或 兆豐 都能篩）
        stock_options = (
            available_stocks["股票代號"] + "　" + available_stocks["股票名稱"]
            + "（" + available_stocks["產業"].fillna("其他") + "）"
        ).tolist()
        with st.form("manual_input_form"):
            stock_choice = st.selectbox(
                "股票代號或名稱", stock_options, index=None,
                placeholder="輸入代號或名稱片段搜尋，例如：00、兆豐、台積",
                help="打字即時篩選，涵蓋 300 檔示範標的")
            col1, col2 = st.columns(2)
            with col1:
                cost_input = st.number_input("買進成本（元）", min_value=0.0, step=0.1, format="%.2f")
            with col2:
                shares_input = st.number_input("持有股數", min_value=0, step=1000, value=1000)
            buy_date_input = st.date_input("買進日期", value=datetime(2025, 6, 15),
                                           min_value=datetime(2020, 1, 1),
                                           max_value=datetime(2025, 12, 31))
            if st.form_submit_button("➕ 加入持股清單", use_container_width=True):
                if stock_choice is None:
                    st.error("請先從清單選擇一檔股票")
                elif cost_input <= 0:
                    st.error("請輸入買進成本")
                else:
                    stock_id = stock_choice.split("　")[0]
                    add_holdings(
                        [{"stock_id": stock_id, "cost": cost_input, "shares": shares_input}],
                        buy_date_input.strftime("%Y-%m-%d"))
                    st.success(f"✅ 已加入 {stock_choice}")

        with st.expander("🔍 查看資料庫中可用的股票代號"):
            st.dataframe(available_stocks[["股票代號", "股票名稱", "產業"]],
                         use_container_width=True, height=300)

    # --- Tab 2: OCR（結果存 session_state，修正巢狀按鈕 bug）---
    with tab2:
        st.markdown("上傳看盤 App 庫存截圖，AI 自動辨識後**由你確認**再匯入。")
        uploaded_image = st.file_uploader("上傳庫存截圖", type=["png", "jpg", "jpeg"], key="ocr_upload")
        if uploaded_image is not None:
            st.image(uploaded_image, caption="已上傳的截圖", width=400)
            if st.button("🔍 開始 OCR 辨識", key="ocr_btn"):
                with st.spinner("正在辨識圖片中的文字..."):
                    st.session_state.ocr_result = ocr_parse(uploaded_image)

        if st.session_state.ocr_result:
            st.success(f"✅ 辨識到 {len(st.session_state.ocr_result)} 筆，請確認或修正後匯入")
            edited = st.data_editor(pd.DataFrame(st.session_state.ocr_result),
                                    use_container_width=True, key="ocr_editor")
            if st.button("📥 確認並全部加入", key="ocr_add_all"):
                added, skipped = add_holdings(edited.to_dict("records"), "2025-06-01")
                st.session_state.ocr_result = None
                if skipped:
                    st.warning(f"已加入 {added} 筆；{', '.join(skipped)} 不在 300 檔資料庫中，已略過")
                st.rerun()

    # --- Tab 3: 對帳單（同樣改用 session_state）---
    with tab3:
        st.markdown("上傳券商電子對帳單（PDF/HTML/CSV），自動解析庫存。")
        uploaded_statement = st.file_uploader("上傳電子對帳單",
                                              type=["pdf", "html", "htm", "csv"], key="statement_upload")
        if uploaded_statement is not None:
            if st.button("📊 解析對帳單", key="parse_btn"):
                with st.spinner("正在解析對帳單..."):
                    st.session_state.statement_result = parse_statement(uploaded_statement)

        if st.session_state.statement_result:
            st.success(f"✅ 解析到 {len(st.session_state.statement_result)} 筆，請確認後匯入")
            edited = st.data_editor(pd.DataFrame(st.session_state.statement_result),
                                    use_container_width=True, key="stmt_editor")
            if st.button("📥 確認並全部加入", key="statement_add_all"):
                added, skipped = add_holdings(edited.to_dict("records"), "2025-01-01")
                st.session_state.statement_result = None
                if skipped:
                    st.warning(f"已加入 {added} 筆；{', '.join(skipped)} 不在 300 檔資料庫中，已略過")
                st.rerun()

    # --- 持股清單 ---
    st.divider()
    st.markdown("### 📋 目前持股清單")
    if st.session_state.holdings:
        df = pd.DataFrame(st.session_state.holdings)
        df.columns = ["股票代號", "買進成本", "持有股數", "買進日期"]
        st.dataframe(df, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ 清空持股清單", use_container_width=True):
                st.session_state.holdings = []
                invalidate_analysis()
                st.rerun()
        with col2:
            if st.button("🚀 開始 AI 健診", type="primary", use_container_width=True):
                st.session_state.current_step = 2
                st.rerun()
    else:
        st.info("尚未加入任何持股，請使用上方任一方式導入。")


# ===== 步驟 2：組合診斷 =====
def render_step2_analysis():
    st.markdown('<div class="step-header"><h3>🔬 步驟 2：組合層數據診斷</h3></div>', unsafe_allow_html=True)
    holdings = st.session_state.holdings
    if not holdings:
        st.warning("沒有持股資料，請返回步驟 1。")
        if st.button("⬅️ 返回步驟 1"):
            st.session_state.current_step = 1
            st.rerun()
        return

    if st.session_state.portfolio_context is None:
        with st.spinner("計算含息報酬、買點分位與社群脈動..."):
            st.session_state.portfolio_context = get_processor().build_portfolio_context(holdings)

    ctx = st.session_state.portfolio_context
    if ctx is None:
        st.error("❌ 無法取得任何持股的數據，請確認股票代號。")
        if st.button("⬅️ 返回步驟 1"):
            st.session_state.current_step = 1
            st.rerun()
        return

    ov = ctx["組合總覽"]
    st.markdown("#### 📒 組合存摺")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("總市值", f"{ov['總市值']:,}")
    c2.metric("帳面報酬", f"{ov['帳面報酬(%)']}%")
    c3.metric("含息總報酬", f"{ov['含息總報酬(%)']}%",
              help="帳面 + 2025 年度現金股利，存股族該看的數字")
    c4.metric("年股息現金流", f"{ov['年股息現金流']:,} 元")
    c5.metric("同期 0050", f"+{MKT_0050_RETURN}%")

    st.caption(f"產業配置：{'、'.join(f'{k} {v}%' for k, v in ov['產業配置(%)'].items())}"
               f"｜最大單一持股 {ov['最大單一持股權重(%)']}%")

    if ctx["系統警示"]:
        st.markdown("#### ⚠️ 系統警示")
        for w in ctx["系統警示"]:
            st.markdown(f'<div class="warn-line">{w}</div>', unsafe_allow_html=True)

    st.markdown("#### 📊 逐檔明細")
    for r in ctx["持股明細"]:
        with st.expander(f"{r['名稱']} ({r['代號']}) · 佔 {r['權重(%)']}%", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("帳面損益", f"{r['帳面損益(%)']}%")
            col2.metric("含息總報酬", f"{r['含息總報酬(%)']}%")
            col3.metric("成本殖利率", f"{r['成本殖利率(%)']}%")
            col4.metric("連續配息", f"{r['連續配息年數']} 年" if r["連續配息年數"] else "—")
            pctl = r["成本買點分位(%)"]
            pctl_str = pctl if isinstance(pctl, str) else (f"{pctl}%（0=年低 100=年高）" if pctl is not None else "—")
            st.markdown(f"**成本買點分位：** {pctl_str}　**年內高/低：** {r['年內高低'] or '—'}"
                        + ("　*(已排除拆分影響)*" if r["含拆分調整"] else ""))
            p = r["社群30日"]
            if p:
                st.markdown(f"**📢 同學會近 30 日：** 發文 {p['近30日發文']:,} 則"
                            f"（聲量 {p['聲量變化(%)']:+}%）· 看多 {p['看多']} / 看空 {p['看空']}"
                            f" · 多空比 {p['多空比']}")
            if r["下次除息日"]:
                st.markdown(f"**📅 下次除息：** {r['下次除息日']}")

    with st.expander("🧠 AI 理解包（Prompt Context）預覽"):
        st.json(ctx)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅️ 返回修改持股", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col2:
        if st.button("🤖 送出 AI 陪伴診斷", type="primary", use_container_width=True):
            st.session_state.current_step = 3
            st.rerun()


# ===== 步驟 3：AI 陪伴報告 =====
def render_step3_output():
    st.markdown('<div class="step-header"><h3>💬 步驟 3：AI 陪伴診斷報告</h3></div>', unsafe_allow_html=True)
    ctx = st.session_state.portfolio_context
    if not ctx:
        st.warning("沒有分析數據，請返回步驟 1。")
        if st.button("⬅️ 返回步驟 1"):
            st.session_state.current_step = 1
            st.rerun()
        return

    # AI 診斷結果快取：只在沒有快取時呼叫 Bedrock（一個組合一次呼叫）
    if st.session_state.ai_report is None:
        with st.spinner("🌳 AI 投資樹洞正在讀懂你的存摺..."):
            st.session_state.ai_report = get_bedrock_service().diagnose_portfolio(ctx)

    report = st.session_state.ai_report
    if report["ok"]:
        with st.chat_message("assistant", avatar="🌳"):
            st.markdown(report["text"])
    else:
        st.error(f"🚨 AI 診斷失敗，請排除後重試（不會顯示假結果）\n\n{report['error']}")

    st.markdown("---")
    st.markdown("### 🛡️ 下一步行動")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 重新產生診斷", use_container_width=True):
            st.session_state.ai_report = None
            st.rerun()
    with col2:
        if st.button("📥 修改持股", use_container_width=True):
            st.session_state.current_step = 1
            st.rerun()
    with col3:
        if st.button("🛡️ 開啟持股防護罩（訂閱）", type="primary", use_container_width=True):
            st.balloons()
            ex_dates = [f"{r['名稱']} {r['下次除息日']}" for r in ctx["持股明細"] if r["下次除息日"]]
            st.success("🎉 已開啟持股防護罩，我們會在關鍵時刻通知你。")
            st.markdown("**你將收到以下警示通知：**\n"
                        "- 📈 股價突破年度新高/新低時\n"
                        "- 🏦 法人連續大量買賣超時\n"
                        "- 💬 社群情緒劇烈變化時\n"
                        + (f"- 📅 除息提醒：{'、'.join(ex_dates)}" if ex_dates else "- 📅 除息日前 7 天提醒"))


# ===== 側邊欄 =====
def render_sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ 系統資訊")
        st.markdown("**系統時間基準：** 2025/12/31")
        st.markdown("**資料來源：** CMoney（300 檔）")
        st.markdown("**AI 模型：** Claude（Amazon Bedrock）")
        st.divider()
        st.markdown("### 📌 使用說明")
        st.markdown("1. **導入持股** — 手動 / OCR / 對帳單\n"
                    "2. **組合診斷** — 含息報酬、分位、社群脈動\n"
                    "3. **AI 報告** — 組合層個人化陪伴")
        st.divider()
        st.markdown("### ⚠️ 免責聲明")
        st.caption("本服務僅提供數據分析與陪伴視角，不構成任何投資建議。投資有風險，決策請自行判斷。")
        st.divider()
        if st.button("🔄 重置應用", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    render_sidebar()
    main()
