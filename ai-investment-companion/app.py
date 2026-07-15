"""
app.py - AI 投資樹洞 v5
升級：雙引擎切換(AWS Bedrock + Ollama)、背景預載AI、氣球金幣特效
"""
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import threading
from datetime import datetime
from data_processor import get_processor, get_shield_engine
from llm_service import get_llm_service
from ocr_service import ocr_parse_image, OCRError
import persona_service as persona

st.set_page_config(page_title="AI 投資樹洞", page_icon="🌳", layout="wide", initial_sidebar_state="expanded")


# ══════════ 快取 ══════════
@st.cache_data(ttl=600)
def cached_stock_list():
    p = get_processor()
    df = p.get_available_stocks()
    # 格式：代號 名稱 [產業]
    opts = df.apply(lambda r: f"{r['股票代號']} {r['股票名稱']} [{r['產業']}]", axis=1).tolist()
    return df, opts


@st.cache_data(ttl=30)
def cached_engine_status():
    llm = get_llm_service()
    return llm.get_status()


@st.cache_data(ttl=3600)
def cached_market_reference():
    """全市場個股波動度/年報酬分布，供投資人格卡算真實百分位（掃一次即快取）。"""
    return persona.build_market_reference(get_processor())


# ══════════ CSS（動態：字體大小 + 黑白模式）══════════
_fs_map = {"小": "14px", "中": "16px", "大": "18px", "特大": "21px"}
_fs = _fs_map.get(st.session_state.get("font_size", "中"), "16px")
_dark = st.session_state.get("dark_mode", True)

if _dark:
    _bg = "#0f172a"; _card_bg = "#1e293b"; _text = "#e2e8f0"; _border = "#334155"
    _sub_text = "#94a3b8"; _label = "#64748b"
    _input_bg = "#1e293b"
    _gd_bg = "#3b2c08"; _gd_border = "#b45309"; _gd_ll = "#fbbf24"; _gd_lv = "#fde68a"
else:
    _bg = "#ffffff"; _card_bg = "#ffffff"; _text = "#1e293b"; _border = "#e2e8f0"
    _sub_text = "#64748b"; _label = "#64748b"
    _input_bg = "#ffffff"
    _gd_bg = "#fffbeb"; _gd_border = "#fbbf24"; _gd_ll = "#92400e"; _gd_lv = "#78350f"

st.markdown(f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@300;400;500;700&display=swap');
*:not([data-testid="stIconMaterial"]):not(.material-symbols-rounded){{font-family:'Noto Sans TC',sans-serif!important; font-size:{_fs}!important}}
[data-testid="stIconMaterial"],.material-symbols-rounded{{font-family:'Material Symbols Rounded'!important}}
footer,#MainMenu{{visibility:hidden}}
.stApp,.block-container{{background:{_bg}}}
/* ── 文字全域（深淺模式一致覆蓋）── */
p,li,span,.stMarkdown,h1,h2,h3,h4,h5,h6,label{{color:{_text}!important}}
[data-testid="stCaptionContainer"] p{{color:{_sub_text}!important}}
hr{{border-color:{_border}!important}}
/* ── 側邊欄 ── */
section[data-testid="stSidebar"]{{background:{_card_bg};border-right:1px solid {_border}}}
/* ── Metric ── */
[data-testid="stMetricValue"]{{color:{_text}!important}}
[data-testid="stMetricLabel"] p{{color:{_sub_text}!important}}
/* ── 按鈕：secondary 用主題色，primary 固定深底金字（兩種模式都清楚）── */
header[data-testid="stHeader"]{{background:{_bg}}}
header[data-testid="stHeader"] *{{color:{_sub_text}}}
.stButton button,.stFormSubmitButton button,.stDownloadButton button{{background:{_card_bg};color:{_text};border:1px solid {_border}}}
.stButton button p,.stFormSubmitButton button p,.stDownloadButton button p{{color:inherit!important}}
.stButton button[kind="primary"],.stButton button[data-testid="stBaseButton-primary"]{{
  background:#1e293b;border-color:#fbbf24;color:#fbbf24}}
.stButton button[kind="primary"] p,.stButton button[data-testid="stBaseButton-primary"] p{{color:#fbbf24!important}}
/* ── 輸入框 / 下拉選單（含展開的選項清單）── */
.stTextInput input,.stNumberInput input,.stDateInput input,textarea,
[data-baseweb="select"]>div{{background:{_input_bg}!important;color:{_text}!important;border-color:{_border}!important}}
[data-baseweb="select"] div,[data-baseweb="select"] span{{color:{_text}!important}}
[data-baseweb="popover"] [role="listbox"]{{background:{_card_bg}!important;border:1px solid {_border}}}
[data-baseweb="popover"] [role="option"],[data-baseweb="popover"] [role="option"] *{{color:{_text}!important}}
/* ── st.info / st.success / st.warning：底色固定為淺色系，文字一律深色 ── */
[data-testid="stAlert"] p,[data-testid="stAlert"] span{{color:{_text}!important}}
[data-testid="stFileUploaderDropzone"]{{background:{_card_bg}!important;border:1px dashed {_border}!important}}
[data-testid="stFileUploaderDropzone"] span,[data-testid="stFileUploaderDropzone"] small,[data-testid="stFileUploaderDropzone"] div{{color:{_sub_text}!important}}
[data-testid="stFileUploaderDropzone"] button{{background:{_card_bg}!important;color:{_text}!important;border:1px solid {_border}!important}}
[data-testid="stFileUploaderDeleteBtn"] button{{color:{_sub_text}!important}}
[data-testid="stToast"]{{background:{_card_bg}!important;border:1px solid {_border}!important}}
[data-testid="stToast"] p,[data-testid="stToast"] span,[data-testid="stToast"] div{{color:{_text}!important}}
/* ── Tabs / Expander ── */
button[data-baseweb="tab"] p{{color:{_sub_text}!important}}
button[data-baseweb="tab"][aria-selected="true"] p{{color:{_text}!important}}
[data-testid="stExpander"] details{{background:{_card_bg};border:1px solid {_border}}}
/* ── Hero：本身就是固定深底，維持原樣 ── */
.hero{{text-align:center;padding:1.5rem 1rem 1rem;background:linear-gradient(135deg,#0f172a,#1e293b 60%,#334155);border-radius:16px;margin-bottom:1.3rem;color:#fff;border:1px solid #334155}}
.hero h1,.hero h1 span{{font-size:2rem!important;color:#fbbf24!important;margin:0;letter-spacing:1px}}
.hero .sub,.hero .sub span{{color:#94a3b8!important}}
.hero .sub{{font-size:.9em;margin-top:.3rem}}
.hero .pain,.hero .pain span{{color:#fcd34d!important}}
.hero .pain{{font-size:.82em;margin-top:.55rem;background:rgba(251,191,36,.12);display:inline-block;padding:4px 14px;border-radius:14px;border:1px solid rgba(251,191,36,.35)}}
.steps{{display:flex;gap:6px;margin-bottom:1.3rem}}
.sp{{flex:1;text-align:center;padding:.5rem 0;border-radius:20px;font-size:.82em;font-weight:500}}
.sp.a{{background:#fbbf24;color:#0f172a!important;font-weight:700;box-shadow:0 2px 8px rgba(251,191,36,.3)}}
.sp.a span,.sp.d span,.sp.p span{{color:inherit!important}}
.sp.d{{background:#bbf7d0;color:#14532d!important}}
.sp.p{{background:{('#334155' if _dark else '#f1f5f9')};color:{('#94a3b8' if _dark else '#94a3b8')}!important}}
/* ── 組合存摺卡片 ── */
.ldg{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:1.2rem}}
.li{{flex:1;min-width:130px;padding:.8rem .7rem;background:{_card_bg};border-radius:10px;border:1px solid {_border};text-align:center}}
.ll{{font-size:.7em;color:{_label}!important;margin-bottom:3px;text-transform:uppercase;letter-spacing:.5px}}
.lv{{font-size:1.1em;font-weight:700;color:{_text}!important}}
.lv.up{{color:#dc2626!important}}.lv.dn{{color:#16a34a!important}}
/* 金色卡片：背景與文字隨模式一起換（修正深色模式白字疊淺金底） */
.li.gd{{background:{_gd_bg};border-color:{_gd_border}}}
.li.gd .ll{{color:{_gd_ll}!important}}
.li.gd .lv{{color:{_gd_lv}!important}}
/* ── 分位標尺 ── */
.rail{{margin:.6rem 0 .3rem}}
.rb{{position:relative;height:8px;background:linear-gradient(to right,#22c55e 0%,#eab308 50%,#ef4444 100%);border-radius:4px;margin:16px 0 6px}}
.rm{{position:absolute;top:-7px;width:3px;height:22px;border-radius:2px;transform:translateX(-50%)}}
.rm.n{{background:{('#e2e8f0' if _dark else '#0f172a')}}}.rm.c{{background:#dc2626;opacity:.85}}
.rl{{display:flex;justify-content:space-between;font-size:.68em;color:{_sub_text}!important}}
.rl span,.rg span{{color:{_sub_text}!important}}
.rg{{display:flex;gap:10px;font-size:.68em;color:{_sub_text}!important;margin-top:2px}}
/* ── 警示條：底色固定淺色系，字色固定深色（兩種模式都清楚）── */
.alrt{{background:#fef2f2;border:1px solid #fecaca;border-left:4px solid #ef4444;border-radius:8px;padding:.7rem 1rem;margin:.4rem 0;font-size:.85em;color:#7f1d1d!important}}
.alrt.ec{{background:#fffbeb;border-color:#fde68a;border-left-color:#f59e0b;color:#78350f!important}}
/* ── AI 報告框 ── */
.abox{{background:{('#1e293b' if _dark else '#f8fafc')};border-left:4px solid #fbbf24;border-radius:0 10px 10px 0;padding:1.2rem 1.4rem;margin:.8rem 0;line-height:1.85;font-size:.9em;color:{_text}!important;white-space:pre-wrap}}
.persona-gate{{text-align:center;padding:1.6rem 1rem;margin:.4rem 0 1rem;border-radius:18px;
  background:linear-gradient(135deg,#0b1220,#312e81 130%);border:1px solid #4c1d95;color:#fff!important}}
.persona-gate .pg-emoji{{font-size:2.6rem;animation:pgfloat 2.4s ease-in-out infinite}}
.persona-gate .pg-title,.persona-gate .pg-title span{{font-size:1.35rem!important;font-weight:700;color:#fef3c7!important;margin:.3rem 0 .2rem}}
.persona-gate .pg-sub,.persona-gate .pg-sub span{{font-size:.85rem!important;color:#c7d2fe!important}}
@keyframes pgfloat{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-6px)}}}}
.sc{{background:{_card_bg};border:1px solid {_border};border-radius:12px;padding:1rem;margin-bottom:.8rem}}
.hcard{{background:{_card_bg};border:1px solid {_border};border-radius:10px;padding:.55rem .9rem;margin:.35rem 0}}
.hcard code{{background:transparent;color:#fbbf24!important;font-weight:700}}
.hcard .hind,.hcard .hind span{{color:{_sub_text}!important;font-size:.82em}}
.hcard .hins{{margin-top:.25rem;font-size:.8em;color:{_sub_text}!important}}
.hcard .hins b{{color:{_text}!important}}
.ins.up{{color:#dc2626!important;font-weight:700}}
.ins.dn{{color:#16a34a!important;font-weight:700}}
.ins.gd{{color:#d97706!important;font-weight:700}}
.ibars{{background:{_card_bg};border:1px solid {_border};border-radius:12px;padding:.9rem 1.1rem}}
.irow{{display:flex;align-items:center;gap:10px;margin:.35rem 0}}
.irow .ilabel,.irow .ilabel span{{color:{_text}!important}}
.irow .ilabel{{flex:0 0 130px;font-size:.85em;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.irow .itrack{{flex:1;height:10px;background:{_border};border-radius:5px;overflow:hidden;display:block}}
.irow .ifill{{display:block;height:100%;background:linear-gradient(90deg,#fbbf24,#d97706);border-radius:5px}}
.irow .ipct,.irow .ipct span{{color:{_text}!important}}
.irow .ipct{{flex:0 0 52px;text-align:right;font-weight:700;font-size:.85em}}
.irow .imv,.irow .imv span{{color:{_sub_text}!important}}
.irow .imv{{flex:0 0 100px;text-align:right;font-size:.75em}}
.hest,.hest span{{color:#d97706!important}}
.hest{{font-size:.72em;border:1px dashed #d97706;border-radius:8px;padding:1px 7px;margin-left:4px}}
.hcard .hpeer,.hcard .hpeer span{{color:{_sub_text}!important}}
.hcard .hpeer{{margin-top:.25rem;font-size:.78em;font-style:italic}}
.sc h4{{margin:0 0 .5rem;font-size:1em;color:{_text}!important}}
/* 氣球特效 iframe：固定全螢幕覆蓋層，脫離文件流不佔版面，點擊穿透 */
[data-testid="stIFrame"]{{position:fixed!important;inset:0!important;width:100vw!important;height:100vh!important;
  z-index:9999!important;pointer-events:none!important;border:0!important;background:transparent!important}}
.engine-badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.7em;font-weight:600}}
.engine-badge.aws{{background:#dbeafe;color:#1e40af!important}}
.engine-badge.ollama{{background:#fef3c7;color:#92400e!important}}
</style>""", unsafe_allow_html=True)


# ══════════ 氣球金幣特效 ══════════
BALLOON_COIN_HTML = """
<div id="balloon-game" style="position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:9999;pointer-events:none;">
<canvas id="bc-canvas" style="width:100%;height:100%;"></canvas>
</div>
<script>
(function(){
const canvas = document.getElementById('bc-canvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;

const balloons = [];
const coins = [];
const colors = ['#ef4444','#f59e0b','#3b82f6','#8b5cf6','#ec4899','#10b981'];

// 生成氣球
for(let i=0;i<12;i++){
    balloons.push({
        x: Math.random()*canvas.width,
        y: canvas.height + Math.random()*200 + 50,
        r: 28 + Math.random()*18,
        color: colors[Math.floor(Math.random()*colors.length)],
        speed: 1.2 + Math.random()*1.5,
        wobble: Math.random()*Math.PI*2,
        alive: true
    });
}

// 自動爆破（特效改為覆蓋層後點擊會穿透到頁面，故氣球升到畫面中改為自動爆開成金幣）
function pop(b){
    b.alive = false;
    for(let j=0;j<6;j++){
        coins.push({
            x: b.x, y: b.y,
            vx: (Math.random()-0.5)*8,
            vy: -Math.random()*6 - 2,
            size: 10+Math.random()*8,
            rotation: Math.random()*360,
            alpha: 1,
            gravity: 0.3
        });
    }
}
setInterval(function(){
    const alive = balloons.filter(b => b.alive && b.y < canvas.height*0.75);
    if(alive.length) pop(alive[Math.floor(Math.random()*alive.length)]);
}, 450);
// 視窗尺寸變化時同步畫布（覆蓋層以 CSS 撐滿全螢幕）
window.addEventListener('resize', function(){
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
});

function drawBalloon(b){
    ctx.beginPath();
    ctx.ellipse(b.x, b.y, b.r*0.8, b.r, 0, 0, Math.PI*2);
    ctx.fillStyle = b.color;
    ctx.fill();
    // 光澤
    ctx.beginPath();
    ctx.ellipse(b.x-b.r*0.25, b.y-b.r*0.3, b.r*0.2, b.r*0.3, -0.5, 0, Math.PI*2);
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.fill();
    // 繩子
    ctx.beginPath();
    ctx.moveTo(b.x, b.y+b.r);
    ctx.lineTo(b.x+Math.sin(b.wobble)*5, b.y+b.r+25);
    ctx.strokeStyle = '#666';
    ctx.lineWidth = 1;
    ctx.stroke();
}

function drawCoin(c){
    ctx.save();
    ctx.translate(c.x, c.y);
    ctx.rotate(c.rotation*Math.PI/180);
    ctx.globalAlpha = c.alpha;
    ctx.beginPath();
    ctx.arc(0, 0, c.size, 0, Math.PI*2);
    ctx.fillStyle = '#fbbf24';
    ctx.fill();
    ctx.strokeStyle = '#d97706';
    ctx.lineWidth = 2;
    ctx.stroke();
    // $ 符號
    ctx.fillStyle = '#92400e';
    ctx.font = `bold ${c.size}px sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('$', 0, 1);
    ctx.restore();
}

let frame = 0;
function animate(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    let anyAlive = false;

    balloons.forEach(b => {
        if(!b.alive) return;
        anyAlive = true;
        b.y -= b.speed;
        b.wobble += 0.03;
        b.x += Math.sin(b.wobble)*0.8;
        drawBalloon(b);
    });

    coins.forEach(c => {
        c.x += c.vx;
        c.y += c.vy;
        c.vy += c.gravity;
        c.rotation += 5;
        c.alpha -= 0.012;
        if(c.alpha > 0) drawCoin(c);
    });

    frame++;
    // 8秒後或全部破完+金幣消失 → 移除
    if(frame > 480 || (!anyAlive && coins.every(c=>c.alpha<=0))){
        canvas.parentElement.remove();
        return;
    }
    requestAnimationFrame(animate);
}
animate();
})();
</script>
"""


# ══════════ 工具函式 ══════════
def mk_rail(yl, yh, now_p, cost_p):
    if yl is None or yh is None:
        return ""
    n = max(0, min(100, now_p)) if now_p is not None else None
    c = max(0, min(100, cost_p)) if cost_p is not None else None
    markers = ""
    if n is not None:
        markers += f'<div class="rm n" style="left:{n}%"></div>'
    if c is not None:
        markers += f'<div class="rm c" style="left:{c}%"></div>'
    nl = f"{now_p:.0f}%" if now_p is not None else "—"
    cl = f"{cost_p:.0f}%" if cost_p is not None else "—"
    return (f'<div class="rail"><div class="rb">{markers}</div>'
            f'<div class="rl"><span>${yl:.1f}</span><span>${yh:.1f}</span></div>'
            f'<div class="rg"><span>■ 現價 {nl}</span>'
            f'<span style="color:#dc2626">■ 成本 {cl}</span></div></div>')


def fmoney(v):
    if v is None or v == 0:
        return "—"
    if abs(v) >= 1e8:
        return f"${v/1e8:.1f}億"
    if abs(v) >= 1e4:
        return f"${v/1e4:.1f}萬"
    return f"${v:,.0f}"


def pcls(v):
    if v is None:
        return ""
    return "up" if v >= 0 else "dn"


# ══════════ Session State ══════════
for _k, _v in [("step", 1), ("holdings", []), ("contexts", []), ("alerts", []),
               ("engine", "auto"), ("ai_prefetch_done", False), ("show_coins", False),
               ("font_size", "中"), ("dark_mode", True), ("ocr_records", None),
               ("persona_revealed", False), ("persona_mask_code", True),
               ("persona_ai_flavor", False)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ══════════ 背景預載 AI ══════════
def _prefetch_ai_in_background():
    """
    在背景 thread 中呼叫 AI，結果寫入 st.session_state。
    Streamlit 的 session_state 在同一 session 中是 thread-safe 的。
    """
    ctxs = st.session_state.get("contexts", [])
    alerts = st.session_state.get("alerts", [])
    engine = st.session_state.get("engine", "auto")
    llm = get_llm_service()

    if ctxs and "ai_portfolio" not in st.session_state:
        st.session_state["ai_portfolio"] = llm.diagnose_portfolio(ctxs, alerts, engine=engine)

    st.session_state["ai_prefetch_done"] = True


def _start_prefetch_thread():
    """啟動背景預載（如果還沒跑的話）"""
    if st.session_state.get("ai_prefetch_done"):
        return
    if st.session_state.get("_prefetch_started"):
        return
    st.session_state["_prefetch_started"] = True
    t = threading.Thread(target=_prefetch_ai_in_background, daemon=True)
    t.start()


# ══════════ MAIN ══════════
def main():
    # 引擎狀態 badge
    status = cached_engine_status()
    if status["aws_available"]:
        engine_label = "☁️ AWS Bedrock"
        badge_cls = "aws"
    elif status["ollama_available"]:
        engine_label = "🖥️ Ollama"
        badge_cls = "ollama"
    else:
        engine_label = "⚠️ 無 AI"
        badge_cls = "ollama"

    st.markdown(f"""<div class="hero"><h1>🌳 AI 投資樹洞</h1>
    <div class="sub">敞開庫存，讓 AI 陪你面對每一次波動</div>
    <div class="pain">📉 2025 年高股息 ETF 平均落後大盤 18%——你的存股，還好嗎？</div>
    <div style="font-size:.72rem;color:#6a8caf;margin-top:.3rem">
    📅 2025/12/31 ｜ <span class="engine-badge {badge_cls}">{engine_label}</span> ｜ 📊 300 檔標的
    </div></div>""", unsafe_allow_html=True)

    s = st.session_state.step
    c1 = "d" if s > 1 else ("a" if s == 1 else "p")
    c2 = "d" if s > 2 else ("a" if s == 2 else "p")
    c3 = "a" if s == 3 else "p"
    st.markdown(f"""<div class="steps">
    <div class="sp {c1}">{"✅" if s>1 else "①"} 持股導入</div>
    <div class="sp {c2}">{"✅" if s>2 else "②"} 數據分析</div>
    <div class="sp {c3}">③ AI 診斷</div></div>""", unsafe_allow_html=True)

    [page_input, page_analysis, page_ai][s - 1]()


# ══════════ 步驟 1 ══════════
def page_input():
    if st.session_state.get("_added_msg"):
        st.toast(st.session_state.pop("_added_msg"), icon="✅")
    st.markdown("### 📥 導入持股")
    st.caption("🔒 你的持股僅存在本次瀏覽階段——不綁券商帳號、不上傳雲端、不需註冊，關閉頁面即清除，也可隨時按「清空」。")
    tab1, tab2 = st.tabs(["✍️ 手動輸入", "📸 截圖 OCR"])

    with tab1:
        _, stock_opts = cached_stock_list()
        selected = st.selectbox(
            "🔍 輸入代號或名稱搜尋",
            options=stock_opts,
            index=None,
            placeholder="輸入股票代號或名稱搜尋...",
            help="支援前綴搜尋，輸入 '00' 列出 ETF，'23' 列出台積電等",
        )

        with st.form("add_form", clear_on_submit=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                cost = st.number_input("買進均價（元）｜留 0 = 用年均價估", min_value=0.0, value=0.0, step=1.0,
                                       format="%.2f", help="不記得成本？留 0，我們先用該股 2025 年均價估算，之後隨時可修正")
            with c2:
                shares = st.number_input("股數", min_value=1, value=1000, step=1000)
            with c3:
                bdate = st.date_input("買進日期", value=datetime(2025, 6, 15),
                                      min_value=datetime(2020, 1, 1), max_value=datetime(2025, 12, 31))
            if st.form_submit_button("➕ 加入持股", use_container_width=True):
                if selected:
                    sid = selected.split(" ")[0]
                    est = False
                    if cost <= 0:
                        avg = get_processor().get_year_avg_price(sid)
                        if avg is None:
                            st.warning("查無此股價格資料，請手動輸入成本")
                            st.stop()
                        cost, est = avg, True
                    lot_n = sum(1 for x in st.session_state.holdings if x["stock_id"] == sid) + 1
                    st.session_state.holdings.append({
                        "stock_id": sid, "cost": cost, "cost_estimated": est,
                        "shares": shares, "buy_date": bdate.strftime("%Y-%m-%d"),
                    })
                    st.session_state["_added_msg"] = (
                        f"已加入 {selected}" + (f" 第 {lot_n} 筆" if lot_n > 1 else "")
                        + f"（${cost:.2f} × {shares:,} 股" + ("，成本為年均價估算" if est else "") + "）")
                    st.rerun()
                else:
                    st.warning("請先從搜尋框選擇一檔股票")

    with tab2:
        st.caption("上傳看盤 App 截圖（庫存頁或對帳單），AI 視覺辨識後由你確認再匯入")
        img = st.file_uploader("　", type=["png", "jpg", "jpeg"], key="ocr_uploader")
        if img:
            st.image(img, width=300)
            if st.button("🔍 AI 辨識截圖"):
                with st.spinner("AI 正在讀取截圖..."):
                    st.session_state.ocr_records = _run_image_extraction(img)
                    st.rerun()

        if st.session_state.get("ocr_records") is not None:
            recs, skipped, engine_used = st.session_state.ocr_records
            if recs:
                st.success(f"✅ {engine_used} 辨識到 {len(recs)} 筆交易，每筆為獨立持股批次，可修改後匯入（成本留 0 以年均價估算）")
                edited = st.data_editor(
                    pd.DataFrame(recs)[["stock_id", "name", "cost", "shares", "buy_date"]].rename(
                        columns={"stock_id": "代號", "name": "名稱", "cost": "成本",
                                 "shares": "股數", "buy_date": "買進日期"}),
                    use_container_width=True, key="ocr_editor", num_rows="dynamic")
                if skipped:
                    st.warning("以下標的不在 300 檔示範資料庫，已略過：" + "、".join(skipped))
                if st.button("📥 確認並匯入", type="primary", use_container_width=True):
                    # 重匯同一張截圖不重複：先移除本次匯入涉及股票的既有批次
                    sids_in_import = {str(r["代號"]).strip() for _, r in edited.iterrows()
                                      if str(r["代號"]).strip() and str(r["代號"]).strip() != "nan"}
                    st.session_state.holdings = [x for x in st.session_state.holdings
                                                 if x["stock_id"] not in sids_in_import]
                    n = 0
                    for _, row in edited.iterrows():
                        sid = str(row["代號"]).strip()
                        if not sid or sid == "nan":
                            continue
                        cost = float(row["成本"]) if pd.notna(row["成本"]) and row["成本"] else 0.0
                        est = False
                        if cost <= 0:
                            avg = get_processor().get_year_avg_price(sid)
                            if avg is None:
                                continue
                            cost, est = avg, True
                        bd = str(row["買進日期"]).strip()[:10] if pd.notna(row["買進日期"]) and str(row["買進日期"]).strip() else "2025-06-01"
                        st.session_state.holdings.append({
                            "stock_id": sid, "cost": cost, "cost_estimated": est,
                            "shares": int(row["股數"]) if pd.notna(row["股數"]) and row["股數"] else 1000,
                            "buy_date": bd})
                        n += 1
                    st.session_state.ocr_records = None
                    st.session_state["_added_msg"] = f"已從截圖匯入 {n} 筆持股批次"
                    st.rerun()
            else:
                st.warning("未辨識到可對應的股票。" + ("；".join(skipped) if skipped else ""))
                st.session_state.ocr_records = None

    st.markdown("---")
    _show_holdings()


def _run_image_extraction(img) -> tuple:
    """截圖抽取：Bedrock 視覺優先，EasyOCR 備援；名稱/代號對照 300 檔資料庫"""
    stocks_db = get_processor().get_available_stocks()
    code_set = set(stocks_db["股票代號"])
    name_to_code = dict(zip(stocks_db["股票名稱"], stocks_db["股票代號"]))

    def resolve(rec):
        """把辨識結果對應到資料庫代號：代號直接比對 → 名稱完全相符 → 名稱包含"""
        sid = rec.get("stock_id")
        if sid and sid in code_set:
            return sid
        nm = (rec.get("name") or "").replace(" ", "")
        if nm in name_to_code:
            return name_to_code[nm]
        for db_name, code in name_to_code.items():
            if nm and (nm in db_name or db_name in nm):
                return code
        return None

    raw, engine_used = [], ""
    # 1) Bedrock 視覺（主力）
    llm = get_llm_service()
    if llm.is_aws_available():
        try:
            fmt = "png" if img.name.lower().endswith(".png") else "jpeg"
            raw = llm.extract_holdings_from_image(img.getvalue(), fmt)
            engine_used = "☁️ Bedrock 視覺"
        except Exception as e:
            st.info(f"Bedrock 視覺辨識失敗（{type(e).__name__}），改用本地 OCR")
    # 2) EasyOCR 備援
    if not raw:
        try:
            parsed, _ = ocr_parse_image(img)
            raw = [{"name": "", "stock_id": r["stock_id"],
                    "shares": r["shares"], "cost": r["cost"]} for r in parsed]
            engine_used = "🖥️ 本地 OCR"
        except OCRError:
            return [], [], engine_used or "辨識"

    # 逐筆保留：每筆交易的成本/股數/買進日期都不同，是獨立的持股批次（命題 INPUT 規格含買進日期）
    records, skipped = [], []
    for rec in raw:
        sid = resolve(rec)
        label = rec.get("name") or rec.get("stock_id") or "?"
        if sid is None:
            if label not in skipped:
                skipped.append(label)
            continue
        db_name = stocks_db.loc[stocks_db["股票代號"] == sid, "股票名稱"].iloc[0]
        records.append({"stock_id": sid, "name": db_name,
                        "cost": rec.get("cost") or 0.0,
                        "shares": rec.get("shares") or 1000,
                        "buy_date": rec.get("buy_date") or ""})
    return records, skipped, engine_used


DEMO_PORTFOLIO = [
    {"stock_id": "00919", "cost": 23.5, "shares": 10000, "buy_date": "2025-06-01"},
    {"stock_id": "2886",  "cost": 42.0, "shares": 3000,  "buy_date": "2025-03-15"},
    {"stock_id": "1101",  "cost": 38.0, "shares": 2000,  "buy_date": "2023-08-01"},
]


def _instant_insight(item: dict, s: dict) -> str:
    """加入持股當下的即時洞察（一檔就有回饋）"""
    bits = []
    close, div = s.get("收盤價"), s.get("現金股利")
    if close and item["cost"] > 0:
        pnl = (close / item["cost"] - 1) * 100
        cls = "up" if pnl >= 0 else "dn"
        bits.append(f'帳面 <span class="ins {cls}">{pnl:+.1f}%</span>')
        if div:
            total = ((close + div) / item["cost"] - 1) * 100
            cls2 = "up" if total >= 0 else "dn"
            bits.append(f'含息 <span class="ins {cls2}">{total:+.1f}%</span>')
            bits.append(f'成本殖利率 <span class="ins gd">{div / item["cost"] * 100:.2f}%</span>')
    try:
        pctl = get_processor().calculate_percentile(item["cost"], item["stock_id"])
        if pctl is not None:
            bits.append(f"成本在年內 <b>{pctl:.0f}</b> 分位")
    except Exception:
        pass
    yrs = s.get("連續配息年數")
    if yrs and yrs > 0:
        bits.append(f"連續配息 <b>{yrs:.0f}</b> 年")
    trend = s.get("股利連N年遞增")
    if trend and trend >= 2:
        bits.append(f'股利金額<span class="ins up">連 {trend:.0f} 年遞增</span>')
    elif trend and trend <= -2:
        bits.append(f'股利金額<span class="ins dn">連 {abs(trend):.0f} 年遞減</span>')
    return "　·　".join(bits)


def _peer_note(item: dict, s: dict) -> str:
    """匿名同儕對照：正常化虧損感（樹洞定位的靈魂），僅在使用者帳面虧損且該股落後大盤時顯示"""
    close = s.get("收盤價")
    if not close or item["cost"] <= 0:
        return ""
    pnl = (close / item["cost"] - 1) * 100
    vs_mkt = s.get("與大盤比")
    if pnl > -10 or vs_mkt is None or vs_mkt > -10:
        return ""
    stats = get_processor().get_forum_year_stats(item["stock_id"])
    if not stats or stats["全年發文"] < 1000:
        return ""
    posts = stats["全年發文"]
    posts_txt = f"{posts/10000:.1f} 萬" if posts >= 10000 else f"{posts:,}"
    return (f"2025 年{s['股票名稱']}同學會有 {posts_txt} 則討論、"
            f"全年落後大盤 {abs(vs_mkt):.0f}%——套牢的不只你，AI 會陪你一起看該怎麼想")


def _remove_holding(idx: int, name: str):
    """移除單一持股批次（以索引），並清除所有依賴舊組合的下游快取"""
    st.session_state.holdings = [x for i, x in enumerate(st.session_state.holdings)
                                 if i != idx]
    st.session_state.contexts = []
    st.session_state.alerts = []
    for k in [k for k in st.session_state if k.startswith("ai_")]:
        del st.session_state[k]
    st.session_state.ai_prefetch_done = False
    st.session_state["_prefetch_started"] = False
    st.session_state.shield_on = False
    st.session_state.shield_alerts = None
    st.session_state["_added_msg"] = f"已移除一筆 {name}"


def _show_holdings():
    h = st.session_state.holdings
    if not h:
        st.info("👆 選一檔股票填入成本與股數——**一檔就能健診**，不用一次填完。")
        if st.button("👀 還沒準備好？一鍵載入示範組合看看別人的健診", use_container_width=True):
            st.session_state.holdings = [dict(x) for x in DEMO_PORTFOLIO]
            st.rerun()
        return

    st.markdown(f"**📋 持股清單（{len(h)} 檔）**")
    p = get_processor()
    for idx, item in enumerate(h):
        s = p.get_stock_summary(item["stock_id"])
        nm = s["股票名稱"] if s else ""
        ind = s["產業"] if s else ""
        insight = _instant_insight(item, s) if s else ""
        est_tag = ' <span class="hest">年均價估算，可修正</span>' if item.get("cost_estimated") else ""
        peer = _peer_note(item, s) if s else ""
        col_card, col_del = st.columns([12, 1])
        with col_card:
            st.markdown(
                f'<div class="hcard"><div class="hmain"><code>{item["stock_id"]}</code> '
                f'<b>{nm}</b> <span class="hind">[{ind}]</span> — ${item["cost"]:.2f}{est_tag} × {item["shares"]:,} 股'
                f' <span class="hind">({item.get("buy_date", "")} 買進)</span></div>'
                + (f'<div class="hins">⚡ {insight}</div>' if insight else "")
                + (f'<div class="hpeer">🫂 {peer}</div>' if peer else "")
                + "</div>", unsafe_allow_html=True)
        with col_del:
            if st.button("✕", key=f"del_{idx}_{item['stock_id']}", help=f"移除這筆 {nm}", use_container_width=True):
                _remove_holding(idx, nm)
                st.rerun()

    c1, c3 = st.columns([1, 3])
    with c1:
        if st.button("🗑️ 清空", use_container_width=True):
            st.session_state.holdings = []
            st.session_state.contexts = []
            st.session_state.alerts = []
            st.rerun()
    with c3:
        if st.button("🚀 開始健診", type="primary", use_container_width=True):
            # 只做數據計算（快），AI 留給步驟 2 背景跑
            _compute_contexts()
            st.session_state.ai_prefetch_done = False
            st.session_state["_prefetch_started"] = False
            st.session_state.step = 2
            st.rerun()


def _compute_contexts():
    """只計算 contexts 和 alerts（不等 AI）"""
    p = get_processor()
    ctxs = []
    for h in st.session_state.holdings:
        ctx = p.build_ai_context(h["stock_id"], h["cost"], h["shares"], h.get("buy_date", ""))
        if ctx:
            ctxs.append(ctx)
    alerts = p.compute_portfolio_alerts(ctxs) if ctxs else []
    st.session_state.contexts = ctxs
    st.session_state.alerts = alerts
    # 清除舊的 AI 回覆
    for k in [k for k in st.session_state if k.startswith("ai_")]:
        del st.session_state[k]
    # 組合變了 → 重置投資人格卡，讓使用者重新翻牌
    st.session_state.persona_revealed = False
    for k in [k for k in st.session_state if k.startswith("persona_flavor_")]:
        del st.session_state[k]


# ══════════ 步驟 2 ══════════
def page_analysis():
    ctxs = st.session_state.contexts
    alerts = st.session_state.alerts
    if not ctxs:
        st.warning("無持股數據")
        if st.button("⬅️ 返回"):
            st.session_state.step = 1
            st.rerun()
        return

    # ★ 背景偷跑 AI（使用者一邊看數據，AI 一邊在跑）
    _start_prefetch_thread()

    # Ledger
    total_val = sum(c["市值"] for c in ctxs if c["市值"])
    total_cost = sum(c["買進成本"] * c["持有股數"] for c in ctxs)
    pnl_pct = round((total_val / total_cost - 1) * 100, 2) if total_cost > 0 else 0
    total_div = sum(c["年股息現金流"] for c in ctxs if c["年股息現金流"])
    total_wd = sum((c["收盤價"] + (c["現金股利"] or 0)) * c["持有股數"] for c in ctxs if c["收盤價"])
    tr_pct = round((total_wd / total_cost - 1) * 100, 2) if total_cost > 0 else 0

    st.markdown(f"""<div class="ldg">
    <div class="li"><div class="ll">總市值</div><div class="lv">{fmoney(total_val)}</div></div>
    <div class="li"><div class="ll">帳面報酬</div><div class="lv {pcls(pnl_pct)}">{pnl_pct:+.2f}%</div></div>
    <div class="li"><div class="ll">含息總報酬</div><div class="lv {pcls(tr_pct)}">{tr_pct:+.2f}%</div></div>
    <div class="li gd"><div class="ll">年股息現金流</div><div class="lv">{fmoney(total_div)}</div></div>
    <div class="li"><div class="ll">同期 0050</div><div class="lv up">+36.9%</div></div>
    </div>""", unsafe_allow_html=True)

    # 警示
    if alerts:
        for a in alerts:
            cls = "ec" if a["type"] == "echo_chamber" else ""
            st.markdown(f'<div class="alrt {cls}">{a["message"]}</div>', unsafe_allow_html=True)
        st.markdown("")

    # 個股卡片
    for ctx in ctxs:
        _render_stock_card(ctx)

    # 預載狀態提示
    if st.session_state.get("ai_prefetch_done"):
        st.caption("✅ AI 診斷已準備完成，可直接查看報告")
    else:
        st.caption("⏳ AI 診斷正在背景載入中...")

    # 導航
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("⬅️ 返回修改", use_container_width=True):
            st.session_state.step = 1
            st.rerun()
    with c2:
        if st.button("🤖 查看 AI 診斷報告", type="primary", use_container_width=True):
            st.session_state.step = 3
            st.rerun()


def _render_stock_card(ctx):
    name = f"{ctx['股票名稱']}（{ctx['股票代號']}）"
    pnl = ctx["帳面損益"]
    tr = ctx["含息總報酬"]
    cy = ctx["成本殖利率"]

    st.markdown(f'<div class="sc"><h4>{name}</h4>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("成本", f"${ctx['買進成本']:.2f}")
    with c2:
        st.metric("現價", f"${ctx['收盤價']:.2f}" if ctx["收盤價"] else "—")
    with c3:
        st.metric("帳面損益", f"{pnl:+.2f}%" if pnl is not None else "—")
    with c4:
        st.metric("含息總報酬", f"{tr:+.2f}%" if tr is not None else "—")
    with c5:
        st.metric("成本殖利率", f"{cy:.2f}%" if cy else "—")

    html = mk_rail(ctx["年低"], ctx["年高"], ctx["現價分位"], ctx["成本分位"])
    if html:
        st.markdown(html, unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        dy = ctx["連續配息年數"]
        st.caption(f"🏦 配息 **{int(dy)}** 年" if dy else "🏦 —")
    with sc2:
        st.caption(f"💰 年息 **{fmoney(ctx['年股息現金流'])}**")
    with sc3:
        r = ctx["多空比"]
        rd = "—" if r is None else ("全看多🔥" if r == float("inf") else f"{r:.1f}")
        st.caption(f"📢 多空比 **{rd}**")
    with sc4:
        w = ctx.get("持股權重", 0)
        st.caption(f"⚖️ 佔比 **{w:.0f}%**")
    st.markdown("</div>", unsafe_allow_html=True)




def _news_link(alert: dict) -> str:
    """產生該警示事件的 Google 新聞搜尋連結（限定警示日前後 4 天，真實新聞、非生成內容）"""
    from urllib.parse import quote
    from datetime import timedelta
    kw = {
        "🏦 法人動向": "法人 買賣超",
        "📈 價格動能": "股價 新高",
        "📉 價格動能": "股價 下跌",
        "💬 社群情緒": "股價",
        "📅 除息事件": "除息",
    }.get(alert["類型"], "")
    q = quote(f'{alert.get("名稱", "")} {kw}')
    try:
        d = datetime.strptime(alert["日期"], "%Y/%m/%d")
        lo, hi = d - timedelta(days=4), d + timedelta(days=4)
        tbs = f"cdr:1,cd_min:{lo.month}/{lo.day}/{lo.year},cd_max:{hi.month}/{hi.day}/{hi.year}"
        return f"https://www.google.com/search?q={q}&tbm=nws&tbs={quote(tbs, safe=':,/')}"
    except Exception:
        return f"https://www.google.com/search?q={q}&tbm=nws"


# ══════════ 持股防護罩：警示中心 ══════════
def render_shield_center():
    st.markdown("---")
    st.markdown("### 🛡️ 持股防護罩已開啟")
    if st.session_state.get("shield_alerts") is None:
        with st.spinner("掃描法人動向、價格動能、社群情緒與除息事件..."):
            st.session_state.shield_alerts = get_shield_engine().generate(st.session_state.holdings)
    alerts = st.session_state.shield_alerts

    c1, c2, c3 = st.columns(3)
    c1.metric("2025 全年通知", f"{alerts['總數']} 則")
    c2.metric("平均每月", f"{alerts['總數']/12:.1f} 則")
    c3.metric("即將到來", f"{len(alerts['即將到來'])} 個事件")
    st.caption("　·　".join(f"{k} {v} 則" for k, v in alerts["統計"].items()))

    tab_up, tab_replay = st.tabs(["📅 即將到來", "⏪ 2025 全年回放"])
    with tab_up:
        if alerts["即將到來"]:
            for a in alerts["即將到來"]:
                st.info(f"**{a['日期']}** {a['類型']}　{a['訊息']}")
        else:
            st.markdown("目前沒有已排定的未來事件。防護罩會在法人籌碼、價格動能或社群情緒出現變化時即時通知你。")
    with tab_replay:
        st.caption("用 2025 全年真實資料回放：如果你年初就開啟防護罩，會在這些時刻收到通知——這就是你回來看一眼的理由。")
        def _alert_line(a):
            # 社群情緒是 CMoney 站內行為數據，無對應外部新聞，不掛連結
            link = "" if a["類型"] == "💬 社群情緒" else f"　[🔎 相關新聞]({_news_link(a)})"
            return f"**{a['日期']}**　{a['類型']}　{a['訊息']}{link}"

        for a in alerts["回放"][:15]:
            st.markdown(_alert_line(a))
        if alerts["總數"] > 15:
            with st.expander(f"查看其餘 {alerts['總數']-15} 則"):
                for a in alerts["回放"][15:]:
                    st.markdown(_alert_line(a))


# ══════════ 投資人格卡（社群分享揭曉）══════════
def render_persona_card():
    """步驟 3 的「揭曉」環節：把組合收斂成一張可炫耀/自嘲的 9:16 人格卡。"""
    ctxs = st.session_state.contexts
    if not ctxs:
        return

    feats = persona.compute_features(ctxs)
    result = persona.determine_persona(feats, cached_market_reference())

    # ── 抽卡閘門：先給一個懸念按鈕，點下去才揭曉（強化「揭曉」儀式感）──
    if not st.session_state.persona_revealed:
        st.markdown("""<div class="persona-gate">
        <div class="pg-emoji">🃏</div>
        <div class="pg-title">你是哪種股海奇葩？</div>
        <div class="pg-sub">AI 已讀完你的庫存，一張專屬投資人格卡正在成形——只差你翻牌</div>
        </div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if st.button("🎴 翻牌！抽出我的投資人格卡", type="primary", use_container_width=True):
                st.session_state.persona_revealed = True
                st.balloons()
                st.rerun()
        return

    # ── 隱私 / AI 加碼 開關 ──
    oc1, oc2 = st.columns(2)
    with oc1:
        mask = st.toggle("🛡️ 代號打碼（顯示產業類別）",
                         value=st.session_state.persona_mask_code, key="persona_mask_code")
    with oc2:
        ai_flavor = st.toggle("🤖 AI 加碼一句專屬吐槽",
                              value=st.session_state.persona_ai_flavor, key="persona_ai_flavor")

    # ── AI 客製吐槽（可選；快取避免每次 rerun 重打）──
    if ai_flavor:
        cache_key = f"persona_flavor_{result['primary']}_{result['secondary']}"
        if cache_key not in st.session_state:
            with st.spinner("AI 正在為你量身吐槽..."):
                st.session_state[cache_key] = persona.generate_ai_flavor(
                    result, feats, engine=st.session_state.get("engine", "auto"))
        flavor = st.session_state.get(cache_key, "")
        if flavor:
            result = {**result, "roast": flavor}

    rows = persona.composition_breakdown(ctxs, result)
    svg = persona.build_card_svg(result, feats, rows, mask_amount=True, mask_code=mask)

    # ── 卡片渲染（抽卡翻牌動畫，置中）──
    card_html = f"""
    <style>
      .persona-stage{{display:flex;justify-content:center;padding:8px 0 4px;perspective:1200px}}
      .persona-flip{{animation:pflip .7s cubic-bezier(.2,.7,.2,1) both}}
      @keyframes pflip{{
        0%{{transform:rotateY(90deg) scale(.9);opacity:0}}
        60%{{transform:rotateY(-8deg) scale(1.02)}}
        100%{{transform:rotateY(0) scale(1);opacity:1}}
      }}
      .persona-flip svg{{filter:drop-shadow(0 12px 30px rgba(0,0,0,.45));border-radius:26px}}
      @media (prefers-reduced-motion:reduce){{.persona-flip{{animation:none}}}}
    </style>
    <div class="persona-stage"><div class="persona-flip">{svg}</div></div>
    """
    components.html(card_html, height=690, scrolling=False)

    # ── 成分揭曉的一句話（呼應卡片 mix_text）──
    st.markdown(
        f'<div style="text-align:center;color:#94a3b8;font-size:.9rem;margin:-6px 0 10px">'
        f'成分鑑定：<b style="color:{result["accent"]}">{result["mix_text"]}</b>　·　'
        f'{result["pr_label"]}：{result["pr_prefix"]} <b>{result["pr"]}%</b>'
        f'（{result["pr_ref"]}）</div>',
        unsafe_allow_html=True)

    # ── 下載 + 一鍵複製分享文案 ──
    share_text = persona.build_share_text(result)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "⬇️ 下載人格卡（SVG）",
            data=svg.encode("utf-8"),
            file_name=f"投資人格卡_{result['label']}.svg",
            mime="image/svg+xml",
            use_container_width=True,
        )
    with d2:
        if st.button("🔁 重新翻牌", use_container_width=True):
            for k in [k for k in st.session_state if k.startswith("persona_flavor_")]:
                del st.session_state[k]
            st.session_state.persona_revealed = False
            st.rerun()

    st.caption("📲 分享到限時動態 / Threads / Dcard：長按卡片存圖，或複製下方文案。")
    st.code(share_text, language=None)


# ══════════ 步驟 3 ══════════
def page_ai():
    ctxs = st.session_state.contexts
    alerts = st.session_state.alerts
    if not ctxs:
        st.warning("請先完成健診")
        if st.button("⬅️ 返回"):
            st.session_state.step = 1
            st.rerun()
        return

    # 引擎狀態
    status = cached_engine_status()

    # ★ 引擎選擇（直接在步驟3讓使用者選）
    engine_choices = []
    engine_display = {}
    if status["aws_available"]:
        engine_choices.append("aws")
        engine_display["aws"] = "☁️ AWS Bedrock（雲端，快速高品質）"
    if status["ollama_available"]:
        engine_choices.append("ollama")
        engine_display["ollama"] = "🖥️ Ollama 本地（資料不外洩）"

    if not engine_choices:
        st.error("⚠️ 無可用 AI 引擎")
        return

    # 預設引擎
    current = st.session_state.get("engine_step3", engine_choices[0])
    if current not in engine_choices:
        current = engine_choices[0]

    if len(engine_choices) > 1:
        actual = st.radio(
            "選擇 AI 引擎",
            options=engine_choices,
            format_func=lambda x: engine_display.get(x, x),
            index=engine_choices.index(current),
            horizontal=True,
            key="engine_radio_step3",
        )
        if actual != st.session_state.get("engine_step3"):
            st.session_state["engine_step3"] = actual
            # 切引擎時清除舊回覆
            for k in [k for k in st.session_state if k.startswith("ai_")]:
                del st.session_state[k]
            st.session_state.shield_on = False
            st.session_state.shield_alerts = None
            st.rerun()
    else:
        actual = engine_choices[0]
        st.session_state["engine_step3"] = actual

    if actual == "aws":
        st.caption(f"☁️ AWS Bedrock Claude Haiku 4.5（{status['aws_region']}）")
    else:
        st.caption(f"🖥️ Ollama {status['ollama_model']}（本地推論，資料不離開電腦）")

    st.markdown("### 🌳 AI 陪伴診斷報告")

    # ── 🎭 投資人格卡：社群分享揭曉（放在報告最前，作為「揭曉」獎勵）──
    with st.container():
        st.markdown("#### 🎭 你的投資人格卡")
        render_persona_card()
    st.markdown("---")

    llm = get_llm_service()

    # 逐檔數據摘要（指標＋分位標尺），AI 報告在下方只出「一份」組合診斷
    for ctx in ctxs:
        name = f"{ctx['股票名稱']}（{ctx['股票代號']}）"
        pnl = ctx["帳面損益"]
        tr = ctx["含息總報酬"]

        st.markdown(f"#### 🔮 {name}")
        pstr = f"{pnl:+.2f}%" if pnl is not None else "—"
        tstr = f"{tr:+.2f}%" if tr is not None else "—"
        st.markdown(f"成本 **${ctx['買進成本']:.2f}** → 現價 **${ctx['收盤價']:.2f}** ｜ "
                    f"帳面 **{pstr}** ｜ 含息 **{tstr}**")

        html = mk_rail(ctx["年低"], ctx["年高"], ctx["現價分位"], ctx["成本分位"])
        if html:
            st.markdown(html, unsafe_allow_html=True)
        st.markdown("")

    # ── 產業配置（依市值）──
    total_mv = sum((c.get("收盤價") or 0) * (c.get("持有股數") or 0) for c in ctxs)
    if total_mv > 0:
        ind_mv = {}
        for c in ctxs:
            mv = (c.get("收盤價") or 0) * (c.get("持有股數") or 0)
            ind_mv[c.get("產業", "未知")] = ind_mv.get(c.get("產業", "未知"), 0) + mv
        st.markdown("#### 🧩 產業配置（依目前市值）")
        rows = ""
        for ind, mv in sorted(ind_mv.items(), key=lambda x: -x[1]):
            pct = mv / total_mv * 100
            rows += (f'<div class="irow"><span class="ilabel">{ind}</span>'
                     f'<span class="itrack"><span class="ifill" style="width:{pct:.1f}%"></span></span>'
                     f'<span class="ipct">{pct:.1f}%</span>'
                     f'<span class="imv">{mv:,.0f} 元</span></div>')
        st.markdown(f'<div class="ibars">{rows}</div>', unsafe_allow_html=True)
        st.markdown("")

    # ── 組合層 AI 診斷：整個組合一份報告（優先讀預載結果）──
    st.markdown(f"#### 🌳 組合診斷報告（{len(ctxs)} 檔持股綜合分析）")
    if "ai_portfolio" not in st.session_state:
        with st.spinner("AI 正在綜合分析你的持股組合..."):
            st.session_state["ai_portfolio"] = llm.diagnose_portfolio(ctxs, alerts, engine=actual)

    st.markdown(f'<div class="abox">{st.session_state["ai_portfolio"]}</div>', unsafe_allow_html=True)

    # CTA
    st.markdown("---")
    st.markdown("""<div style="text-align:center;padding:1.3rem;
    background:linear-gradient(135deg,#0f172a,#1e293b);border-radius:12px;color:#fff;margin-bottom:.8rem">
    <h3 style="color:#fbbf24;margin:0 0 .4rem">🛡️ 開啟持股防護罩</h3>
    <p style="margin:0;font-size:.85rem;color:#94a3b8">買點劇變・法人異常・社群翻轉 → 即時通知</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🎈 點我訂閱（開啟防護罩慶祝一下！）", type="primary", use_container_width=True):
            st.session_state.show_coins = True
            st.session_state.shield_on = True
            st.rerun()

    # 氣球金幣特效（可重複觸發）
    if st.session_state.get("show_coins"):
        # 加入隨機種子讓每次 HTML 不同，強制重新渲染
        import random
        seed = random.randint(0, 999999)
        balloon_html = BALLOON_COIN_HTML.replace("balloon-game", f"balloon-game-{seed}").replace("bc-canvas", f"bc-canvas-{seed}")
        st.success("🎉 持股防護罩已啟動！氣球正帶著金幣為你慶祝　⬇️ 你的專屬警示中心已在下方展開")
        components.html(balloon_html, height=1, scrolling=False)

    # ── 持股防護罩：警示中心（真實資料驅動，事件文字在前）──
    if st.session_state.get("shield_on"):
        render_shield_center()

    # 特效控制按鈕固定在頁面最下方
    if st.session_state.get("show_coins"):
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🎈 再來一波氣球！", use_container_width=True):
                st.rerun()
        with col_b:
            if st.button("✖️ 關閉特效", use_container_width=True):
                st.session_state.show_coins = False
                st.rerun()

    st.markdown("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 重新健診", use_container_width=True):
            for k in [k for k in st.session_state if k.startswith("ai_")]:
                del st.session_state[k]
            st.session_state.shield_on = False
            st.session_state.shield_alerts = None
            st.session_state.ai_prefetch_done = False
            st.session_state["_prefetch_started"] = False
            st.session_state.step = 1
            st.rerun()
    with c2:
        if st.button("📊 數據總覽", use_container_width=True):
            st.session_state.step = 2
            st.rerun()


# ══════════ 側邊欄 ══════════
with st.sidebar:
    st.markdown("## 🎨 顯示設定")

    # 字體大小
    font_opts = ["小", "中", "大", "特大"]
    current_fs = st.session_state.get("font_size", "中")
    new_fs = st.select_slider("字體大小", options=font_opts, value=current_fs)
    if new_fs != current_fs:
        st.session_state.font_size = new_fs
        st.rerun()

    # 黑白模式
    dark = st.toggle("🌙 深色模式", value=st.session_state.get("dark_mode", True))
    if dark != st.session_state.dark_mode:
        st.session_state.dark_mode = dark
        st.rerun()

    st.markdown("---")
    st.markdown("## ⚙️ AI 引擎設定")

    status = cached_engine_status()

    # 引擎選擇
    engine_options = ["auto"]
    engine_labels = {"auto": "🔄 自動（優先 AWS）", "aws": "☁️ AWS Bedrock", "ollama": "🖥️ Ollama 本地"}
    if status["aws_available"]:
        engine_options.append("aws")
    if status["ollama_available"]:
        engine_options.append("ollama")

    current_engine = st.session_state.get("engine", "auto")
    selected_engine = st.radio(
        "選擇 AI 引擎",
        options=engine_options,
        format_func=lambda x: engine_labels.get(x, x),
        index=engine_options.index(current_engine) if current_engine in engine_options else 0,
    )
    if selected_engine != st.session_state.engine:
        st.session_state.engine = selected_engine
        # 清除預載結果（引擎切換需重新生成）
        for k in [k for k in st.session_state if k.startswith("ai_")]:
            del st.session_state[k]
        st.session_state.ai_prefetch_done = False
        st.session_state["_prefetch_started"] = False

    st.markdown("---")
    st.markdown("**引擎狀態**")
    if status["aws_available"]:
        st.success(f"☁️ AWS Bedrock 可用")
        st.caption(f"Region: {status['aws_region']}")
        st.caption(f"Model: Claude Haiku 4.5")
    else:
        st.warning("☁️ AWS 未設定（缺少環境變數）")

    if status["ollama_available"]:
        st.success(f"🖥️ Ollama 可用")
        st.caption(f"Model: {status['ollama_model']}")
    else:
        st.warning("🖥️ Ollama 離線")

    st.markdown("---")
    st.caption("📅 資料基準 2025/12/31")
    st.caption("⚠️ 不構成投資建議")
    st.markdown("---")
    if st.button("🔄 完全重置"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        cached_stock_list.clear()
        cached_engine_status.clear()
        st.rerun()

main()
