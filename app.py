from __future__ import annotations

import pandas as pd
import streamlit as st

from modules import (
    AnalysisInputs,
    available_sectors,
    MODEL_VERSION,
    OPTIONAL_UNSTRUCTURED,
    build_cam_docx,
    build_cam_pdf,
    empty_source_result,
    extract_pdf_text_with_fallback,
    fetch_live_research,
    get_demo_scenarios,
    load_annual_report_text,
    load_bank_statement,
    load_gst_summary,
    load_manual_notes,
    load_structured_or_pdf,
    run_analysis,
    train_hybrid_model,
)

st.set_page_config(
    page_title="Intelli-Credit",
    page_icon="IC",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system
# ---------------------------------------------------------------------------
# One token set drives every surface. Semantic colours are keyed to credit
# outcomes (positive / caution / adverse) rather than to raw hues, so a decision
# reads the same way wherever it appears — a chip, a badge or a KPI card.

TONE_POSITIVE = "positive"
TONE_CAUTION = "caution"
TONE_ADVERSE = "adverse"
TONE_NEUTRAL = "neutral"

_CAUTION_WORDS = {
    "review", "moderate", "watch", "possible", "placeholder", "tight",
    "moderate mismatch", "unknown", "offline_sample", "not extracted",
}
_ADVERSE_WORDS = {
    "reject", "high", "high mismatch", "alert", "failed", "elevated", "stress",
    "not found", "missing",
}
_DECISION_TONE = {"lend": TONE_POSITIVE, "review": TONE_CAUTION, "reject": TONE_ADVERSE}


def _tone_for(value) -> str:
    text = str(value).strip().lower()
    if text in _ADVERSE_WORDS:
        return TONE_ADVERSE
    if text in _CAUTION_WORDS:
        return TONE_CAUTION
    return TONE_POSITIVE


def _status_chip(label: str, value: str) -> str:
    """A label/value row colour-coded by what the value means for credit risk."""
    return (
        f"<div class='ic-chip ic-tone-{_tone_for(value)}'>"
        f"<span class='ic-chip-label'>{label}</span>"
        f"<span class='ic-chip-value'>{value}</span></div>"
    )


def _kpi(label: str, value: str, tone: str = TONE_NEUTRAL, sub: str = "") -> str:
    sub_html = f"<div class='ic-kpi-sub'>{sub}</div>" if sub else ""
    return (
        f"<div class='ic-kpi ic-tone-{tone}'>"
        f"<div class='ic-kpi-label'>{label}</div>"
        f"<div class='ic-kpi-value'>{value}</div>{sub_html}</div>"
    )


def _decision_badge(decision: str, confidence=None) -> str:
    tone = _DECISION_TONE.get(str(decision).strip().lower(), TONE_NEUTRAL)
    conf = (
        f"<span class='ic-badge-conf'>{float(confidence):.0%} confidence</span>"
        if confidence is not None
        else ""
    )
    return f"<div class='ic-badge ic-tone-{tone}'><span>{decision}</span>{conf}</div>"


def _section(title: str, caption: str = "") -> None:
    """Every section opens the same way, so the eye learns one pattern."""
    st.markdown(f"<div class='ic-section'><h3>{title}</h3></div>", unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def _table(df, height=None):
    """Dataframes render identically everywhere: no index, full width."""
    kwargs = {"use_container_width": True, "hide_index": True}
    if height:
        kwargs["height"] = height
    st.dataframe(df, **kwargs)


def _fmt_amount(amount) -> str:
    """Render a figure, or say it was not found. Never show a false zero."""
    if amount is None:
        return "Not extracted"
    return f"INR {amount:,.0f}"


def _fmt_inr_short(amount) -> str:
    """Compact Indian notation for headline figures.

    A full rupee figure overflows a metric card and gets truncated mid-number,
    which is the one thing a credit figure must never do. Crore and lakh are also
    how the number is actually spoken in an Indian credit committee.
    """
    if amount is None:
        return "Not extracted"
    sign = "-" if amount < 0 else ""
    value = abs(float(amount))
    if value >= 10_000_000:
        return f"{sign}INR {value / 10_000_000:,.2f} Cr"
    if value >= 100_000:
        return f"{sign}INR {value / 100_000:,.2f} L"
    return f"{sign}INR {value:,.0f}"


def _read_text_upload(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    try:
        return uploaded_file.getvalue().decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""


@st.cache_data(ttl=900, show_spinner=False)
def _cached_research(company_name: str, sector: str, enabled: bool) -> dict:
    """Fetch secondary research, cached for 15 minutes per borrower."""
    return fetch_live_research(company_name, sector, enabled=enabled)


@st.cache_resource(show_spinner=False)
def _get_hybrid_model(_version: str):
    return train_hybrid_model()


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --ic-navy:        #0E2A4D;
        --ic-blue:        #1B4F8C;
        --ic-blue-soft:   #2E6BB0;
        --ic-bg:          #F6F8FC;
        --ic-surface:     #FFFFFF;
        --ic-surface-2:   #F1F5FB;
        --ic-border:      #DCE5F2;
        --ic-border-firm: #C2D3E8;
        --ic-text:        #111C2E;
        --ic-muted:       #566B88;
        --ic-faint:       #8496AE;

        --ic-pos:     #0F766E;
        --ic-pos-bg:  #E8F5F2;
        --ic-warn:    #B45309;
        --ic-warn-bg: #FDF3E5;
        --ic-neg:     #B4232A;
        --ic-neg-bg:  #FCECEC;

        --r-sm: 8px;
        --r-md: 12px;
        --r-lg: 16px;
        --sh-1: 0 1px 2px rgba(16,40,72,.06), 0 1px 3px rgba(16,40,72,.04);
        --sh-2: 0 4px 16px rgba(16,40,72,.08);
        --sh-3: 0 10px 30px rgba(14,42,77,.18);
    }

    html, body, button, input, textarea, select,
    [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
    [data-testid="stMarkdownContainer"] {
        font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
    }
    /* Streamlit draws icons as ligature text in a Material font. The font rule
       above must never reach them, or "keyboard_arrow_right" renders literally
       on top of the label it sits beside. */
    [data-testid="stIconMaterial"], .material-icons, .material-icons-outlined,
    span[class*="material"] {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
    }
    .stApp { background: var(--ic-bg); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMainBlockContainer"] { padding-top: 2.2rem; max-width: 1500px; }

    /* ---------- Tone helpers ---------- */
    .ic-tone-positive { --tone: var(--ic-pos); --tone-bg: var(--ic-pos-bg); }
    .ic-tone-caution  { --tone: var(--ic-warn); --tone-bg: var(--ic-warn-bg); }
    .ic-tone-adverse  { --tone: var(--ic-neg); --tone-bg: var(--ic-neg-bg); }
    .ic-tone-neutral  { --tone: var(--ic-blue); --tone-bg: var(--ic-surface-2); }

    /* ---------- Masthead ---------- */
    .ic-masthead {
        background: linear-gradient(118deg, var(--ic-navy) 0%, var(--ic-blue) 58%, var(--ic-blue-soft) 100%);
        border-radius: var(--r-lg);
        padding: 22px 28px;
        color: #fff;
        box-shadow: var(--sh-3);
        margin-bottom: 18px;
        display: flex; align-items: center; justify-content: space-between; gap: 24px;
        flex-wrap: wrap;
    }
    .ic-masthead h1 {
        margin: 0; font-size: 1.55rem; font-weight: 700; letter-spacing: -.02em; color: #fff;
    }
    .ic-masthead p {
        margin: 6px 0 0; font-size: .88rem; color: #C9DCF2; font-weight: 400;
    }
    .ic-mast-tag {
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.22);
        border-radius: 999px; padding: 6px 14px;
        font-size: .72rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase;
        white-space: nowrap;
    }

    /* ---------- Borrower header ---------- */
    .ic-borrower-name {
        font-size: 1.28rem; font-weight: 700; color: var(--ic-text);
        letter-spacing: -.02em; line-height: 1.2;
    }
    .ic-borrower-meta { font-size: .82rem; color: var(--ic-muted); margin-top: 3px; }
    .ic-badge-wrap { display: flex; justify-content: flex-end; }
    .ic-rule {
        height: 1px; background: var(--ic-border);
        margin: 18px 0 14px;
    }

    /* ---------- KPI cards ---------- */
    .ic-kpi-row { display: grid; gap: 12px; margin-bottom: 6px; }
    .ic-kpi {
        background: var(--ic-surface);
        border: 1px solid var(--ic-border);
        border-left: 4px solid var(--tone, var(--ic-blue));
        border-radius: var(--r-md);
        padding: 14px 16px;
        box-shadow: var(--sh-1);
        height: 100%;
    }
    .ic-kpi-label {
        font-size: .68rem; font-weight: 600; letter-spacing: .08em;
        text-transform: uppercase; color: var(--ic-faint); margin-bottom: 6px;
    }
    .ic-kpi-value {
        font-size: 1.5rem; font-weight: 700; color: var(--ic-text);
        line-height: 1.15; letter-spacing: -.02em;
    }
    .ic-kpi-sub { font-size: .74rem; color: var(--ic-muted); margin-top: 4px; }

    /* ---------- Decision badge ---------- */
    .ic-badge {
        display: inline-flex; align-items: center; gap: 10px;
        background: var(--tone-bg); color: var(--tone);
        border: 1px solid var(--tone); border-radius: 999px;
        padding: 8px 18px; font-weight: 700; font-size: 1rem;
        letter-spacing: .01em;
    }
    .ic-badge-conf {
        font-weight: 500; font-size: .76rem; opacity: .8;
        border-left: 1px solid currentColor; padding-left: 10px;
    }

    /* ---------- Chips ---------- */
    .ic-chip {
        display: flex; justify-content: space-between; align-items: center; gap: 12px;
        background: var(--ic-surface); border: 1px solid var(--ic-border);
        border-left: 3px solid var(--tone);
        border-radius: var(--r-sm); padding: 10px 12px; margin-bottom: 8px;
        box-shadow: var(--sh-1);
    }
    .ic-chip-label { font-size: .8rem; color: var(--ic-muted); }
    .ic-chip-value { font-size: .84rem; font-weight: 700; color: var(--tone); text-align: right; }

    /* ---------- Section headings ---------- */
    .ic-section { margin: 6px 0 2px; }
    .ic-section h3 {
        margin: 0; font-size: 1.06rem; font-weight: 650; color: var(--ic-text);
        letter-spacing: -.01em; padding-bottom: 8px;
        border-bottom: 2px solid var(--ic-border);
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: var(--ic-surface); padding: 6px;
        border-radius: var(--r-md); border: 1px solid var(--ic-border);
        box-shadow: var(--sh-1); flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        height: auto; padding: 9px 16px; border-radius: var(--r-sm);
        font-size: .85rem; font-weight: 550; color: var(--ic-muted);
        background: transparent; border: none;
    }
    .stTabs [data-baseweb="tab"]:hover { background: var(--ic-surface-2); color: var(--ic-blue); }
    .stTabs [aria-selected="true"] {
        background: var(--ic-blue) !important; color: #fff !important; font-weight: 600;
        box-shadow: 0 2px 8px rgba(27,79,140,.28);
    }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
    /* Nested tabs are a second level, so they read as quieter than the primary
       row but still leave no doubt which one is open. */
    .stTabs .stTabs [data-baseweb="tab-list"] {
        background: transparent; box-shadow: none; border: none;
        padding: 0; gap: 22px; margin-bottom: 4px;
        border-bottom: 1px solid var(--ic-border); border-radius: 0;
    }
    .stTabs .stTabs [data-baseweb="tab"] {
        padding: 8px 2px; border-radius: 0; font-size: .84rem;
        border-bottom: 2px solid transparent; margin-bottom: -1px;
    }
    .stTabs .stTabs [data-baseweb="tab"]:hover {
        background: transparent; color: var(--ic-blue);
        border-bottom-color: var(--ic-border-firm);
    }
    .stTabs .stTabs [aria-selected="true"] {
        background: transparent !important; color: var(--ic-blue) !important;
        border-bottom: 2px solid var(--ic-blue); box-shadow: none; font-weight: 650;
    }

    /* ---------- Buttons: one height, one radius, always fill their column ---------- */
    .stButton > button, .stDownloadButton > button {
        width: 100%;
        min-height: 42px;
        border-radius: var(--r-sm);
        font-weight: 600; font-size: .86rem;
        border: 1px solid var(--ic-border-firm);
        background: var(--ic-surface); color: var(--ic-blue);
        transition: all .15s ease;
        box-shadow: var(--sh-1);
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        border-color: var(--ic-blue); background: var(--ic-surface-2);
        color: var(--ic-blue); transform: translateY(-1px);
        box-shadow: var(--sh-2);
    }
    .stButton > button[kind="primary"] {
        background: var(--ic-blue); color: #fff; border-color: var(--ic-blue);
    }
    .stButton > button[kind="primary"]:hover {
        background: var(--ic-navy); border-color: var(--ic-navy); color: #fff;
    }
    .stDownloadButton > button { color: var(--ic-pos); border-color: #BFE0D9; }
    .stDownloadButton > button:hover { border-color: var(--ic-pos); background: var(--ic-pos-bg); }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: var(--ic-surface);
        border-right: 1px solid var(--ic-border);
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        border: 1px solid var(--ic-border); border-radius: var(--r-sm);
        margin-bottom: 8px; background: var(--ic-surface);
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary { font-weight: 600; font-size: .84rem; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        padding: 10px; background: var(--ic-surface-2);
        border: 1px dashed var(--ic-border-firm); border-radius: var(--r-sm);
        min-height: unset;
    }
    [data-testid="stSidebar"] small { font-size: .68rem; }
    .ic-side-title {
        font-size: .68rem; font-weight: 700; letter-spacing: .1em;
        text-transform: uppercase; color: var(--ic-faint);
        margin: 14px 0 6px;
    }

    /* ---------- Surfaces ---------- */
    [data-testid="stExpander"] {
        border: 1px solid var(--ic-border); border-radius: var(--r-md);
        background: var(--ic-surface); box-shadow: var(--sh-1);
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--ic-border); border-radius: var(--r-md); overflow: hidden;
    }
    [data-testid="stMetric"] {
        background: var(--ic-surface); border: 1px solid var(--ic-border);
        border-radius: var(--r-md); padding: 12px 16px; box-shadow: var(--sh-1);
    }
    [data-testid="stMetricLabel"] p {
        font-size: .68rem !important; font-weight: 600; letter-spacing: .07em;
        text-transform: uppercase; color: var(--ic-faint);
    }
    [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; color: var(--ic-text); }
    .stTextArea textarea {
        border-radius: var(--r-sm); border-color: var(--ic-border);
        font-size: .84rem; background: var(--ic-surface);
    }
    [data-testid="stAlert"] { border-radius: var(--r-md); border: 1px solid var(--ic-border); }

    /* ---------- Pipeline flow ---------- */
    .ic-flow { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
    .ic-flow-step {
        background: var(--ic-surface); border: 1px solid var(--ic-border);
        border-top: 3px solid var(--ic-blue); border-radius: var(--r-md);
        padding: 14px 12px; text-align: center; box-shadow: var(--sh-1);
        position: relative;
    }
    .ic-flow-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 24px; height: 24px; border-radius: 50%;
        background: var(--ic-blue); color: #fff;
        font-size: .72rem; font-weight: 700; margin-bottom: 8px;
    }
    .ic-flow-title { font-size: .82rem; font-weight: 650; color: var(--ic-text); line-height: 1.3; }
    .ic-flow-desc { font-size: .7rem; color: var(--ic-muted); margin-top: 4px; }

    footer, #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='ic-masthead'>
        <div>
            <h1>Intelli-Credit</h1>
            <p>AI-powered corporate credit appraisal for Indian lending workflows</p>
        </div>
        <div class='ic-mast-tag'>Five Cs &nbsp;·&nbsp; Hybrid ML &nbsp;·&nbsp; Explainable</div>
    </div>
    """,
    unsafe_allow_html=True,
)


scenarios = get_demo_scenarios()

with st.sidebar:
    # Nineteen inputs in one flat column was unreadable. They are grouped by the
    # kind of evidence they carry, and every group past the first is collapsed by
    # default so the panel opens short. Demo Mode is the primary control, so it
    # stays outside any expander.
    st.markdown("<div class='ic-side-title'>Borrower</div>", unsafe_allow_html=True)
    demo_mode = st.toggle("Demo Mode", value=True, help="Preloads a curated borrower case.")

    selected_scenario = None
    if demo_mode:
        selected_scenario = st.selectbox("Scenario", list(scenarios.keys()), index=0)
        st.caption("Uploads are disabled while Demo Mode is on.")

    company_name = st.text_input(
        "Company Name", value="Apex Industrial Components Pvt Ltd", disabled=demo_mode
    )
    sector = st.selectbox("Sector", available_sectors(), index=0, disabled=demo_mode)

    with st.expander("Structured data", expanded=not demo_mode):
        bank_file = st.file_uploader("Bank statement (CSV)", type=["csv"], key="bank", disabled=demo_mode)
        gst_file = st.file_uploader(
            "GST summary (CSV/XLSX)", type=["csv", "xlsx", "xls"], key="gst", disabled=demo_mode
        )
        itr_file = st.file_uploader(
            "ITR (CSV/XLSX/PDF)", type=["csv", "xlsx", "xls", "pdf"], key="itr", disabled=demo_mode
        )
        shareholding_file = st.file_uploader(
            "Shareholding pattern (CSV/XLSX/PDF)",
            type=["csv", "xlsx", "xls", "pdf"],
            key="share",
            disabled=demo_mode,
        )

    with st.expander("Documents", expanded=False):
        annual_report_file = st.file_uploader(
            "Annual report (PDF)", type=["pdf"], key="annual", disabled=demo_mode
        )
        legal_notice_file = st.file_uploader(
            "Legal notice (PDF)", type=["pdf"], key="legal", disabled=demo_mode
        )
        sanction_letter_file = st.file_uploader(
            "Sanction letter (PDF)", type=["pdf"], key="sanction", disabled=demo_mode
        )
        rating_report_file = st.file_uploader(
            "Rating report (PDF)", type=["pdf"], key="rating", disabled=demo_mode
        )
        board_minutes_file = st.file_uploader(
            "Board minutes (PDF)", type=["pdf"], key="board", disabled=demo_mode
        )

    with st.expander("Analyst notes", expanded=False):
        primary_notes_input = st.text_area(
            "Primary due diligence",
            placeholder="Site visit, management quality, collateral condition...",
            height=110,
            disabled=demo_mode,
        )
        external_notes_input = st.text_area(
            "External intelligence",
            placeholder="Market, regulatory or promoter intelligence...",
            height=100,
            disabled=demo_mode,
        )
        external_notes_file = st.file_uploader(
            "Notes file (.txt)", type=["txt"], key="ext", disabled=demo_mode
        )

    with st.expander("Research agent inputs", expanded=False):
        st.caption("Optional observations folded into the research panel.")
        promoter_obs_input = st.text_area("Promoter / management", height=70)
        sector_obs_input = st.text_area("Sector", height=70)
        regulatory_obs_input = st.text_area("Regulatory", height=70)
        mca_notes_input = st.text_area("MCA", height=70)
        litigation_notes_input = st.text_area("Litigation", height=70)

    st.markdown("<div class='ic-side-title'>Secondary Research</div>", unsafe_allow_html=True)
    live_research_mode = st.toggle(
        "Search news & filings",
        value=True,
        help=(
            "Queries public news feeds for promoter, litigation, insolvency and "
            "regulatory signals about this borrower. Falls back to clearly "
            "labelled sample headlines when offline."
        ),
    )

    st.markdown("<div class='ic-rule'></div>", unsafe_allow_html=True)
    run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)


def _gather_inputs() -> AnalysisInputs:
    """Read the sidebar widgets, parsing any uploads, into pipeline inputs.

    Only called when the user clicks Run, so PDF parsing and the live-research
    fetch happen once per analysis rather than once per Streamlit rerun.
    """
    research_agent_inputs = {
        "promoter_obs": promoter_obs_input,
        "sector_obs": sector_obs_input,
        "regulatory_obs": regulatory_obs_input,
        "mca_notes": mca_notes_input,
        "litigation_notes": litigation_notes_input,
    }

    if demo_mode:
        scenario = scenarios[selected_scenario]
        report_text = scenario["report_text"]
        live_research = _cached_research(
            scenario["company_name"], scenario["sector"], live_research_mode
        )
        return AnalysisInputs(
            company_name=scenario["company_name"],
            sector=scenario["sector"],
            bank_df=scenario["bank_df"].copy(),
            gst_df=scenario["gst_df"].copy(),
            primary_notes=scenario["manual_notes"],
            external_notes=scenario.get("external_notes", ""),
            annual_result={
                "text": report_text,
                "uploaded": False,
                "parsed": True,
                "method": "demo_text",
                "message": "Demo scenario narrative",
                "chars": len(report_text),
            },
            additional_unstructured={
                src: empty_source_result() for src in OPTIONAL_UNSTRUCTURED
            },
            research_agent_inputs=research_agent_inputs,
            live_research=live_research,
            completeness_note="High confidence: full scenario dataset available (demo mode).",
            data_source_note=f"Demo Mode: {selected_scenario}",
        )

    external_file_text = _read_text_upload(external_notes_file)
    external_notes = (external_notes_input or "").strip()
    if external_file_text:
        external_notes = f"{external_notes}\n{external_file_text}".strip()

    tracked = [
        annual_report_file,
        bank_file,
        gst_file,
        itr_file,
        shareholding_file,
        legal_notice_file,
        sanction_letter_file,
        rating_report_file,
        board_minutes_file,
    ]
    available_count = sum(1 for item in tracked if item is not None)
    available_count += 1 if primary_notes_input.strip() else 0
    available_count += 1 if external_notes else 0
    completeness_score = int((available_count / 11) * 100)

    live_research = _cached_research(company_name, sector, live_research_mode)

    return AnalysisInputs(
        company_name=company_name,
        sector=sector,
        bank_df=load_bank_statement(bank_file),
        gst_df=load_gst_summary(gst_file),
        primary_notes=load_manual_notes(primary_notes_input),
        external_notes=external_notes,
        annual_result=extract_pdf_text_with_fallback(
            annual_report_file, fallback_text=load_annual_report_text(None)
        ),
        additional_unstructured={
            "Legal Notice": extract_pdf_text_with_fallback(legal_notice_file, fallback_text=""),
            "Sanction Letter": extract_pdf_text_with_fallback(sanction_letter_file, fallback_text=""),
            "Rating Report": extract_pdf_text_with_fallback(rating_report_file, fallback_text=""),
            "Board Minutes": extract_pdf_text_with_fallback(board_minutes_file, fallback_text=""),
        },
        itr_result=load_structured_or_pdf(itr_file),
        share_result=load_structured_or_pdf(shareholding_file),
        research_agent_inputs=research_agent_inputs,
        live_research=live_research,
        completeness_note=f"Data completeness: {completeness_score}% based on provided multi-source inputs.",
        data_source_note="Uploaded files + fallback defaults for missing sources",
        bank_uploaded=bank_file is not None,
        gst_uploaded=gst_file is not None,
    )


# --- Run gate --------------------------------------------------------------
# The analysis runs only on an explicit click and is held in session state, so
# typing in a text area or switching tabs re-renders without recomputing.
if run_clicked:
    with st.spinner("Running Intelli-Credit analysis..."):
        st.session_state.analysis = run_analysis(
            _gather_inputs(),
            model_bundle=_get_hybrid_model(MODEL_VERSION),
        )
        st.session_state.analysis_context = {
            "demo_mode": demo_mode,
            "selected_scenario": selected_scenario,
        }

analysis = st.session_state.get("analysis")
if analysis is None:
    st.info(
        "Set your inputs in the sidebar, then click **Run Intelli-Credit Analysis**. "
        "Demo Mode preloads curated borrower cases."
    )
    st.stop()

# What the *analysed* run used — not the current widget state, which the user
# may have changed since.
_context = st.session_state.get("analysis_context", {})
analysed_demo_mode = _context.get("demo_mode", False)
analysed_scenario = _context.get("selected_scenario")

company_name = analysis["company_name"]
sector = analysis["sector"]
bank_df = analysis["bank_df"]
gst_df = analysis["gst_df"]
primary_notes = analysis["primary_notes"]
external_notes = analysis["external_notes"]
completeness_note = analysis["completeness_note"]
data_source_note = analysis["data_source_note"]
initial_doc_signals = analysis["doc_signals"]
recon = analysis["recon"]
insights = analysis["insights"]
rag_index = analysis["rag_index"]
rag_theme_results = analysis["rag_theme_results"]
rag_adjustments = analysis["rag_adjustments"]
qual_adjustments = analysis["qual_adjustments"]
news_adjustments = analysis["news_adjustments"]
research_agent_findings = analysis["research_agent_findings"]
all_adjustments = analysis["all_adjustments"]
score_bundle = analysis["score_bundle"]
recommendation = analysis["recommendation"]
source_rows = analysis["source_rows"]
source_findings = analysis["source_findings"]
unstructured_processing = analysis["unstructured_processing"]
indian_context_cards = analysis["indian_context_cards"]
ml_features = analysis["ml_features"]
ml_view = analysis["ml_view"]
hybrid_view = analysis["hybrid_view"]
live_research = analysis["live_research"]
cam_text = analysis["cam_text"]
docx_bytes = build_cam_docx(cam_text)
pdf_bytes = build_cam_pdf(cam_text)

# --- Borrower header --------------------------------------------------------
_head_left, _head_right = st.columns([3, 2], vertical_alignment="center")
with _head_left:
    st.markdown(
        f"<div class='ic-borrower'><div class='ic-borrower-name'>{company_name}</div>"
        f"<div class='ic-borrower-meta'>{sector} &nbsp;·&nbsp; {data_source_note}</div></div>",
        unsafe_allow_html=True,
    )
with _head_right:
    st.markdown(
        f"<div class='ic-badge-wrap'>"
        f"{_decision_badge(hybrid_view['final_decision'], hybrid_view['decision_confidence'])}"
        f"</div>",
        unsafe_allow_html=True,
    )

# --- Headline figures -------------------------------------------------------
# The score and the reconciliation status carry risk tone; the limit and pricing
# are facts that follow from the decision, so they stay neutral.
_k = st.columns(5)
_kpis = [
    ("Risk Score", f"{score_bundle['total_score']}/100", _tone_for(score_bundle["risk_category"]),
     f"{score_bundle['risk_category']} risk"),
    ("Reconciliation", recon["overall_status"], _tone_for(recon["overall_status"]),
     f"GST vs bank variance {recon['variance_pct']:.1f}%"),
    ("Recommended Limit", _fmt_inr_short(hybrid_view["hybrid_limit"]), TONE_NEUTRAL,
     "Capped by bank throughput"),
    ("Pricing Band", hybrid_view["hybrid_pricing_band"], TONE_NEUTRAL, "Risk-adjusted"),
    ("Signals Applied", str(len(all_adjustments)), TONE_NEUTRAL,
     f"{len(news_adjustments)} from news, {len(rag_adjustments)} from documents"),
]
for _col, (_label, _value, _tone, _sub) in zip(_k, _kpis):
    _col.markdown(_kpi(_label, _value, _tone, _sub), unsafe_allow_html=True)

if score_bundle.get("data_gaps"):
    st.warning(
        f"Scored with {len(score_bundle['data_gaps'])} missing input(s). "
        "See Documents → Document Insights for detail."
    )

st.markdown("<div class='ic-rule'></div>", unsafe_allow_html=True)

# --- Navigation -------------------------------------------------------------
# Fourteen sibling tabs overflowed the bar and mixed two naming schemes. The same
# fourteen screens are now grouped into six stages that follow the appraisal
# itself, so the top row stays scannable and the order tells the story. Tab
# containers are position-independent, so the section bodies further down keep
# their original `with tabN:` blocks untouched.
_g_inputs, _g_docs, _g_verify, _g_research, _g_scoring, _g_decision = st.tabs(
    [
        "1 · Inputs",
        "2 · Documents",
        "3 · Verification",
        "4 · Research",
        "5 · Scoring",
        "6 · Decision & CAM",
    ]
)

with _g_inputs:
    tab1, tab14 = st.tabs(["Source Coverage", "How It Works"])

with _g_docs:
    tab2, tab7 = st.tabs(["Document Insights", "Retrieved Evidence"])

with _g_verify:
    tab3, tab9 = st.tabs(["GST-Bank Reconciliation", "Indian Context Signals"])

with _g_research:
    tab4, tab8, tab13 = st.tabs(["Research & Qualitative", "Research Agent", "Secondary Research"])

with _g_scoring:
    tab5, tab11 = st.tabs(["Scoring & Explainability", "ML Diagnostics"])

with _g_decision:
    tab10, tab6, tab12 = st.tabs(["Hybrid Recommendation", "Final Recommendation", "Export CAM"])

with tab1:
    _section("Input / Upload / Demo Selection")
    s1, s2 = st.columns(2)
    with s1:
        st.markdown("**Current Input Mode**")
        st.write("Demo Mode" if analysed_demo_mode else "Manual Upload Mode")
        if analysed_demo_mode:
            st.success(f"Scenario loaded: {analysed_scenario}")
        st.caption(data_source_note)
    with s2:
        st.markdown("**Data Completeness & Confidence**")
        st.info(completeness_note)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Bank Statement Snapshot**")
        st.dataframe(bank_df.head(12), use_container_width=True)
    with c2:
        st.markdown("**GST Summary Snapshot**")
        st.dataframe(gst_df.head(12), use_container_width=True)

    st.markdown("**Source Coverage Summary**")
    _table(pd.DataFrame(source_rows))

with tab2:
    _section("Extracted Financial and Multi-Document Insights")
    d1, d2, d3, d4 = st.columns(4)
    d1.markdown(_kpi("Revenue", _fmt_inr_short(initial_doc_signals["revenue"])), unsafe_allow_html=True)
    d2.markdown(_kpi("Total Debt", _fmt_inr_short(initial_doc_signals["debt"])), unsafe_allow_html=True)
    d3.markdown(_kpi("Liabilities", _fmt_inr_short(initial_doc_signals["liabilities"])), unsafe_allow_html=True)
    d4.markdown(
        _kpi(
            "Operating Cash Flow",
            _fmt_inr_short(initial_doc_signals["operating_cash_flow"]),
            _tone_for(initial_doc_signals["cash_flow_indicator"]),
            initial_doc_signals["cash_flow_indicator"],
        ),
        unsafe_allow_html=True,
    )

    st.write(f"Cash Flow Indicator: **{initial_doc_signals['cash_flow_indicator']}**")

    st.markdown("**Extraction Confidence**")
    st.caption(
        "Every figure above is traced to the sentence it came from. Nothing is "
        "assumed — a figure that could not be found is reported as missing, and "
        "the scoring engine skips the rules that depend on it."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Field": label,
                    "Value": _fmt_amount(initial_doc_signals.get(field)),
                    "Status": "Extracted"
                    if initial_doc_signals.get("extraction", {}).get(field, {}).get("confidence")
                    == "extracted"
                    else "Not found",
                    "Candidates Considered": initial_doc_signals.get("extraction", {})
                    .get(field, {})
                    .get("candidates", 0),
                    "Evidence": initial_doc_signals.get("extraction", {})
                    .get(field, {})
                    .get("evidence", "")
                    or "-",
                }
                for field, label in [
                    ("revenue", "Revenue"),
                    ("debt", "Total debt"),
                    ("liabilities", "Total liabilities"),
                    ("operating_cash_flow", "Operating cash flow"),
                ]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    if score_bundle.get("data_gaps"):
        st.warning(
            "**Scored with incomplete data.** The following inputs could not be "
            "extracted, so the rules depending on them were skipped rather than "
            "applied to an assumed value:\n\n"
            + "\n".join(f"- {gap}" for gap in score_bundle["data_gaps"])
        )

    x1, x2 = st.columns(2)
    with x1:
        st.markdown("**Litigation / Legal Notice Indicator**")
        st.markdown(_status_chip("Litigation Signal", str(initial_doc_signals["litigation_mentions"])), unsafe_allow_html=True)
    with x2:
        st.markdown("**Promoter / Governance Risk Indicator**")
        st.markdown(_status_chip("Governance Signal", str(initial_doc_signals["governance_mentions"])), unsafe_allow_html=True)

    st.markdown("**Document Processing Details (Native vs OCR Fallback)**")
    _table(pd.DataFrame(unstructured_processing))

    st.markdown("**Source-Wise Findings**")
    for title, line in source_findings.items():
        st.markdown(f"- **{title}**: {line}")

with tab3:
    _section("GST vs Bank Reconciliation and Detected Flags")
    _table(recon["table"])
    r1, r2, r3 = st.columns(3)
    r1.metric("GST Turnover", f"INR {recon['gst_turnover']:,.0f}")
    r2.metric("Bank Inflows", f"INR {recon['bank_inflows']:,.0f}")
    r3.metric("Variance", f"{recon['variance_pct']:.2f}%")
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown(_status_chip("Mismatch", recon["mismatch_flag"]), unsafe_allow_html=True)
    with f2:
        st.markdown(_status_chip("Circular Trading", recon["circular_flag"]), unsafe_allow_html=True)
    with f3:
        st.markdown(_status_chip("Revenue Inflation", recon["inflated_revenue_flag"]), unsafe_allow_html=True)

    chart_df = bank_df.copy()
    if "date" in chart_df.columns:
        chart_df["month"] = chart_df["date"].dt.to_period("M").astype(str)
        month_flow = chart_df.groupby("month", as_index=True)[["inflow", "outflow"]].sum()
        st.markdown("**Monthly Bank Flow Trend**")
        st.line_chart(month_flow)

    gst_chart = gst_df[["month", "taxable_turnover"]].set_index("month")
    st.markdown("**GST Turnover Trend**")
    st.bar_chart(gst_chart)

with tab4:
    _section("Research Insights and Qualitative Findings")
    st.info(insights["ai_summary"])
    i1, i2 = st.columns(2)
    with i1:
        st.markdown(_status_chip("Regulatory Risk", insights["regulatory_risk"]), unsafe_allow_html=True)
        st.caption(insights["sector_headwind"])
    with i2:
        st.markdown(_status_chip("Promoter News Level", insights["promoter_news_level"]), unsafe_allow_html=True)
        st.caption(insights["promoter_news"])
    st.markdown("**Primary Due Diligence Notes**")
    st.caption(primary_notes if primary_notes else "Not provided")
    st.markdown("**External Intelligence Notes**")
    st.caption(external_notes if external_notes else "Not provided")

with tab5:
    _section("Risk Scoring and Explainability Dashboard")
    score_df = pd.DataFrame([{"Five C": k, "Score (out of 20)": v} for k, v in score_bundle["five_cs_scores"].items()])
    _table(score_df)
    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Top Positive Drivers**")
        for item in score_bundle.get("positive_drivers", []):
            st.write(f"- {item}")
    with p2:
        st.markdown("**Top Negative Drivers**")
        for item in score_bundle.get("negative_drivers", []):
            st.write(f"- {item}")

    st.markdown("**Penalties and Boosts Applied**")
    adj_df = pd.DataFrame(score_bundle.get("adjustments", []))
    if not adj_df.empty:
        _table(adj_df)

    st.markdown("**RAG-Derived Evidence Signals**")
    rag_df = pd.DataFrame(rag_adjustments)
    if not rag_df.empty:
        st.dataframe(rag_df[["signal", "factor", "points", "reason", "evidence_source", "evidence_chunk"]], use_container_width=True)
    st.markdown("**Qualitative Score Adjustments**")
    qual_df = pd.DataFrame(qual_adjustments)
    if not qual_df.empty:
        _table(qual_df)
    # Built from the single adjustment ledger rather than stitched from each
    # producer, so any new signal source appears here the moment it is wired in.
    _ORIGIN_LABELS = {
        "unstructured_rag": "Uploaded documents (retrieval)",
        "secondary_research": "Secondary research (news)",
        "external_intelligence": "External intelligence notes",
    }
    unified_rows = [
        {
            "source": row.get("source"),
            "origin": _ORIGIN_LABELS.get(
                row.get("source_type"),
                "Primary due diligence"
                if row.get("source") == "Primary Due Diligence"
                else "Analyst notes",
            ),
            "cue": row.get("signal"),
            "five_c": row.get("factor"),
            "score_effect": row.get("points"),
            "reason": row.get("reason"),
        }
        for row in all_adjustments
    ]
    st.markdown("**Unified Explainability Signals**")
    st.caption(
        "Every adjustment that moved the score, whatever produced it — analyst "
        "notes, document retrieval, or secondary research — with the reason it fired."
    )
    if unified_rows:
        st.dataframe(pd.DataFrame(unified_rows), use_container_width=True, hide_index=True)
        total_effect = sum(float(r["score_effect"] or 0) for r in unified_rows)
        st.caption(f"Net effect of all signals on the 100-point score: {total_effect:+.1f}")
    else:
        st.info("No signal-derived adjustments fired for this borrower.")
    st.markdown("**Decision, Limit, and Pricing Explainability**")
    st.success(recommendation["explanation"])
    st.write(f"- Decision logic: {recommendation['decision_rationale']}")
    st.write(f"- Limit logic: {recommendation['limit_rationale']}")
    st.write(f"- Pricing logic: {recommendation['pricing_rationale']}")
    st.markdown("**Indian-Context Indicators Used**")
    _table(pd.DataFrame(indian_context_cards))

with tab6:
    _section("Final CAM Preview and Recommendation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Scorecard Decision", recommendation["decision"])
    c2.metric("ML Decision", ml_view["ml_decision"])
    c3.metric("Final Hybrid Decision", hybrid_view["final_decision"])
    st.text_area("Credit Appraisal Memo", value=cam_text, height=520)
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download CAM (.txt)",
            cam_text,
            f"Intelli_Credit_CAM_{company_name.replace(' ', '_')}.txt",
            "text/plain",
            key="cam_txt_main",
            use_container_width=True,
        )
    with d2:
        if docx_bytes:
            st.download_button(
                "Download CAM (.docx)",
                docx_bytes,
                f"Intelli_Credit_CAM_{company_name.replace(' ', '_')}.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="cam_docx_main",
                use_container_width=True,
            )

with tab7:
    _section("Unstructured Intelligence (Local RAG)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Documents Indexed", rag_index.get("document_count", 0))
    c2.metric("Chunk Count", rag_index.get("chunk_count", 0))
    c3.metric("Retrieval Status", rag_index.get("status", "empty"))
    for theme, result in rag_theme_results.items():
        with st.container(border=True):
            st.markdown(f"**{theme}**")
            st.caption(result.get("summary", ""))
            st.write(f"Contributes to risk analysis: {'Yes' if result.get('contributes_to_risk') else 'No'}")
            for snip in result.get("snippets", []):
                st.markdown(f"- `{snip.get('source')} | {snip.get('chunk_id')} | score={snip.get('score')}`: {snip.get('snippet')}")

with tab8:
    _section("Research Agent Panel")
    if st.button("Generate Research Summary", key="gen_research_summary", use_container_width=True):
        st.session_state["research_summary_text"] = research_agent_findings.get("summary", "")
    if "research_summary_text" not in st.session_state:
        st.session_state["research_summary_text"] = research_agent_findings.get("summary", "")
    st.info(st.session_state.get("research_summary_text", "No research summary generated."))
    for title, value in research_agent_findings.get("cards", {}).items():
        st.markdown(f"**{title}**")
        st.caption(value)

with tab9:
    _section("Indian Context Signals")
    for card in indian_context_cards:
        with st.container(border=True):
            st.markdown(_status_chip(card["Indicator"], card["Status"]), unsafe_allow_html=True)
            st.caption(f"Type: {card['Type']}")
            st.caption(card["Detail"])
            st.caption(f"Five C relevance: {card['Five C Relevance']}")

with tab10:
    _section("Hybrid Recommendation")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Scorecard Risk View**")
        st.write(f"- Decision: {recommendation['decision']}")
        st.write(f"- Risk Category: {score_bundle['risk_category']}")
        st.write(f"- Suggested Limit: INR {recommendation['recommended_limit']:,.0f}")
        st.write(f"- Pricing: {recommendation['interest_band']}")
    with c2:
        st.markdown("**ML Risk View**")
        st.write(f"- Decision: {ml_view['ml_decision']}")
        st.write(f"- Risk View: {ml_view['ml_risk']}")
        st.write(f"- Confidence: {ml_view['ml_confidence']}")
        st.write(f"- Probabilities: {ml_view['class_probabilities']}")

    st.markdown("**Final Hybrid Decision**")
    st.success(
        f"Final: {hybrid_view['final_decision']} | Limit: INR {hybrid_view['hybrid_limit']:,.0f} | "
        f"Pricing: {hybrid_view['hybrid_pricing_band']} | Confidence: {hybrid_view['decision_confidence']:.2f}"
    )
    st.caption(hybrid_view["alignment_note"])
    st.caption(hybrid_view["rationale"])

with tab11:
    _section("ML Diagnostics")

    m1, m2, m3 = st.columns(3)
    m1.metric("Model Decision", ml_view["ml_decision"])
    m2.metric("Confidence", f"{ml_view['ml_confidence']:.2f}")
    m3.metric("Model Accuracy (5-fold CV)", f"{ml_view.get('model_cv_accuracy', 0):.1%}")
    st.caption(
        "RandomForest (260 trees, depth 8) over 22 engineered features, trained on "
        "700 synthetic borrowers. Accuracy is held-out cross-validation on that "
        "synthetic set — it measures whether the model learned the generating "
        "structure, not how it would perform on real lending data."
    )

    st.markdown(f"**Why the model said {ml_view['ml_decision']} for this borrower**")
    st.caption(
        "Each row asks: if this borrower were typical on that one input, how much "
        "less confident would the model be in the decision it actually made? "
        "'Supports' means the borrower's real value pushed toward this decision; "
        "'opposes' means the decision was reached despite it. This is computed per "
        "borrower — unlike global feature importance, which is identical for everyone."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Feature": row["feature"],
                    "This Borrower": row["value"],
                    "Typical Borrower": row["typical_value"],
                    "Effect": row["direction"],
                    "Contribution": row["contribution"],
                }
                for row in ml_view["top_features"]
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    opposing = [f for f in ml_view["top_features"] if f["direction"] == "opposes"]
    if opposing:
        st.info(
            "**Reached despite:** "
            + ", ".join(f["feature"] for f in opposing[:4])
            + ". These inputs argued against the model's own conclusion."
        )

    st.markdown("**Engineered Features Used**")
    st.dataframe(pd.DataFrame([ml_features]).T.rename(columns={0: "value"}), use_container_width=True)

with tab12:
    _section("Export CAM")
    st.text_area("CAM Preview", value=cam_text, height=420)
    e1, e2, e3 = st.columns(3)
    with e1:
        st.download_button(
            "Download CAM (.txt)",
            cam_text,
            f"Intelli_Credit_CAM_{company_name.replace(' ', '_')}.txt",
            "text/plain",
            key="cam_txt_export",
            use_container_width=True,
        )
    with e2:
        if docx_bytes:
            st.download_button(
                "Download CAM (.docx)",
                docx_bytes,
                f"Intelli_Credit_CAM_{company_name.replace(' ', '_')}.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="cam_docx_export",
                use_container_width=True,
            )
        else:
            st.caption("DOCX dependency unavailable.")
    with e3:
        if pdf_bytes:
            st.download_button(
                "Download CAM (.pdf)",
                pdf_bytes,
                f"Intelli_Credit_CAM_{company_name.replace(' ', '_')}.pdf",
                "application/pdf",
                key="cam_pdf_export",
                use_container_width=True,
            )
        else:
            st.caption("PDF dependency unavailable.")

with tab13:
    _section("Secondary Research")
    st.caption(
        "Signals from outside the uploaded documents: promoter and management news, "
        "litigation, NCLT/insolvency filings, and regulatory action."
    )

    _mode = live_research.get("mode")
    if _mode == "live":
        st.success(f"**Live** — {live_research.get('status')}")
    elif _mode == "offline_sample":
        st.warning(
            f"**Offline sample** — {live_research.get('status')} "
            "These headlines are synthetic fixtures bundled for demonstration. "
            "They are **not real reporting** about this borrower, and every title "
            "is prefixed `[OFFLINE SAMPLE]` wherever it appears, including in the "
            "exported CAM."
        )
    else:
        st.info(f"**Disabled** — {live_research.get('status')}")

    if live_research.get("queries"):
        with st.expander("Queries issued"):
            for q in live_research["queries"]:
                st.markdown(f"- `{q['label']}` — {q['query']}")

    if live_research.get("headlines"):
        st.markdown("**Headlines retrieved**")
        for h in live_research["headlines"]:
            title = h.get("title", "")
            link = h.get("link", "")
            bullet = f"- [{title}]({link})" if link else f"- {title}"
            st.markdown(f"{bullet}  \n  <small>{h.get('source', '')} · {h.get('label', '')}</small>",
                        unsafe_allow_html=True)

    st.markdown("**Score impact**")
    if news_adjustments:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Signal": a["signal"],
                        "Five C": a["factor"],
                        "Points": a["points"],
                        "Evidence": a["evidence_snippet"][:110],
                        "Source": a["evidence_source"],
                    }
                    for a in news_adjustments
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            f"{len(news_adjustments)} of a maximum 3 news-derived adjustments applied. "
            "One per risk category, so repeated coverage of the same event counts once."
        )
    elif live_research.get("headlines"):
        st.success("No credit-risk language detected in the retrieved headlines.")
    else:
        st.caption("No headlines retrieved, so secondary research did not affect the score.")

with tab14:
    _section("How Intelli-Credit Works")
    _steps = [
        ("Ingest", "CSV, XLSX and PDF with OCR fallback"),
        ("Extract", "Figures with evidence; gaps declared, not filled"),
        ("Cross-verify", "GST turnover against actual bank inflows"),
        ("Retrieve", "TF-IDF evidence across six risk themes"),
        ("Research", "News, insolvency and regulatory signals"),
        ("Score", "Five Cs, 100 points, every change logged"),
        ("Predict", "RandomForest second opinion"),
        ("Fuse", "ML needs 90% confidence to override"),
        ("Report", "29-section memo, TXT / DOCX / PDF"),
    ]
    st.markdown(
        "<div class='ic-flow'>"
        + "".join(
            f"<div class='ic-flow-step'><div class='ic-flow-num'>{i}</div>"
            f"<div class='ic-flow-title'>{title}</div>"
            f"<div class='ic-flow-desc'>{desc}</div></div>"
            for i, (title, desc) in enumerate(_steps, start=1)
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='ic-rule'></div>", unsafe_allow_html=True)
    _section(
        "One ledger, every signal",
        "Whatever produces a signal — a document, a bank statement, a typed note or a "
        "news headline — it enters the score the same way: which of the Five Cs it "
        "affects, how many points, why, and what evidence supports it. That is why the "
        "memo, the explainability table and the audit trail all fall out of one structure.",
    )

st.caption("Hackathon prototype. Mock intelligence is used where live integrations are unavailable.")
