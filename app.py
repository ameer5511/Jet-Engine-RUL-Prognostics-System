"""Jet Engine RUL Prognostics System — dashboard entry point.

Read-only Streamlit app. Reads pre-computed artifacts under results/.
Never trains models, never calls Nemotron.
"""

from __future__ import annotations

import base64
import textwrap
from pathlib import Path

import streamlit as st

# Ensure styles package is importable when run via `streamlit run app.py`
import sys
sys.path.insert(0, str(Path(__file__).parent))

from styles.tokens import (
    ACCENT,
    BG_PAGE,
    BG_PANEL,
    BORDER_HAIRLINE,
    POSITIVE,
    NEGATIVE,
    TEXT_HERO_SECONDARY,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    DISCLAIMER_TEXT,
    PRODUCT_NAME,
)
from dashboard_data import (
    load_hero_base64,
    artifact_status,
    get_val_engines,
    get_prediction_models,
    load_predictions,
    load_metrics,
    load_resolved_config,
    load_stress_json,
    load_nemotron_settings,
    DISCLAIMER,
)

HERO_IMAGE_PATH = "docs/ui/assets/Hero-image.png"
HERO_HEIGHT_DESKTOP = "200px"
HERO_HEIGHT_TABLET = "160px"
HERO_HEIGHT_MOBILE = "144px"


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

def set_page_config() -> None:
    st.set_page_config(
        page_title=f"{PRODUCT_NAME}",
        page_icon="\U0001F680",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ---------------------------------------------------------------------------
# Global CSS (injected once per page)
# ---------------------------------------------------------------------------

def inject_global_css() -> None:
    css = f"""
    <style>
    :root {{
      --bg-page: {BG_PAGE};
      --bg-panel: {BG_PANEL};
      --border-hairline: {BORDER_HAIRLINE};
      --text-primary: {TEXT_PRIMARY};
      --text-secondary: {TEXT_SECONDARY};
      --text-muted: {TEXT_MUTED};
      --accent: {ACCENT};
      --positive: {POSITIVE};
      --negative: {NEGATIVE};
      --warning: {WARNING};
    }}

    /* Hide Streamlit chrome */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    .stApp {{background-color: {BG_PAGE}; padding-top: 0rem;}}

    /* Base typography */
    html, body, [class*="st-"] {{
      font-family: "IBM Plex Sans", sans-serif;
    }}
    .mono {{
      font-family: "IBM Plex Mono", monospace;
      font-variant-numeric: tabular-nums;
    }}

    /* Reduce top padding */
    .block-container {{
      padding-top: 0.5rem;
      padding-bottom: 1rem;
      max-width: 100%;
    }}

    /* Sidebar restyle */
    [data-testid="stSidebar"] {{
      background-color: {BG_PANEL};
      border-right: 1px solid {BORDER_HAIRLINE};
    }}
    [data-testid="stSidebar"] .stSelectbox > div > div {{
      background-color: {BG_PAGE};
      border: 1px solid {BORDER_HAIRLINE};
    }}

    /* Tabs / selectbox restyle */
    .stSelectbox > div > div {{
      background-color: {BG_PAGE};
      border: 1px solid {BORDER_HAIRLINE};
      color: {TEXT_PRIMARY};
    }}

    /* Thin dark scrollbars */
    ::-webkit-scrollbar {{width: 8px; height: 8px;}}
    ::-webkit-scrollbar-track {{background: {BG_PAGE};}}
    ::-webkit-scrollbar-thumb {{background: {BORDER_HAIRLINE}; border-radius: 0;}}
    ::-webkit-scrollbar-thumb:hover {{background: {TEXT_MUTED};}}

    /* Mono + tabular numerals on dataframes */
    .stDataFrame, .stTable {{font-family: "IBM Plex Mono", monospace;}}

    /* KPI card */
    .kpi-cell {{
      border: 1px solid {BORDER_HAIRLINE};
      border-top: 1px solid {BORDER_HAIRLINE};
      border-bottom: 1px solid {BORDER_HAIRLINE};
      padding: 8px 6px;
      text-align: center;
      background: {BG_PANEL};
    }}
    .kpi-label {{
      font-size: 10px;
      color: {TEXT_MUTED};
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
      line-height: 1.2;
    }}
    .kpi-value {{
      font-family: "IBM Plex Mono", monospace;
      font-size: 20px;
      font-weight: 500;
      color: {TEXT_PRIMARY};
      line-height: 1.2;
    }}
    .kpi-subtext {{
      font-size: 10px;
      color: {TEXT_MUTED};
      margin-top: 2px;
      line-height: 1.2;
    }}
    .kpi-sign-pos {{color: {POSITIVE};}}
    .kpi-sign-neg {{color: {NEGATIVE};}}

    /* Panel */
    .panel {{
      border: 1px solid {BORDER_HAIRLINE};
      background: {BG_PANEL};
      padding: 12px 16px;
      margin-bottom: 8px;
    }}
    .panel-header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 8px;
      border-bottom: 1px solid {BORDER_HAIRLINE};
      padding-bottom: 6px;
    }}
    .panel-title {{
      font-family: "IBM Plex Sans", sans-serif;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: {TEXT_SECONDARY};
    }}
    .panel-subtitle {{
      font-size: 11px;
      color: {TEXT_MUTED};
      font-style: italic;
    }}

    /* Metric row */
    .metric-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid {BORDER_HAIRLINE};
      padding: 6px 0;
      font-size: 13px;
    }}
    .metric-name {{color: {TEXT_SECONDARY}; width: 34%;}}
    .metric-value {{font-family: "IBM Plex Mono", monospace; width: 26%; color: {TEXT_PRIMARY};}}
    .metric-explain {{color: {TEXT_MUTED}; width: 40%; font-size: 12px;}}

    /* Disclaimer strip */
    .disclaimer {{
      border: 1px solid {BORDER_HAIRLINE};
      border-left: 3px solid {WARNING};
      background: {BG_PANEL};
      padding: 8px 12px;
      margin-bottom: 8px;
      font-size: 13px;
      color: {TEXT_SECONDARY};
      font-family: "IBM Plex Mono", monospace;
    }}

    /* Top bar (Model Comparison / Stress Lab only) */
    .top-bar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid {BORDER_HAIRLINE};
      padding: 8px 0 8px 0;
      margin-bottom: 8px;
    }}
    .top-bar-left {{
      font-family: "IBM Plex Sans", sans-serif;
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: {TEXT_PRIMARY};
    }}
    .top-bar-subtitle {{
      font-size: 12px;
      color: {TEXT_MUTED};
      margin-top: 2px;
    }}
    .top-bar-right {{
      display: flex;
      align-items: center;
      gap: 12px;
      font-size: 11px;
      color: {TEXT_MUTED};
      font-family: "IBM Plex Mono", monospace;
    }}
    .badge {{
      border: 1px solid {BORDER_HAIRLINE};
      padding: 2px 8px;
      border-radius: 2px;
      font-size: 10px;
      color: {TEXT_SECONDARY};
    }}

    /* Footer */
    .footer {{
      border-top: 1px solid {BORDER_HAIRLINE};
      padding-top: 8px;
      margin-top: 16px;
      font-size: 11px;
      color: {TEXT_MUTED};
      font-family: "IBM Plex Mono", monospace;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}

    /* Status chip */
    .chip {{
      display: inline-block;
      border: 1px solid {BORDER_HAIRLINE};
      padding: 2px 8px;
      border-radius: 2px;
      font-size: 10px;
      color: {TEXT_SECONDARY};
      font-family: "IBM Plex Mono", monospace;
    }}
    .chip-ok {{color: {POSITIVE}; border-color: {POSITIVE};}}
    .chip-warning {{color: {WARNING}; border-color: {WARNING};}}
    .chip-info {{color: {TEXT_MUTED}; border-color: {BORDER_HAIRLINE};}}

    /* Hero banner */
    .hero {{
      position: relative;
      width: 100%;
      height: {HERO_HEIGHT_DESKTOP};
      border-bottom: 1px solid {BORDER_HAIRLINE};
      margin-bottom: 8px;
      overflow: hidden;
      background: {BG_PANEL};
    }}
    .hero-img {{
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: 50% 60%;
      display: block;
    }}
    .hero-overlay {{
      position: absolute;
      inset: 0;
      background: {BG_PAGE};
      opacity: 0.55;
    }}
    .hero-content {{
      position: absolute;
      inset: 0;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding-left: 32px;
      max-width: 640px;
      z-index: 2;
    }}
    .hero-name {{
      font-family: "IBM Plex Sans", sans-serif;
      font-weight: 700;
      font-size: 21px;
      color: {TEXT_PRIMARY};
      text-transform: uppercase;
      letter-spacing: 0.12em;
      line-height: 1.2;
      margin-bottom: 6px;
    }}
    .hero-subtitle {{
      font-size: 14px;
      color: {TEXT_HERO_SECONDARY};
      line-height: 1.4;
      margin-bottom: 8px;
    }}
    .hero-desc {{
      font-size: 13px;
      color: {TEXT_HERO_SECONDARY};
      line-height: 1.5;
      max-width: 640px;
      margin-bottom: 10px;
      display: -webkit-box;
      -webkit-line-clamp: 2;
      -webkit-box-orient: vertical;
      overflow: hidden;
    }}
    .hero-chips {{
      display: flex;
      gap: 8px;
      margin-top: 8px;
    }}
    .hero-badge {{
      position: absolute;
      top: 12px;
      right: 12px;
      font-size: 10px;
      color: {TEXT_SECONDARY};
      border: 1px solid {BORDER_HAIRLINE};
      padding: 2px 8px;
      border-radius: 2px;
      font-family: "IBM Plex Mono", monospace;
      z-index: 3;
      background: rgba(11,13,16,0.6);
    }}

    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero banner (Engine View only)
# ---------------------------------------------------------------------------

def render_hero_banner() -> None:
    """Render the hero banner on Engine View. Falls back to plain panel if image missing."""
    uri = load_hero_base64(HERO_IMAGE_PATH)
    image_html = (
        f'<img src="{uri}" class="hero-img" alt="decorative" aria-hidden="true">'
        if uri
        else ''
    )
    fallback_html = (
        '<div style="width:100%;height:100%;background:#12151a;display:flex;'
        'align-items:center;justify-content:center;color:#6f7782;font-family:'
        '"IBM Plex Mono",monospace;font-size:11px;">'
        "Hero image missing"
        "</div>"
    ) if not uri else ""

    chips = (
        '<span class="chip chip-info">FD001 · NASA C-MAPSS</span>'
        '<span class="chip chip-info">SIMULATED DATA</span>'
    )

    st.markdown(
        textwrap.dedent(f"""
        <div class="hero">
        {image_html}
        <div class="hero-overlay"></div>
        <div class="hero-badge">SNAPSHOT</div>
        <div class="hero-content">
        <div class="hero-name">{PRODUCT_NAME.upper()}</div>
        <div class="hero-subtitle">Remaining useful life prediction and sensor-fault stress testing for turbofan engines.</div>
        <div class="hero-desc">
        Predicts how many flights a simulated turbofan engine has left, then stress-tests
        those predictions with noisy, drifting and missing sensor readings.
        </div>
        <div class="hero-chips">{chips}</div>
        </div>
        {fallback_html}
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Chrome components
# ---------------------------------------------------------------------------

def render_top_bar() -> None:
    """Top bar for Model Comparison and Stress Lab (not Engine View)."""
    st.markdown(
        textwrap.dedent(f"""
        <div class="top-bar">
        <div>
        <div class="top-bar-left">{PRODUCT_NAME.upper()}</div>
        <div class="top-bar-subtitle">FD001 · NASA C-MAPSS</div>
        </div>
        <div class="top-bar-right">
        <span class="badge">SNAPSHOT</span>
        </div>
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


def render_disclaimer_strip() -> None:
    """Disclaimer strip visible on every page."""
    st.markdown(
        f'<div class="disclaimer">{DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Sidebar with project name, page links, status list."""
    st.sidebar.markdown(
        f"<div style='font-family:\"IBM Plex Sans\",sans-serif;font-weight:700;"
        f"font-size:13px;color:{TEXT_PRIMARY};text-transform:uppercase;"
        f"letter-spacing:0.12em;margin-bottom:16px;'>{PRODUCT_NAME.upper()}</div>",
        unsafe_allow_html=True,
    )

    pages = ["Engine View", "Model Comparison", "Stress Lab"]
    st.sidebar.markdown(
        f"<div style='font-size:10px;color:{TEXT_MUTED};text-transform:uppercase;"
        f"letter-spacing:0.08em;margin-bottom:8px;'>{' · '.join(pages)}</div>",
        unsafe_allow_html=True,
    )

    status = artifact_status()
    status_lines = []
    status_lines.append("Dataset: FD001")
    artifacts_ok = status['predictions_csv'] and status['metrics_json']
    status_lines.append(f"Artifacts: {'OK' if artifacts_ok else 'missing'}")
    status_lines.append(f"Nemotron scenarios: {len(load_nemotron_settings())}")
    status_lines.append("Read-only: precomputed results")

    for line in status_lines:
        st.sidebar.markdown(
            f"<div style='font-family:\"IBM Plex Mono\",monospace;font-size:10px;"
            f"color:{TEXT_MUTED};margin-bottom:4px;'>{line}</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown(
        f"<div style='font-family:\"IBM Plex Mono\",monospace;font-size:9px;"
        f"color:{TEXT_MUTED};margin-top:24px;border-top:1px solid {BORDER_HAIRLINE};"
        f"padding-top:8px;'>{PRODUCT_NAME.upper()} · "
        f"<a href='https://github.com/ameer5511/Jet-Engine-RUL-Prognostics-System' "
        f"target='_blank' style='color:{ACCENT};text-decoration:none;'>GitHub</a></div>",
        unsafe_allow_html=True,
    )


def render_footer(snapshot_date: str | None = None) -> None:
    """Footer on every page."""
    date_text = snapshot_date or "—"
    st.markdown(
        textwrap.dedent(f"""
        <div class="footer">
        SIMULATED DATA · NOT FOR OPERATIONAL USE · CHARTS BY PLOTLY ·
        GENERATED {date_text}
        </div>
        """).strip(),
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# KPI band helper
# ---------------------------------------------------------------------------

def render_kpi_band(items: list[dict[str, str]]) -> None:
    """Render a horizontal KPI band: list of {label, value, subtext, sign}.
    sign: 'pos' | 'neg' | None for green/red value colour.
    """
    cells = ""
    for item in items:
        sign_cls = ""
        if item.get("sign") == "pos":
            sign_cls = "kpi-sign-pos"
        elif item.get("sign") == "neg":
            sign_cls = "kpi-sign-neg"
        cells += textwrap.dedent(f"""
        <div class="kpi-cell">
        <div class="kpi-label">{item['label']}</div>
        <div class="kpi-value mono {sign_cls}">{item['value']}</div>
        <div class="kpi-subtext">{item.get('subtext','')}</div>
        </div>
        """).strip()
    st.markdown(f'<div style="display:flex;gap:0;flex-wrap:wrap;">{cells}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Navigation entry point
# ---------------------------------------------------------------------------

pg = st.navigation(
    {
        "Overview": [
            st.Page("pages/0_Overview.py", title="Overview", default=True),
        ],
        "Explore": [
            st.Page("pages/1_Engine_View.py", title="Engine View"),
            st.Page("pages/2_Model_Comparison.py", title="Model Comparison"),
            st.Page("pages/3_Stress_Lab.py", title="Stress Lab"),
        ],
    }
)
pg.run()
