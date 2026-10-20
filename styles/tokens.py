"""Design tokens for Jet Engine RUL Prognostics System.

Single source of truth: colours, fonts, spacing. Charts and injected CSS read from here.
No gradients, shadows, glassmorphism or floating cards. Flat, quiet, terminal style.
"""

from __future__ import annotations

# --- surfaces and text ---
BG_PAGE = "#0b0d10"          # Page background (near-black)
BG_PANEL = "#12151a"          # Panel body (about 4 points lighter than page)
BG_PANEL_HEADER = "#161a21"   # Panel header strip
BORDER_HAIRLINE = "#1f2430"   # 1px rules and borders (replaces shadows)
TEXT_PRIMARY = "#f2f4f8"      # Numbers, titles
TEXT_SECONDARY = "#a2a9b0"    # Section titles, explanations
TEXT_MUTED = "#6f7782"        # Labels, sub-captions, axis text
TEXT_HERO_SECONDARY = "#c6c6c6"  # Subtitle / description text on hero banner

# --- accent and semantic ---
ACCENT = "#4589ff"            # The only brand accent
ACCENT_DIM = "#2a5db0"        # Pressed / secondary accent
POSITIVE = "#42be65"          # Better / improved (sign of number only)
NEGATIVE = "#fa4d56"          # Worse / over-optimistic (sign of number only)
WARNING = "#f1c21b"           # Disclaimer strip border, caution chips

# --- series colours (consistent on every chart) ---
SERIES_TRUE_RUL = "#c6c6c6"           # Dotted grey
SERIES_CONSTANT = "#6f7782"           # Thin solid
SERIES_RANDOM_FOREST = "#78a9ff"      # Solid 1.5px
SERIES_GRADIENT_BOOSTING = "#4589ff"  # Solid 1.5px (main model)
SERIES_SAFE_SIDE = "#08bdba"          # Teal, distinct from red/green
SERIES_P10_P90_BAND = "rgba(69,137,255,0.18)"  # Filled band, no border

# --- heatmap scale (dark to light blue) ---
HEATMAP_COLORSCALE = [
    "#0f1b2d",
    "#1d3557",
    "#2a5db0",
    "#4589ff",
    "#a6c8ff",
]

# --- typography ---
FONT_MONO = "IBM Plex Mono"
FONT_SANS = "IBM Plex Sans"

# --- spacing ---
SPACING_SMALL = 4
SPACING_MEDIUM = 8
SPACING_LARGE = 12
SPACING_XLARGE = 16
SPACING_XXLARGE = 24
SPACING_XXXLARGE = 32

# --- product name (never "AeroGuard" in the UI) ---
PRODUCT_NAME = "Jet Engine RUL Prognostics System"

# --- disclaimer (always visible on every page) ---
DISCLAIMER_TEXT = (
    "NASA C-MAPSS simulation data only. Not for real maintenance or airworthiness decisions."
)
