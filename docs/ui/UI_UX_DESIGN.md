# Jet Engine RUL Prognostics System: UI/UX Design Specification

Remaining useful life prediction and sensor-fault stress testing for turbofan engines.

| Field | Value |
| --- | --- |
| Document | UI_UX_DESIGN.md (main UI/UX spec) |
| Companion | WIREFRAMES.md (text wireframes for every page and state) |
| Product name | **Jet Engine RUL Prognostics System** (use this exact name everywhere) |
| Framework | Streamlit + Plotly (2D charts only) |
| Data | NASA C-MAPSS FD001 (simulated). Read-only precomputed artifacts. |
| Visual reference | Dense, dark, institutional "terminal" dashboard (IBM Carbon discipline). Copy the style, not the content. |
| Status | Proposed design. Nothing here is a result. |
| Revision | v1.1: adds a hero banner (image, project name, short description) at the top of Engine View |

---

## 1. Purpose and principles

The interface lets a reviewer answer three questions quickly:

1. How well does each model track an engine's remaining life? (Engine View)
2. Which model is best, and which one is safest? (Model Comparison)
3. What happens to those predictions when the sensors go bad? (Stress Lab)

### Design principles

| # | Principle | What it means in practice |
| --- | --- | --- |
| P1 | Numbers first | Big mono numbers and charts lead. Prose is short and plain. |
| P2 | General to specific | KPI band, then charts, then detailed tables below the fold. |
| P3 | Flat and quiet | No gradients, shadows, glow, glassmorphism or floating cards. Panels are separated by 1px lines. |
| P4 | One accent | Blue is the only accent. Green and red show the direction of a number, never decoration. |
| P5 | Honest | Every value comes from an artifact file. Show provenance (grid vs Nemotron). Always show the simulation disclaimer. |
| P6 | Explain every metric | Each metric has a one-sentence plain-English meaning. |
| P7 | Motion informs | Animation only introduces data. It never decorates. |
| P8 | Fail clearly | A missing artifact shows the exact command that generates it. |

### Hard constraints

- 3 pages only: Engine View, Model Comparison, Stress Lab.
- 2D Plotly charts only. No 3D.
- No chatbot, no AI-written text on the dashboard, no user accounts, no database.
- No hardcoded results. No invented numbers.
- The disclaimer is always visible: "NASA C-MAPSS simulation data only. Not for real maintenance or airworthiness decisions."
- Never use the old name "AeroGuard" in the UI.

---

## 2. Information architecture

```
Jet Engine RUL Prognostics System
├── Engine View          (default page after the site opens)
├── Model Comparison
└── Stress Lab           (Robustness Lab)
```

There is no separate landing page in v1. Engine View is the opening screen, and it starts with a hero banner (section 7.5) that shows the project name and a short description. An optional full landing page is listed under future scope (section 14).

### Navigation

- Sidebar with 3 page links. The active page is marked with a 2px accent bar on the left and primary text colour.
- URL query parameters are not required in v1.

---

## 3. Colour palette (design tokens)

Single source of truth: `styles/tokens.py`. Charts and CSS both read from it.

### 3.1 Surfaces and text

| Token | Hex | Use |
| --- | --- | --- |
| `bg_page` | `#0b0d10` | Page background (near-black) |
| `bg_panel` | `#12151a` | Panel body (about 4 points lighter than the page) |
| `bg_panel_header` | `#161a21` | Panel header strip |
| `border_hairline` | `#1f2430` | 1px rules and borders (replaces shadows) |
| `text_primary` | `#f2f4f8` | Numbers, titles |
| `text_secondary` | `#a2a9b0` | Section titles, explanations |
| `text_muted` | `#6f7782` | Labels, sub-captions, axis text |
| `hero_overlay` | `#0b0d10` at 55% opacity | Flat layer over the hero image (no gradient) |
| `text_hero_secondary` | `#c6c6c6` | Subtitle and description text on the hero banner |

### 3.2 Accent and semantic colours

| Token | Hex | Rule |
| --- | --- | --- |
| `accent` | `#4589ff` | The only brand accent. Active tab, selected item, main data line. |
| `accent_dim` | `#2a5db0` | Pressed or secondary accent |
| `positive` | `#42be65` | Better / improved. Sign of a number only. |
| `negative` | `#fa4d56` | Worse / over-optimistic. Sign of a number only. |
| `warning` | `#f1c21b` | Disclaimer strip border, caution chips |

Rules:
- Green and red are never used as decorative fills or backgrounds of large areas.
- Never rely on colour alone. Also use signs (+/-), labels or icons.

### 3.3 Series colours (consistent on every chart)

| Series | Hex | Style |
| --- | --- | --- |
| True / reference RUL | `#c6c6c6` | Dotted, 1.5px |
| Constant baseline | `#6f7782` | Thin solid |
| Random Forest | `#78a9ff` | Solid 1.5px |
| Gradient Boosting | `#4589ff` | Solid 1.5px (main model) |
| Safe-side model | `#08bdba` | Solid 1.5px (teal, distinct from red/green) |
| P10 to P90 range | `rgba(69,137,255,0.18)` | Filled band, no border (or dashed accent at 50% opacity) |

### 3.4 Heatmap scale (dark to light blue)

`#0f1b2d` → `#1d3557` → `#2a5db0` → `#4589ff` → `#a6c8ff`

Numbers are printed in every cell (mono). A colour legend bar sits under the heatmap with min and max labels.

### 3.5 Contrast

- Body text on panels meets at least 4.5:1.
- `text_muted` is for non-essential labels only.

---

## 4. Typography

| Role | Font | Size | Weight | Notes |
| --- | --- | --- | --- | --- |
| Section title | IBM Plex Sans | 11px | 600 | UPPERCASE, letter-spacing 0.08em, `text_secondary` |
| KPI label | IBM Plex Sans | 10px | 500 | UPPERCASE, `text_muted` |
| KPI value | **IBM Plex Mono** | 24 to 28px | 500 | `text_primary` |
| KPI subtext | IBM Plex Sans | 10px | 400 | `text_muted` |
| Body | IBM Plex Sans | 13 to 14px | 400 | `text_secondary` |
| Numbers, dates, axis labels, table values | **IBM Plex Mono** | 11 to 13px | 400 | Always mono |
| Page title (top bar) | IBM Plex Sans | 13px | 700 | UPPERCASE, letter-spacing 0.12em |

Rules:
- Every number, percentage, date and axis label uses IBM Plex Mono.
- Enable tabular numerals: `font-variant-numeric: tabular-nums;` so digits align in columns.
- Load fonts from Google Fonts in the injected CSS (self-host later if needed).
- Plain sentences. No jargon without an explanation.

---

## 5. Spacing, grid and shape

| Item | Value |
| --- | --- |
| Spacing scale | 4, 8, 12, 16, 24, 32 px only |
| Panel inner padding | 12 to 16 px |
| Gap between panels | 0 to 8 px (panels share 1px borders) |
| Border radius | 0 to 2 px (square look) |
| Border | 1px `border_hairline` |
| Shadows | none |
| Layout width | Streamlit wide layout |
| KPI band | 5 to 8 equal cells, 1px vertical separators, 1px top and bottom border |
| Hero banner height | 200px desktop, 160px below about 1280px, 144px below about 900px (all multiples of 8) |
| Min supported width | Laptop (about 1280px). Below about 900px, KPI band wraps to 2 columns. |

Column patterns (see WIREFRAMES.md):
- Full width: KPI band, main chart, tables.
- 2 columns (about 8/4): main chart plus side card.
- 3 columns: small panels (for example per-model bars, coverage card, metrics explained).

---

## 6. Streamlit setup

### 6.1 `.streamlit/config.toml`

```toml
[theme]
base = "dark"
backgroundColor = "#0b0d10"
secondaryBackgroundColor = "#12151a"
primaryColor = "#4589ff"
textColor = "#f2f4f8"
font = "sans serif"

[server]
headless = true
```

### 6.2 Global CSS (injected once with `st.markdown(unsafe_allow_html=True)`)

Must do:
- Hide Streamlit menu, footer and default header. Reduce top padding.
- Restyle the sidebar, tabs, selectboxes, radios, sliders, dataframes and metrics using the tokens.
- Thin dark scrollbars.
- Apply mono and tabular numerals to numbers.
- `set_page_config(page_title="Jet Engine RUL Prognostics System", layout="wide")`.

### 6.3 Plotly template `jerps`

Apply to every figure:

| Setting | Value |
| --- | --- |
| Paper and plot background | transparent |
| Font | IBM Plex Mono, 11px, `text_secondary` |
| Gridlines | hairline `#1f2430`, no zero-line glow |
| Axes | no border; tick labels in mono |
| Line width | 1.5px; no markers unless needed |
| Legend | horizontal, top-left, small |
| Hover | unified mode with vertical crosshair; label background `bg_panel`, hairline border |
| Mode bar | hidden |
| Margins | tight (about 8 to 16 px) |

---

## 7. Global chrome (on every page)

### 7.1 Top bar

- Left: `JET ENGINE RUL PROGNOSTICS SYSTEM` (uppercase, small, bold, spaced). Under it, the subtitle in muted text.
- Right: muted tag `FD001 · NASA C-MAPSS` and an outlined `SNAPSHOT` badge.
- On Engine View the top bar is not drawn. The hero banner (7.5) takes its place and carries the title, subtitle, tag and badge. Model Comparison and Stress Lab keep this top bar.

### 7.2 Disclaimer strip

- Directly under the top bar (on Engine View: directly under the hero banner), so it is always on the first screen. 1px border with a 3px `warning` left border.
- Text: "NASA C-MAPSS simulation data only. Not for real maintenance or airworthiness decisions."

### 7.3 Sidebar

1. Project name.
2. Page links: Engine View, Model Comparison, Stress Lab.
3. Status list (small, muted):
   - Dataset: FD001
   - Artifacts: OK / missing (per file)
   - Model version / schema: read from the resolved config artifact
   - GitHub link
   - "Read-only: precomputed results"

### 7.4 Footer

Small muted uppercase line: `SIMULATED DATA · NOT FOR OPERATIONAL USE · CHARTS BY PLOTLY · GENERATED <date from artifact>`

### 7.5 Hero banner (Engine View only)

A wide, short image strip at the very top of the first page. It carries the project name and a short description. It does not appear on Model Comparison or Stress Lab.

| Item | Spec |
| --- | --- |
| Position | Top of Engine View, above the disclaimer strip and the engine header. Replaces the top bar on this page. |
| Image file | `assets/hero.webp` (source 2940 x 802, WebP, about 80 KB; keep under 300 KB) |
| Shape | Very wide strip. The source is about 3.7:1 and is shown cropped to the heights below. |
| Height | 200px desktop, 160px below about 1280px, 144px below about 900px |
| Width | Full width of the page area |
| Fit | `object-fit: cover` (or CSS `background-size: cover`) with `object-position: 50% 60%` so the exhaust flame and engine body stay in view. Tune this value if the crop looks wrong. |
| Overlay | Flat `hero_overlay` layer (`#0b0d10` at 55% opacity) so the text stays readable. No gradient, no shadow, no glow. |
| Border | 1px `border_hairline` along the bottom edge |
| Content block | Left-aligned, 32px from the left edge, vertically centred, max width about 640px |
| Line 1: project name | `JET ENGINE RUL PROGNOSTICS SYSTEM`. IBM Plex Sans 700, 20 to 22px, UPPERCASE, letter-spacing 0.12em, `text_primary` |
| Line 2: subtitle | `Remaining useful life prediction and sensor-fault stress testing for turbofan engines`. 14px, `text_hero_secondary` |
| Line 3: short description | At most two lines, 13px, `text_hero_secondary`: "Predicts how many flights a simulated turbofan engine has left, then stress-tests those predictions with noisy, drifting and missing sensor readings." |
| Chips | Two outlined chips under the text: `FD001 · NASA C-MAPSS` and `SIMULATED DATA`. The `SNAPSHOT` badge sits at the top-right of the hero. |
| Image credit | Optional, 10px muted text at the bottom-right: "Photo: <credit>" |

Rules:
- The text is real HTML text, never baked into the image, so it stays sharp on every screen.
- The description says what the system does. It must not claim any result or accuracy.
- The image is for atmosphere only. Do not caption it as the engine in the dataset (C-MAPSS is a simulation).
- The KPI band must still be visible without scrolling on a 1280 x 720 laptop screen.
- Streamlit cannot point CSS at a local file. Read `assets/hero.webp` once, encode it as a base64 data URI, cache it (for example with `st.cache_resource`), and embed it in the hero HTML block.
- If the file is missing, show a flat `bg_panel` block of the same height with the same text and an `info` chip ("hero image missing"). Never crash.
- No parallax and no animation on the image.

---

## 8. Component library

Each component lives in its own file under `components/`.

| Component | Purpose | Anatomy | Notes |
| --- | --- | --- | --- |
| `kpi_band(items)` | Headline numbers | Cell = tiny uppercase label, large mono value, tiny muted subtext | Value is green/red only when it is a signed change. |
| `panel(title, subtitle)` | Container | Header row (uppercase title left, muted subtitle or controls right), flat body, 1px border | No shadow. |
| `metric_row(name, value, explanation)` | Metrics explained | Left name, middle mono value, right plain-English sentence | Rows divided by hairlines. |
| `sparkline(values)` | Inline trend | SVG about 84x18px, 1.5px stroke, no axes | Green if improving, red if worsening. |
| `segmented_toggle(options)` | Small mode switch | Bordered button group, active filled with accent | Used for fault kind and ranges. |
| `status_chip(text, kind)` | Labels and provenance | Tiny outlined chip: `info`, `ok`, `warning` | Used for "grid + nemotron". |
| `data_table(df)` | Sortable tables | Mono, right-aligned numbers, hairline rows, sticky header | Signed error cells coloured green/red. |
| `empty_state(message, command)` | Missing artifact | Flat bordered box with the exact command to run | See section 11. |
| `heatmap(df)` | Fault grid | Blue scale (3.4), numbers in cells, legend bar | Plotly heatmap. |
| `hero_banner(image_path, title, subtitle, description, tags)` | Engine View header image | Wide image strip, flat dark overlay, project name, subtitle, short description, chips (see 7.5) | Image embedded as base64. Falls back to a plain panel if the file is missing. |

### Component states

| State | Treatment |
| --- | --- |
| Default | As specified |
| Hover (table row, chart) | Row background `#161a21`; chart shows crosshair and exact values |
| Active / selected | Accent colour (text or 2px bar), never a large fill |
| Focus | 2px accent outline (keyboard users) |
| Disabled | `text_muted`, no hover change |

---

## 9. Page specifications

### 9.1 Page 1: Engine View (default)

Goal: see how each model tracks true remaining life for one engine.

| Block | Content | Source |
| --- | --- | --- |
| Hero banner | Wide image strip with the project name, subtitle, short description and chips (section 7.5). Replaces the top bar on this page only. | Static asset (`assets/hero.webp`) and fixed text, no artifact data |
| Header | "ENGINE <id>" label. Large mono predicted RUL at the selected cycle (flights left). Smaller true RUL and signed difference. Engine dropdown and cycle slider on the right. | Validation predictions artifact |
| KPI band | Engine MAE, RMSE, bias, worst overestimate, too-optimistic status | Computed from the per-engine rows |
| Main chart | X = cycle, Y = RUL. True RUL dotted grey. Model lines in series colours. P10 to P90 as a soft blue band. Vertical marker at the selected cycle. Model visibility toggle. | Validation predictions artifact |
| Table | Last 10 to 20 cycles: unit, cycle, true RUL, each model's prediction, signed error | Validation predictions artifact |

Rules:
- Positive bias means over-prediction (too optimistic). Show this convention in a small note.
- Distinguish "reference RUL" (benchmark) from unknown true RUL of a hypothetical new engine.

### 9.2 Page 2: Model Comparison

Goal: compare models overall and show the cost of being safe.

| Block | Content |
| --- | --- |
| KPI band | Best model's MAE, RMSE, NASA score, too-optimistic rate, cost of safety |
| Model scores table | One row per model (constant, random forest, gradient boosting, safe-side, plus quantile 10/90 where relevant). Columns: engine-mean MAE, RMSE, bias, worst-engine MAE, too-optimistic rate, NASA score. Seed spread (mean ± std over 5 seeds) as muted subtext. Optional sparkline of per-engine error. |
| Too optimistic panel | Horizontal bars per model at the fixed margin. Red means more engines predicted too high. |
| Range coverage card | P10 to P90 coverage versus the 0.80 nominal, with the known limitation written out |
| Metrics explained | `metric_row` for MAE, RMSE, bias, NASA score, too-optimistic rate, cost of safety |

Rules:
- NASA score is shown prominently, with the direction convention stated.
- Note on targets: training used the target policy recorded in the config artifact. State it once.

### 9.3 Page 3: Stress Lab (Robustness Lab)

Goal: show how predictions degrade under sensor faults and what safety costs.

| Block | Content |
| --- | --- |
| KPI band | Scenarios run, seeds, worst fault, largest MAE increase, cost of safety |
| Provenance chips | "Fault scenarios: grid (N) + nemotron (M)" or "grid only" |
| Controls | Segmented toggle for fault kind (Noise / Drift / Missing readings). Model multiselect. |
| Chart A: Error increase vs fault size | X = fault magnitude, Y = change in MAE vs clean data, one line per model, zero baseline, negative = improved |
| Chart B: Cost of safety | Safe-side MAE minus gradient-boosting MAE across the same magnitudes, with a zero line and the note "positive = the safe side pays extra, negative = it wins" |
| Heatmap: Which faults hurt most | Rows = fault kind and sensor, columns = magnitude, colour = MAE increase, numbers in cells, legend bar |
| Stress results table | Scenario, kind, magnitude, **source (grid / nemotron)**, clean MAE, fault MAE, change per model. Source column always visible. |

Rules:
- Faults apply to held-out data only.
- Magnitudes per fault kind come from the artifact, not from the UI.
- Nemotron is shown only as the origin of some scenarios (provenance). It never produces numbers.

---

## 10. Interaction and motion

| Item | Spec |
| --- | --- |
| Chart hover | Crosshair; label shows engine, cycle, model and exact mono value |
| First load | Short line draw-in (under 600ms). Numbers may count in. Then still. |
| Reduced motion | Respect `prefers-reduced-motion`: disable all animation |
| Tables | Sortable; keyboard accessible |
| Controls | Engine dropdown, cycle slider, model toggles, fault-kind toggle |
| No | Hover bounce, parallax, decorative animation, auto-refresh |

---

## 11. States

| State | Treatment |
| --- | --- |
| Loading | Thin 2px accent progress line under the top bar |
| Missing artifact | Flat bordered message: what is missing and the exact command to generate it (for example `conda run -n ml python train.py`, then `python stress_test.py`) |
| Partial data | Render what exists; label the missing block with an `info` chip |
| No Nemotron scenarios | Chip reads "grid only"; page still fully works |
| Invalid selection | Inline muted message; never a silent crash |
| Missing hero image | Plain `bg_panel` block of the same height with the same title and description, plus an `info` chip |

---

## 12. Accessibility

- Text contrast at least 4.5:1.
- Colour is never the only signal: use signs, labels and chips.
- Keyboard focus visible (2px accent outline).
- Every chart has a short text title and, where useful, a one-line caption.
- The hero image is decorative: give its container `role="img"` and an `aria-label` such as "Turbofan engine on a test stand, dark blue banner". The title and description stay real text.
- Text on the hero meets at least 4.5:1 against the image plus overlay. Darken the overlay if it does not.
- Respect reduced motion.

---

## 13. Implementation notes

### 13.1 Suggested files

```
styles/
  tokens.py            # all colours, fonts, spacing (single source of truth)
  global.css           # injected once
  plotly_template.py   # "jerps" template
components/
  kpi_band.py
  panel.py
  metric_row.py
  sparkline.py
  segmented_toggle.py
  status_chip.py
  data_table.py
  empty_state.py
  heatmap.py
  hero_banner.py
views/
  1_Engine_View.py
  2_Model_Comparison.py
  3_Stress_Lab.py
dashboard.py           # entry point, chrome, artifact loading
assets/
  hero.webp            # hero banner image (about 80 KB)
.streamlit/config.toml
docs/ui/               # optional small text or image wireframes
```

### 13.2 Rules

- The dashboard reads artifacts only. It never trains or calls Nemotron.
- Load artifacts once with caching; show the status in the sidebar.
- All colours, fonts and spacing come from `tokens.py`.
- Do not duplicate feature logic in the UI.

---

## 14. Future scope (UI)

| Idea | When |
| --- | --- |
| Landing page: title, one-line explanation, three headline results, "Open Dashboard" button | After the hackathon, if time allows |
| Light theme variant | Only if reviewers ask |
| Downloadable result tables (CSV) | After the core pages are stable |
| Native React front end for pixel-exact terminal styling | Paper phase |

---

## 15. Acceptance checklist

- [ ] The name "Jet Engine RUL Prognostics System" appears in the tab title, top bar, sidebar, README and footer.
- [ ] The disclaimer is visible on every page.
- [ ] Only the 3 specified pages exist; no 3D charts.
- [ ] All colours and fonts match the tokens; every number is IBM Plex Mono with tabular numerals.
- [ ] Green and red appear only as sign indicators.
- [ ] Every metric has a plain-English explanation.
- [ ] Stress Lab shows the Source column and provenance chips.
- [ ] Missing artifacts show an actionable message, not a stack trace.
- [ ] Every number traces to an artifact file; nothing hardcoded.
- [ ] Contrast, focus and reduced-motion requirements are met.
- [ ] Engine View opens with the hero banner showing the project name and short description. Model Comparison and Stress Lab do not show it.
- [ ] The hero is 200px tall on desktop with a flat overlay (no gradient), and the KPI band is visible without scrolling at 1280 x 720.
- [ ] A missing hero image shows the plain fallback panel, not an error.

---

## 16. Visual references

- Style reference: the dense dark "terminal" dashboard at https://github.com/chrispathway/portfolio-dashboard (MIT licence; keep the licence notice if any code is copied).
- Design system discipline: IBM Carbon, https://carbondesignsystem.com/
- Streamlit gallery: https://streamlit.io/gallery
- Plotly examples: https://plotly.com/examples/energy-utility
- Turbofan dashboard example: https://dashtera.com/?p=26096
- Hero image: engine test-stand photo supplied by the project owner, cropped, tinted blue and darkened. Confirm its licence and add the credit to the README before publishing.
