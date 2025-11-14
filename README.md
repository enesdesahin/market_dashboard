# Cross-Asset Market Monitor

Interactive Streamlit workspace that keeps bonds, equities, currencies, and commodities in one place. The app blends live macro data (FRED, Yahoo Finance, OECD) with curated CSVs so you can compare regimes, spot anomalies, and export tidy tables in seconds.

---

## What You Can Do

- **Single launch point:** Run `streamlit run dashboard.py` once and navigate across tabs via Streamlit’s new multi-page API (see `dashboard.py`).
- **Macro-aware context:** Each plot can display the active market regime badge, shaded backdrops, and custom subtitle copy to highlight “why now”.
- **Actionable visuals:** Orange/grey gradients, consistent palettes, and cards that emphasize changes (e.g., Top Gainers, spread deltas).
- **Data retention:** Every loader writes to `data_cache/` so repeated queries load instantly even when offline; CSV downloads are exposed per tab.
- **Developer friendly:** Shared helpers live in `views/common.py`, so extending a tab is mostly configuration plus a Plotly figure.

---

## Getting Started

| Step | Command / Notes |
| --- | --- |
| **1. Prereqs** | Python 3.10+, `pip`, and internet connectivity for the initial data pull. |
| **2. Create env** | `python -m venv .venv && source .venv/bin/activate` (Windows: `.venv\Scripts\activate`). |
| **3. Install deps** | `pip install -r requirements.txt` (or `pip install streamlit pandas numpy plotly requests yfinance matplotlib`). |
| **4. Launch** | `streamlit run dashboard.py` |

Streamlit opens in wide mode. Use the sidebar to pick instruments, switch on/off regime overlays, adjust the lookback window, and refresh cached data.

---

## Tab Overview

| Tab | Primary Views | Extras |
| --- | --- | --- |
| **Stocks** (`views/equities.py`) | Normalized performance, 30D vol, drawdowns, correlation heatmaps | Subtitle callouts, Top Gainer cards, grey→orange heatmap scale |
| **Bonds** (`views/bonds.py`) | US Treasury curve, OECD 10Y peers, spread ladders, 3D vol surface | Explicit tenor color overrides, _regime-aware annotations_, cached FRED fetches |
| **Commodities** (`views/commodities.py`) | Metals, energy, grains, vol/drawdown tracking | Scenario captions, exportable tables, matching palettes |
| **Currencies** (`views/currencies.py`) | Major FX pairs, normalized scores, heatmap correlations | Cached FX data, consistent subtitles, optional scenario notes |

---

## Project Layout

| Path | Purpose |
| --- | --- |
| `dashboard.py` | Entry point that wires Streamlit’s navigation pages for Stocks/Bonds/Commodities/Currencies. |
| `views/` | One Python module per tab plus `common.py` shared utilities (regime shading, caching, figure helpers). |
| `data/` | Semi-static CSVs bundled with the repo (baseline FX/commodity histories). |
| `data_cache/` | Auto-generated cache files (Yahoo Finance equities/FX, FRED yields). Safe to delete to force a refresh. |
| `.streamlit/config.toml` | Dark theme + typography overrides (Rethink Sans, orange accent). |

---

## Configuration & Customization

- **Theme & palette:** Update `.streamlit/config.toml` to change colors, typography, or border radii. The palette is reused by default in each tab.
- **Default selections:** Every tab defines a `DEFAULT_SELECTION` tuple near the top. Edit those tuples to change the symbols that load on first render.
- **Regime visuals:** `views/common.py` owns `REGIME_COLORS`, `REGIME_ACCENTS`, and `REGIME_DESCRIPTIONS`. Adjust these maps to tweak shading, copy, or the regime badge.
- **Color scales:** Constants such as `HEATMAP_COLOR_SCALE` or `_orange_gradient` live inside `views/bonds.py`/`views/equities.py`. Update once to propagate across charts.
- **Social links:** The sidebar footer uses `render_social_links()` in `views/common.py`; pass custom URLs if you fork the project.

---

## Data & Caching Strategy

- **Live fetches:** `yfinance.download` covers equities/FX; FRED & OECD CSV endpoints supply yield data; Streamlit’s `@st.cache_data` wraps all heavy loaders.
- **On-disk cache:** Results land under `data_cache/`. Delete individual files (e.g., `data_cache/US10Y.csv`) to force a live refresh without touching the whole directory.
- **Bundled baselines:** `data/` ships with representative commodity/FX histories so the UI stays functional even without internet access.
- **CSV exports:** Each tab exposes clean tables and `st.download_button` outputs so you can take the data offline.

---

## Development Notes

- **Syntax check:** `python -m compileall dashboard.py views`
- **Streamlit hot reload:** `streamlit run dashboard.py --server.runOnSave true`
- **Adding a tab:** Copy one of the existing view modules, update constants for defaults/colors, register it inside `dashboard.py`, and reuse helpers from `views/common.py`.
- **Testing ideas:** Add pytest smoke tests for the loader functions and snapshot-test Plotly figure specs to catch layout regressions.

---

## Troubleshooting

| Issue | Things to try |
| ----- | ------------- |
| Missing dependencies | Recreate the virtualenv and reinstall packages. |
| Slow loads / timeouts | Temporarily disable live refresh, verify network connectivity, or increase cache TTLs. |
| Empty charts | Confirm the selected tickers have data for the chosen lookback window. |
| Stale/invalid cache | Remove the relevant files in `data/` or `data_cache/` and rerun the app. |

---

## Roadmap Ideas

- Saved “views” (e.g., Risk-On, Recession Watch) that auto-select symbols.
- Alert hooks for curve inversions, vol spikes, or FX regime flips.
- Background scheduler / API to precompute heavy stats before market open.

Issues and feature requests are always welcome. Enjoy exploring the markets! 🚀
