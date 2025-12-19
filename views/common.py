from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from streamlit.delta_generator import DeltaGenerator

# Default bounds used across the app. They can be overridden per section if needed.
# Shared min/max bounds so tabs behave consistently out of the box.
DEFAULT_START_DATE = date(2010, 1, 1)
DEFAULT_END_DATE = date.today()

# Central cache directory used by all download helpers.
DATA_CACHE_DIR = Path("data_cache")
DATA_CACHE_DIR.mkdir(exist_ok=True)

# Visual styling knobs for regime shading and badges.
REGIME_COLORS = {
    "Expansion": "rgba(34,197,94,0.26)",
    "Stagflation": "rgba(249,115,22,0.26)",
    "Recession": "rgba(239,68,68,0.26)",
    "Recovery": "rgba(59,130,246,0.26)",
}

REGIME_ACCENTS = {
    "Expansion": ("#22c55e", "rgba(34,197,94,0.14)"),
    "Stagflation": ("#f97316", "rgba(249,115,22,0.14)"),
    "Recession": ("#ef4444", "rgba(239,68,68,0.14)"),
    "Recovery": ("#3b82f6", "rgba(59,130,246,0.14)"),
}

REGIME_DESCRIPTIONS = {
    "Expansion": (
        "Growth is above trend, inflation remains contained, and risk assets tend to lead. "
        "Credit spreads usually tighten while defensive assets consolidate."
    ),
    "Stagflation": (
        "Economic momentum is soft but inflation stays elevated. "
        "This mix penalises duration and equities alike, favouring hard assets and inflation hedges."
    ),
    "Recession": (
        "Activity contracts and market volatility rises. "
        "Investors often rotate into safe havens, and earnings expectations reset lower."
    ),
    "Recovery": (
        "Growth stabilises after a slowdown while inflation cools. "
        "Early-cycle assets and credit typically outperform as confidence rebuilds."
    ),
}

GITHUB_PROFILE_URL = "https://github.com/enesdesahin"
LINKEDIN_PROFILE_URL = "https://linkedin.com/in/sahinenes42/"

_SOCIAL_SVGS = {
    "github": (
        "GitHub",
        """
<svg viewBox="0 0 24 24" role="img" aria-label="GitHub" xmlns="http://www.w3.org/2000/svg">
    <path fill="currentColor" d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 
    0-.285-.01-1.04-.015-2.04-3.338.726-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.757-1.333-1.757-1.089-.745.083-.73.083-.73 
    1.205.085 1.84 1.236 1.84 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.418-1.305.762-1.605-2.665-.304-5.466-1.332-5.466-5.932 
    0-1.31.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23a11.5 11.5 0 0 1 3.003-.404 
    c1.018.005 2.045.138 3.003.404 2.291-1.552 3.297-1.23 3.297-1.23.655 1.653.244 2.874.12 3.176.77.84 1.235 1.911 1.235 3.221 
    0 4.61-2.807 5.625-5.479 5.921.43.372.823 1.102.823 2.222 0 1.604-.015 2.896-.015 3.286 0 .319.216.694.825.576 
    C20.565 22.092 24 17.592 24 12.297 24 5.67 18.627.297 12 .297z"/>
</svg>
        """,
    ),
    "linkedin": (
        "LinkedIn",
        """
<svg viewBox="0 0 448 512" role="img" aria-label="LinkedIn" xmlns="http://www.w3.org/2000/svg">
    <path fill="currentColor" d="M416 32H32A32 32 0 0 0 0 64v384a32 32 0 0 0 32 32h384a32 32 0 0 0 32-32V64a32 32 0 0 0-32-32zM135.4 416H69.1V202.2h66.3zm-33.1-243a38.4 38.4 0 1 1 38.4-38.4 38.4 38.4 0 0 1-38.4 38.4zM384 416h-66.2V312c0-24.8-.5-56.7-34.5-56.7-34.5 0-39.8 27-39.8 54.9V416h-66.2V202.2h63.6v29.2h.9c8.9-16.8 30.6-34.5 63-34.5 67.3 0 79.7 44.4 79.7 102.1z"/>
</svg>
        """,
    ),
}

def render_social_links(
    *,
    github_url: str | None = None,
    linkedin_url: str | None = None,
) -> None:
    """Render social profile badges at the bottom of the sidebar."""
    github_target = github_url or GITHUB_PROFILE_URL
    linkedin_target = linkedin_url or LINKEDIN_PROFILE_URL

    entries: list[Tuple[str, str]] = []
    if github_target:
        entries.append(("github", github_target))
    if linkedin_target:
        entries.append(("linkedin", linkedin_target))

    if not entries:
        return

    badges = []
    for key, url in entries:
        label, svg_markup = _SOCIAL_SVGS.get(key, ("", ""))  # type: ignore[assignment]
        if not svg_markup:
            continue
        extra_style = "margin-left:-8px;" if key == "linkedin" else ""
        badge = (
            "<a href=\"{url}\" target=\"_blank\" rel=\"noopener noreferrer\" "
            "style=\"display:inline-flex;width:38px;height:38px;border-radius:999px;"
            "align-items:center;justify-content:center;color:#f8fafc;text-decoration:none;{extra_style}\" "
            "title=\"{label}\">"
            "<span style=\"display:inline-flex;width:22px;height:22px;color:#f8fafc;\">{svg}</span>"
            "</a>"
        ).format(url=url, label=label, svg=svg_markup, extra_style=extra_style)
        badges.append(badge)

    if not badges:
        return

    container = (
        "<div style=\"margin-top:1.5rem;padding-top:1rem;border-top:1px solid rgba(148,163,184,0.35);\">"
        "<div style=\"display:flex;flex-direction:column;align-items:flex-start;\">"
        "<div style=\"font-size:0.85rem;font-weight:600;color:#e2e8f0;margin-bottom:0.4rem;\">Connect with me</div>"
        f"<div style=\"display:flex;gap:0.3rem;margin-left:-6px;\">{''.join(badges)}</div>"
        "<div style=\"margin-top:0.45rem;font-size:0.8rem;color:rgba(248,250,252,0.65);\">Developed by Enes SAHIN</div>"
        "</div>"
        "</div>"
    )
    st.sidebar.markdown(container, unsafe_allow_html=True)

def _cache_path(symbol: str) -> Path:
    return DATA_CACHE_DIR / f"{symbol}.csv"


@st.cache_data(show_spinner=False)
def download_data(
    tickers: Tuple[str, ...],
    start_date: date,
    end_date: date,
    *,
    use_live: bool = True,
) -> pd.DataFrame:
    """Fetch adjusted close prices with optional CSV fallback when live data is disabled or unavailable."""
    if not tickers:
        return pd.DataFrame()

    tickers = tuple(tickers)
    fetched = pd.DataFrame()

    if use_live:
        try:
            price_frame = yf.download(
                list(tickers),
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False,
            )
            close_prices = price_frame["Close"] if "Close" in price_frame else price_frame
            if isinstance(close_prices, pd.Series):
                close_prices = close_prices.to_frame(name=tickers[0])

            close_prices.index = close_prices.index.tz_localize(None)

            if not close_prices.empty:
                fetched = close_prices
                for symbol in tickers:
                    if symbol not in fetched.columns:
                        continue
                    cache_series = fetched[[symbol]].rename(columns={symbol: "Close"})
                    path = _cache_path(symbol)
                    if path.exists():
                        existing = pd.read_csv(path, index_col=0, parse_dates=True)
                        existing.index = existing.index.tz_localize(None)
                        cache_series = pd.concat([existing, cache_series]).sort_index()
                        cache_series = cache_series[~cache_series.index.duplicated(keep="last")]
                    cache_series.to_csv(path)
        except Exception:
            fetched = pd.DataFrame()

    if fetched.empty:
        cached_frames = []
        for symbol in tickers:
            path = _cache_path(symbol)
            if not path.exists():
                continue
            cached = pd.read_csv(path, index_col=0, parse_dates=True)
            cached.index = cached.index.tz_localize(None)
            cached_frames.append(cached.rename(columns={cached.columns[0]: symbol}))
        if cached_frames:
            fetched = pd.concat(cached_frames, axis=1).sort_index()

    if fetched.empty:
        return pd.DataFrame()

    mask = (fetched.index >= pd.Timestamp(start_date)) & (fetched.index <= pd.Timestamp(end_date))
    return fetched.loc[mask]


def load_data(file_path: str) -> pd.DataFrame:
    """Load a price series stored on disk."""
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    data.index = data.index.tz_localize(None)
    return data


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill and remove missing observations."""
    if data.empty:
        return data
    return data.ffill().dropna()


def ensure_datetime_index(data: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with a DatetimeIndex, even when the input is empty."""
    if data.empty:
        if isinstance(data.index, pd.DatetimeIndex):
            return data
        safe = data.copy()
        safe.index = pd.DatetimeIndex([], name="Date")
        return safe
    if isinstance(data.index, pd.DatetimeIndex):
        return data
    safe = data.copy()
    safe.index = pd.to_datetime(safe.index, errors="coerce")
    safe = safe[~safe.index.isna()]
    return safe


def normalize(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a price series into cumulative returns starting at 1."""
    if prices.empty:
        return prices
    returns = prices.pct_change().fillna(0)
    return (1 + returns).cumprod()


def plot(data: pd.DataFrame, title: str) -> plt.Figure:
    """Create a simple matplotlib line chart for cumulative returns."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(data.index, data.values)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (×)")
    ax.legend(data.columns, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig


def create_dashboard(
    title: str,
    ticker_map: Dict[str, str],
    default_selection: Iterable[str],
    key_prefix: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    sidebar_note: str | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Render the full workflow: data download, normalization, and plotting."""
    available = tuple(ticker_map.keys())
    start = start_date or DEFAULT_START_DATE
    end = end_date or DEFAULT_END_DATE

    sidebar = st.sidebar
    if sidebar_note:
        sidebar.caption(sidebar_note)

    selected = sidebar.multiselect(
        "Select assets",
        options=available,
        default=[s for s in default_selection if s in available] or list(available),
        format_func=lambda symbol: symbol,
        key=f"{key_prefix}_assets",
    )

    date_range = sidebar.date_input(
        "Date range",
        value=(start, end),
        min_value=DEFAULT_START_DATE,
        max_value=DEFAULT_END_DATE,
        key=f"{key_prefix}_dates",
    )

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.warning("Please select both a start and an end date.")
        return pd.DataFrame(), pd.DataFrame()

    start_value, end_value = sorted(date_range)
    if start_value == end_value:
        st.warning("Start and end date cannot be the same. Please widen the range.")
        return pd.DataFrame(), pd.DataFrame()

    if not selected:
        st.info("Select at least one asset from the sidebar to continue.")
        return pd.DataFrame(), pd.DataFrame()

    prices = download_data(tuple(selected), start_value, end_value, use_live=True)
    prices = preprocess(prices)

    if prices.empty:
        st.error("No data returned for the selected parameters. Adjust the filters and try again.")
        return pd.DataFrame(), pd.DataFrame()

    sidebar.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    normalized = normalize(prices)
    chart = plot(normalized, f"{title} — Normalized Performance")
    st.pyplot(chart)

    with st.expander("Normalized cumulative returns"):
        st.dataframe(normalized.style.format("{:.2f}"))

    return prices, normalized


def classify_regime(growth: float, inflation: float, vol: float) -> str:
    """Categorize the macro environment based on growth, inflation, and volatility proxies."""
    if growth > 0 and inflation < 0.03 and vol < 20:
        return "Expansion"
    if growth < 0 and inflation > 0.03:
        return "Stagflation"
    if growth < 0 and vol > 25:
        return "Recession"
    return "Recovery"


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def get_market_regime_data(
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Compute market regime labels from growth, inflation, and volatility proxies."""
    empty_frame = pd.DataFrame(
        columns=["Growth", "Inflation", "Volatility", "Regime"],
        index=pd.DatetimeIndex([], name="Date"),
    )
    start_date = start or DEFAULT_START_DATE
    end_date = end or DEFAULT_END_DATE
    tickers = ("^GSPC", "^VIX", "CL=F")
    prices = download_data(tickers, start_date, end_date, use_live=True)

    if prices.empty or not all(symbol in prices.columns for symbol in tickers):
        return empty_frame

    prices = preprocess(prices)

    sp_returns = prices["^GSPC"].pct_change()
    growth = sp_returns.rolling(90).mean()  # 90-day rolling average of daily returns
    inflation = prices["CL=F"].pct_change(252)  # 12-month price change proxy
    volatility = prices["^VIX"]

    regime_df = pd.DataFrame(
        {
            "Growth": growth,
            "Inflation": inflation,
            "Volatility": volatility,
        }
    ).dropna()

    if regime_df.empty:
        return empty_frame

    regime_df["Regime"] = regime_df.apply(
        lambda row: classify_regime(row["Growth"], row["Inflation"], row["Volatility"]),
        axis=1,
    )
    return regime_df


def filter_by_regime(
    data: pd.DataFrame,
    regime_df: pd.DataFrame,
    selected_regimes: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict a dataset to observations matching the chosen regime labels."""
    if regime_df.empty:
        return data, regime_df

    selected = set(selected_regimes) if selected_regimes else set(regime_df["Regime"].unique())
    filtered_regimes = regime_df[regime_df["Regime"].isin(selected)]
    if filtered_regimes.empty:
        return data.iloc[0:0], filtered_regimes

    filtered_index = data.index.intersection(filtered_regimes.index)
    return data.loc[filtered_index], filtered_regimes


def add_regime_shading(
    fig: go.Figure,
    regime_df: pd.DataFrame,
    *,
    layer: str = "below",
) -> go.Figure:
    """Overlay translucent shading for macro regimes on a Plotly time-series chart."""
    if regime_df.empty or "Regime" not in regime_df.columns:
        return fig

    regime_series = regime_df["Regime"].sort_index()
    start = None
    current = None
    prev_index = None

    for timestamp, regime in regime_series.items():
        if start is None:
            start = timestamp
            current = regime
            prev_index = timestamp
            continue

        if regime != current:
            fig.add_vrect(
                x0=start,
                x1=prev_index,
                fillcolor=REGIME_COLORS.get(current, "rgba(148,163,184,0.24)"),
                opacity=0.28,
                layer=layer,
                line_width=0,
            )
            start = timestamp
            current = regime
        prev_index = timestamp

    if start is not None and prev_index is not None:
        fig.add_vrect(
            x0=start,
            x1=prev_index,
            fillcolor=REGIME_COLORS.get(current, "rgba(148,163,184,0.24)"),
            opacity=0.28,
            layer=layer,
            line_width=0,
        )

    return fig


def render_regime_status_panel(regime_df: pd.DataFrame) -> None:
    """Display the current market regime headline."""
    st.subheader("Current Market Regime")
    if regime_df.empty or "Regime" not in regime_df.columns:
        st.info("Current market regime is unavailable for the selected range.")
        return

    regime_series = regime_df["Regime"].dropna()
    if regime_series.empty:
        st.info("Current market regime is unavailable for the selected range.")
        return

    current_regime = regime_series.iloc[-1]
    accent_color, accent_background = REGIME_ACCENTS.get(
        current_regime,
        ("#38bdf8", "rgba(56,189,248,0.14)"),
    )
    description = REGIME_DESCRIPTIONS.get(
        current_regime,
        "Market signals are mixed, and cross-asset leadership is rotating.",
    )

    columns = st.columns([1])
    with columns[0]:
        card_html = (
            "<div style=\"border:1px solid rgba(148,163,184,0.45);border-radius:0.9rem;"
            "padding:0.85rem;background:rgba(15,23,42,0.25);display:flex;flex-direction:column;"
            "gap:0.6rem;min-height:90px;width:100%;box-sizing:border-box;\">"
            f"<div style=\"display:flex;align-items:center;gap:0.6rem;flex-wrap:wrap;\">"
            f"<span style=\"font-size:1.1rem;font-weight:600;color:#f8fafc;letter-spacing:0.04em;\">{current_regime}</span>"
            f"<span style=\"padding:0.22rem 0.6rem;border-radius:999px;background:{accent_background};color:{accent_color};"
            "font-size:0.7rem;font-weight:640;letter-spacing:0.08em;text-transform:uppercase;\">Active</span>"
            "</div>"
            f"<p style=\"margin:0;font-size:0.9rem;line-height:1.4;color:#e2e8f0;\">{description}</p>"
            "</div>"
        )

        st.markdown(card_html, unsafe_allow_html=True)


def regime_options(regime_df: pd.DataFrame) -> list[str]:
    """Return available regime labels preserving insertion order."""
    if regime_df.empty or "Regime" not in regime_df.columns:
        return []
    return list(dict.fromkeys(regime_df["Regime"].tolist()))


def render_regime_legend(container: DeltaGenerator) -> None:
    """Display the color legend for market regimes inside the provided container."""
    if not REGIME_COLORS:
        return

    legend_rows = []
    for name, rgba in REGIME_COLORS.items():
        parts = rgba.rsplit(",", 1)
        base_color = parts[0] + ",0.65)" if len(parts) == 2 else rgba
        legend_rows.append(
            "<div style=\"display:flex;align-items:center;gap:0.55rem;margin-bottom:0.45rem;\">"
            f"<span style=\"display:inline-block;width:36px;height:4px;border-radius:2px;background:{base_color};"
            "border:1px solid rgba(255,255,255,0.35);\"></span>"
            f"<span style=\"font-size:0.85rem;color:#e2e8f0;font-weight:600;\">{name}</span>"
            "</div>"
        )

    legend_html = (
        "<div style=\"margin-top:0.5rem;padding:0.6rem 0.4rem 0.2rem;\">"
        "<div style=\"font-size:0.9rem;font-weight:600;margin-bottom:0.4rem;color:#f1f5f9;\">Market Regime Legend</div>"
        f"{''.join(legend_rows)}"
        "</div>"
    )
    container.markdown(legend_html, unsafe_allow_html=True)
