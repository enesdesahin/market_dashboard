from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from views.common import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    add_regime_shading,
    download_data,
    ensure_datetime_index,
    get_market_regime_data,
    normalize,
    preprocess,
    render_regime_legend,
    render_regime_status_panel,
    render_social_links,
)

# Sidebar universe: flagship commodity futures mapped to friendly names.
COMMODITY_TICKERS: Dict[str, str] = {
    "GC=F": "Gold Futures",
    "SI=F": "Silver Futures",
    "CL=F": "Crude Oil Futures",
    "BZ=F": "Brent Crude Futures",
    "NG=F": "Natural Gas Futures",
    "HG=F": "Copper Futures",
    "ZW=F": "Wheat Futures",
    "ZC=F": "Corn Futures",
}

# Consistent orange-to-grey palette reused across the dashboard.
COLOR_PALETTE: Dict[str, str] = {
    "GC=F": "#FFB000",
    "SI=F": "#FF7C00",
    "CL=F": "#CC5500",
    "BZ=F": "#808080",
    "NG=F": "#A0A0A0",
    "HG=F": "#B8B8B8",
    "ZW=F": "#D0D0D0",
    "ZC=F": "#FFD580",
}

# Shared heatmap gradient for correlation visuals.
HEATMAP_COLOR_SCALE = [
    [0.0, "#4A4A4A"],
    [0.5, "#FFFFFF"],
    [1.0, "#FF962F"],
]

# Persisted Yahoo Finance extracts live here; checked into git if you want seed data.
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
DATA_PATH = DATA_DIR / "commodities_prices.csv"


def _refresh_cached_dataset() -> pd.DataFrame:
    """Download the full commodity universe and persist it to disk."""
    dataset = download_data(
        tuple(COMMODITY_TICKERS.keys()),
        DEFAULT_START_DATE,
        DEFAULT_END_DATE,
        use_live=True,
    )
    dataset.to_csv(DATA_PATH)
    return dataset


def _load_full_dataset(use_live: bool) -> pd.DataFrame:
    """Load the cached dataset, refreshing when older than 24 hours or when requested."""
    if DATA_PATH.exists():
        if use_live:
            file_age = datetime.now() - datetime.fromtimestamp(DATA_PATH.stat().st_mtime)
            if file_age > timedelta(hours=24):
                return _refresh_cached_dataset()
        data = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
        data.index = data.index.tz_localize(None)
        return data
    return _refresh_cached_dataset() if use_live else pd.DataFrame()


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_commodity_prices(symbols: Tuple[str, ...], start: date, end: date, use_live: bool) -> pd.DataFrame:
    """Retrieve commodity prices for the selected universe and date range."""
    if not symbols:
        return pd.DataFrame()

    dataset = _load_full_dataset(use_live)
    if dataset.empty:
        return pd.DataFrame()
    subset = dataset.loc[
        (dataset.index >= pd.Timestamp(start)) & (dataset.index <= pd.Timestamp(end)),
        [s for s in symbols if s in dataset.columns],
    ]
    return preprocess(subset)


def format_labels(symbols: Iterable[str]) -> Dict[str, str]:
    """Return ticker labels for sidebar selection."""
    return {symbol: symbol for symbol in symbols}


def build_color_map(labels: Iterable[str]) -> Dict[str, str]:
    """Construct a color map keyed by friendly instrument names."""
    fallback_colors = px.colors.sequential.Sunset
    color_map: Dict[str, str] = {}
    for idx, label in enumerate(labels):
        color_map[label] = COLOR_PALETTE.get(label, fallback_colors[idx % len(fallback_colors)])
    return color_map


def compute_metrics(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Derive normalized prices, returns, risk metrics, and drawdowns."""
    normalized = normalize(prices)
    returns = prices.pct_change().dropna()

    metrics = pd.DataFrame(index=prices.columns)
    if not returns.empty:
        total_return = (prices.iloc[-1] / prices.iloc[0] - 1) * 100
        ann_return = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol.replace(0, np.nan)
        rolling_vol = returns.rolling(30).std() * np.sqrt(252) * 100
    else:
        total_return = pd.Series(data=np.nan, index=prices.columns)
        ann_return = pd.Series(data=np.nan, index=prices.columns)
        ann_vol = pd.Series(data=np.nan, index=prices.columns)
        sharpe = pd.Series(data=np.nan, index=prices.columns)
        rolling_vol = pd.DataFrame(columns=prices.columns, index=prices.index)

    drawdown = normalized.divide(normalized.cummax()).subtract(1)
    max_drawdown = drawdown.min() * 100

    metrics["Total Return (%)"] = total_return
    metrics["Annualized Return (%)"] = ann_return * 100
    metrics["Annualized Volatility (%)"] = ann_vol * 100
    metrics["Sharpe Ratio"] = sharpe
    metrics["Max Drawdown (%)"] = max_drawdown

    return {
        "normalized": normalized,
        "returns": returns,
        "metrics": metrics,
        "rolling_vol": rolling_vol,
        "drawdown": drawdown,
    }


def render_top_gainers(metrics: pd.DataFrame, closing_prices: pd.Series) -> None:
    """Display top-performing commodities in card format."""
    if "Total Return (%)" not in metrics.columns:
        st.info("Total return data unavailable for gainers view.")
        return

    total_returns = metrics["Total Return (%)"].dropna()
    if total_returns.empty:
        st.info("No performance history available to rank gainers.")
        return

    count = min(5, len(total_returns))
    top_movers = total_returns.nlargest(count)

    def build_entry(label: str, value: float) -> str:
        is_positive = value >= 0
        arrow = "&#9650;" if is_positive else "&#9660;"
        color = "#16a34a" if is_positive else "#dc2626"
        last_price = closing_prices.get(label, np.nan)
        price_text = f"${last_price:,.2f}" if pd.notna(last_price) else "N/A"
        delta_text = f"{value:+.1f}%"
        return (
            "<div style=\"border:1px solid rgba(148,163,184,0.45);border-radius:0.9rem;"
            "padding:0.85rem;background:rgba(15,23,42,0.25);min-height:105px;display:flex;"
            "flex-direction:column;justify-content:center;gap:0.6rem;\">"
            f"<div style=\"font-size:1.1rem;font-weight:600;color:#f8fafc;\">{label}</div>"
            "<div style=\"display:flex;align-items:flex-end;gap:0.75rem;\">"
            f"<span style=\"font-size:1.8rem;font-weight:600;color:#e2e8f0;line-height:1;\">{price_text}</span>"
            f"<span style=\"font-size:0.9rem;font-weight:600;color:{color};line-height:1;align-self:flex-end;\">{arrow} {delta_text}</span>"
            "</div>"
            "</div>"
        )

    st.subheader("Top Gainers")
    columns = st.columns(count, gap="medium")
    for column, (label, value) in zip(columns, top_movers.items()):
        column.markdown(build_entry(label, value), unsafe_allow_html=True)


def render_performance_chart(
    normalized: pd.DataFrame,
    color_map: Dict[str, str],
    regime_window: pd.DataFrame,
) -> None:
    """Render the primary normalized performance chart."""
    fig = go.Figure()
    for column in normalized.columns:
        fig.add_trace(
            go.Scatter(
                x=normalized.index,
                y=normalized[column],
                mode="lines",
                name=column,
                line=dict(width=2, color=color_map.get(column)),
                hovertemplate="%{y:.2f}x<br>%{x|%Y-%m-%d}<extra>%{fullData.name}</extra>",
            )
        )

    fig.update_layout(
        title=(
            "Global Commodities Dashboard<br>"
            "<span style=\"font-size:0.85em;font-weight:400;\">"
            "Normalized cumulative performance for key commodities to gauge cross-sector momentum."
            "</span>"
        ),
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis_title="Date",
        yaxis_title="Normalized Cumulative Return (×)",
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    add_regime_shading(fig, regime_window)
    st.plotly_chart(fig, use_container_width=True)


def render_volatility_drawdown_tab(
    rolling_vol: pd.DataFrame,
    drawdown: pd.DataFrame,
    color_map: Dict[str, str],
) -> None:
    """Render the volatility and drawdown analytics."""
    if rolling_vol.empty:
        st.info("Rolling volatility requires at least 30 observations.")
    else:
        vol_fig = go.Figure()
        for column in rolling_vol.columns:
            vol_fig.add_trace(
                go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol[column],
                    mode="lines",
                    name=column,
                    line=dict(width=2, color=color_map.get(column)),
                    hovertemplate="%{y:.1f}%<br>%{x|%Y-%m-%d}<extra>%{fullData.name}</extra>",
                )
            )
        vol_fig.update_layout(
            template="plotly_white",
            height=350,
            title=(
                "30-Day Rolling Volatility (Annualized)<br>"
                "<span style=\"font-size:0.85em;font-weight:400;\">"
                "Annualized σ computed from a 30-day rolling window of commodity returns."
                "</span>"
            ),
            xaxis_title="Date",
            yaxis_title="Volatility (%)",
            hovermode="x unified",
        )
        st.plotly_chart(vol_fig, use_container_width=True)

    dd_fig = go.Figure()
    for column in drawdown.columns:
        dd_fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown[column] * 100,
                mode="lines",
                name=column,
                line=dict(width=2, color=color_map.get(column)),
                fill="tozeroy",
                hovertemplate="%{y:.1f}%<br>%{x|%Y-%m-%d}<extra>%{fullData.name}</extra>",
            )
        )
    dd_fig.update_layout(
        template="plotly_white",
        height=350,
        title=(
            "Drawdown Profile (Peak-to-Trough)<br>"
            "<span style=\"font-size:0.85em;font-weight:400;\">"
            "Peak-to-trough drawdowns showing losses and recoveries across commodity markets."
            "</span>"
        ),
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode="x unified",
    )
    st.plotly_chart(dd_fig, use_container_width=True)


def render_correlation_tab(returns: pd.DataFrame) -> None:
    """Display the correlation heatmap for daily returns."""
    if returns.empty or returns.shape[1] < 2:
        st.info("Correlation analysis requires at least two commodities with return history.")
        return

    corr = returns.corr()
    heatmap = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=HEATMAP_COLOR_SCALE,
        zmin=-1,
        zmax=1,
    )
    heatmap.update_layout(
        template="plotly_white",
        title=(
            "Correlation Heatmap (Daily Returns)<br>"
            "<span style=\"font-size:0.85em;font-weight:400;\">"
            "Correlations across commodity returns to spot shared supply-demand drivers."
            "</span>"
        ),
        xaxis_title="",
        yaxis_title="",
        height=500,
    )
    st.plotly_chart(heatmap, use_container_width=True)


def render() -> None:
    """Entry point for the commodities tab."""
    try:
        st.set_page_config(
            page_title="Commodities | Cross-Asset Market Monitor",
            page_icon=":material/oil_barrel:",
            layout="wide",
        )
    except RuntimeError:
        # set_page_config can only be called once per app launch; ignore subsequent calls.
        pass

    st.header(":material/oil_barrel: Commodities")
    st.caption(
        "Interactive analytics covering global commodities across metals, energy, and agricultural markets."
    )

    default_start = max(DEFAULT_START_DATE, DEFAULT_END_DATE - timedelta(days=365))

    with st.sidebar:
        options = list(COMMODITY_TICKERS.keys())
        default_selection = tuple(options[:4])
        selected_symbols = st.multiselect(
            "Select commodities",
            options=options,
            default=default_selection,
        )

        date_range = st.date_input(
            "Date range",
            value=(default_start, DEFAULT_END_DATE),
            min_value=DEFAULT_START_DATE,
            max_value=DEFAULT_END_DATE,
        )

        show_regime_overlay = st.toggle(
            "Display Market Regime",
            value=False,
            help="Toggle macro regime shading and the current regime status banner.",
        )

    if not isinstance(date_range, (tuple, list)) or len(date_range) != 2:
        st.warning("Please provide both a start and an end date.")
        return

    start_date, end_date = sorted(date_range)
    if start_date == end_date:
        st.warning("Date range must span at least two distinct dates.")
        return

    regime_universe = ensure_datetime_index(get_market_regime_data())
    regime_df = regime_universe.loc[
        (regime_universe.index >= pd.Timestamp(start_date))
        & (regime_universe.index <= pd.Timestamp(end_date))
    ]
    if show_regime_overlay and regime_df.empty:
        st.sidebar.info("Market regime data unavailable for the selected date range.")
        show_regime_overlay = False
    if show_regime_overlay:
        render_regime_legend(st.sidebar)

    if not selected_symbols:
        st.info("Select at least one commodity from the sidebar to begin the analysis.")
        return

    prices = load_commodity_prices(tuple(selected_symbols), start_date, end_date, use_live=True)
    if prices.empty:
        st.error("No data returned for the selected parameters. Adjust the filters and try again.")
        return

    st.sidebar.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    render_social_links()

    if show_regime_overlay:
        render_regime_status_panel(regime_universe)

    available_symbols = [symbol for symbol in selected_symbols if symbol in prices.columns]
    missing_symbols = sorted(set(selected_symbols) - set(available_symbols))
    if missing_symbols:
        st.warning(
            "Data unavailable for: {}. These instruments were excluded from the analysis.".format(
                ", ".join(missing_symbols)
            )
        )

    if not available_symbols:
        st.error("All selected instruments returned empty data. Please adjust your selection.")
        return

    prices = prices[available_symbols]
    labels = available_symbols
    prices.columns = labels

    regime_scope = regime_df.loc[regime_df.index.intersection(prices.index)] if not regime_df.empty else pd.DataFrame()
    regime_window = regime_scope if show_regime_overlay else pd.DataFrame()

    analytics = compute_metrics(prices)
    normalized = analytics["normalized"]
    metrics = analytics["metrics"].sort_values("Total Return (%)", ascending=False)

    color_map = build_color_map(labels)

    render_top_gainers(metrics, prices.iloc[-1])
    render_performance_chart(normalized, color_map, regime_window)

    tab_risk, tab_correlation = st.tabs(["Volatility & Drawdown", "Correlation Heatmap"])

    with tab_risk:
        render_volatility_drawdown_tab(
            analytics["rolling_vol"],
            analytics["drawdown"],
            color_map,
        )

    with tab_correlation:
        render_correlation_tab(analytics["returns"])

    st.subheader("Performance & Risk Summary")
    st.dataframe(
        metrics.style.format(
            {
                "Total Return (%)": "{:.1f}",
                "Annualized Return (%)": "{:.1f}",
                "Annualized Volatility (%)": "{:.1f}",
                "Sharpe Ratio": "{:.2f}",
                "Max Drawdown (%)": "{:.1f}",
            }
        )
    )

    st.subheader("Raw Data")
    export_frame = prices.copy()
    st.dataframe(export_frame.style.format("{:.2f}"))

    csv_bytes = export_frame.to_csv(index_label="Date").encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"commodities_raw_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_commodities_raw",
    )


if __name__ == "__main__":
    render()
