from __future__ import annotations

from datetime import date, timedelta
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
    get_market_regime_data,
    normalize,
    preprocess,
    render_regime_legend,
    render_regime_status_panel,
    render_social_links,
)

# Core watchlist tickers displayed in the sidebar selector.
STOCK_TICKERS: Dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet (Class A)",
    "NVDA": "NVIDIA",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "JPM": "JPMorgan Chase",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson",
}

# Palette aligned with the dashboard's orange-to-grey branding.
COLOR_PALETTE: Dict[str, str] = {
    "AAPL": "#FFB000",
    "MSFT": "#FF7C00",
    "AMZN": "#CC5500",
    "GOOGL": "#808080",
    "NVDA": "#A0A0A0",
    "META": "#B8B8B8",
    "TSLA": "#D0D0D0",
    "JPM": "#E0E0E0",
    "XOM": "#FFD580",
    "JNJ": "#FFFFFF",
}

# Custom heatmap gradient shared across tabs for brand alignment.
HEATMAP_COLOR_SCALE = [
    [0.0, "#4A4A4A"],
    [0.5, "#FFFFFF"],
    [1.0, "#FF962F"],
]

# Cache equity prices for 24 hours to limit repeated downloads each rerun.
@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_equity_prices(symbols: Tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """Download, cache, and preprocess adjusted close data for the requested symbols."""
    raw_prices = download_data(symbols, start, end, use_live=True)
    return preprocess(raw_prices)


def format_labels(symbols: Iterable[str], mapping: Dict[str, str]) -> Dict[str, str]:
    """Return a mapping that keeps tickers unchanged for display."""
    return {symbol: symbol for symbol in symbols}


def build_color_map(labels: Iterable[str]) -> Dict[str, str]:
    """Assign stable colors for each instrument label."""
    fallback_colors = px.colors.qualitative.Dark24
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

    # Drawdown compares the series to its running peak to gauge peak-to-trough loss.
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


def render_top_movers(metrics: pd.DataFrame, closing_prices: pd.Series) -> None:
    """Display the leading performers as a horizontal strip above the main chart."""
    if "Total Return (%)" not in metrics.columns:
        st.info("Total return data unavailable for movers view.")
        return

    total_returns = metrics["Total Return (%)"].dropna()
    if total_returns.empty:
        st.info("No performance history available to rank movers.")
        return

    movers_count = min(5, len(total_returns))
    top_movers = total_returns.nlargest(movers_count)

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
    columns = st.columns(movers_count, gap="medium")
    for column, (label, value) in zip(columns, top_movers.items()):
        column.markdown(build_entry(label, value), unsafe_allow_html=True)


def render_performance_chart(
    normalized: pd.DataFrame,
    color_map: Dict[str, str],
    regime_window: pd.DataFrame,
) -> None:
    """Render the top-level normalized performance chart."""
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
            "Global Stocks Dashboard<br>"
            "<span style=\"font-size:0.85em;font-weight:400;\">"
            "Normalized cumulative performance baseline to compare momentum across selected equities."
            "</span>"
        ),
        height=500,
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        yaxis_title="Normalized Cumulative Return (×)",
        xaxis_title="Date",
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    add_regime_shading(fig, regime_window)
    st.plotly_chart(fig, use_container_width=True)


def render_vol_drawdown_tab(
    rolling_vol: pd.DataFrame,
    drawdown: pd.DataFrame,
    color_map: Dict[str, str],
) -> None:
    """Display 30-day rolling volatility and drawdowns."""
    if not rolling_vol.empty:
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
            height=350,
            template="plotly_white",
            title=(
                "30-Day Rolling Volatility (Annualized)<br>"
                "<span style=\"font-size:0.85em;font-weight:400;\">"
                "Annualized σ computed from a 30-day rolling window of daily returns."
                "</span>"
            ),
            yaxis_title="Volatility (%)",
            xaxis_title="Date",
            hovermode="x unified",
        )
        st.plotly_chart(vol_fig, use_container_width=True)
    else:
        st.info("Rolling volatility requires at least 30 observations.")

    drawdown_fig = go.Figure()
    for column in drawdown.columns:
        drawdown_fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown[column] * 100,
                mode="lines",
                name=column,
                line=dict(width=2, color=color_map.get(column)),
                fill="tozeroy",
                hovertemplate="%{y:.1f}% drawdown<br>%{x|%Y-%m-%d}<extra>%{fullData.name}</extra>",
            )
        )
    drawdown_fig.update_layout(
        height=350,
        template="plotly_white",
        title=(
            "Drawdown Profile (From Peak)<br>"
            "<span style=\"font-size:0.85em;font-weight:400;\">"
            "Peak-to-trough declines, highlighting loss depth and recovery across stocks."
            "</span>"
        ),
        yaxis_title="Drawdown (%)",
        xaxis_title="Date",
        hovermode="x unified",
    )
    st.plotly_chart(drawdown_fig, use_container_width=True)


def render_correlation_tab(returns: pd.DataFrame) -> None:
    """Display correlation heatmap of daily returns."""
    if returns.empty or returns.shape[1] < 2:
        st.info("Correlation analysis requires at least two assets with return history.")
        return

    corr = returns.corr()
    heatmap = px.imshow(
        corr,
        color_continuous_scale=HEATMAP_COLOR_SCALE,
        zmin=-1,
        zmax=1,
        text_auto=".2f",
    )
    heatmap.update_layout(
        title=(
            "Correlation Heatmap (Daily Returns)<br>"
            "<span style=\"font-size:0.85em;font-weight:400;\">"
            "Matrix of pairwise daily return correlations to spot co-movement patterns."
            "</span>"
        ),
        template="plotly_white",
        xaxis_title="",
        yaxis_title="",
    )
    st.plotly_chart(heatmap, use_container_width=True)


def render() -> None:
    """Entry point for the equities tab."""
    try:
        st.set_page_config(
            page_title="Stocks | Cross-Asset Market Monitor",
            page_icon=":material/table_chart_view:",
            layout="wide",
        )
    except RuntimeError:
        # Streamlit raises if page config is set multiple times; safe to ignore within multi-page apps.
        pass

    st.header(":material/table_chart_view: Stocks")
    st.caption("Integrated stock analytics covering flagship single names.")

    default_start = max(DEFAULT_START_DATE, DEFAULT_END_DATE - timedelta(days=365))

    with st.sidebar:
        ticker_map = STOCK_TICKERS
        options = list(ticker_map.keys())
        default_selection = tuple(options[:4])
        selected_symbols = st.multiselect(
            "Select tickers",
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
        st.warning("Please provide both a start and end date.")
        return

    start_date, end_date = sorted(date_range)
    if start_date == end_date:
        st.warning("Date range must span at least two distinct dates.")
        return

    regime_universe = get_market_regime_data()
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
        st.info("Select at least one instrument from the sidebar to begin the analysis.")
        return

    prices = load_equity_prices(tuple(selected_symbols), start_date, end_date)
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

    regime_scope = regime_df.loc[regime_df.index.intersection(prices.index)] if not regime_df.empty else pd.DataFrame()
    regime_window = regime_scope if show_regime_overlay else pd.DataFrame()

    analytics = compute_metrics(prices)
    normalized = analytics["normalized"]
    metrics = analytics["metrics"].sort_values("Total Return (%)", ascending=False)

    color_map = build_color_map(normalized.columns)

    render_top_movers(metrics, prices.iloc[-1])
    render_performance_chart(normalized, color_map, regime_window)

    tab_risk, tab_correlation = st.tabs(["Volatility & Drawdown", "Correlation Heatmap"])

    with tab_risk:
        render_vol_drawdown_tab(analytics["rolling_vol"], analytics["drawdown"], color_map)

    with tab_correlation:
        render_correlation_tab(analytics["returns"])

    st.subheader("Performance & Risk Summary")
    overlay_state = "Enabled" if show_regime_overlay else "Hidden"
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
        file_name=f"equities_raw_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_equities_raw",
    )


if __name__ == "__main__":
    render()
