from __future__ import annotations

import io
import itertools
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from plotly.colors import hex_to_rgb
import streamlit as st

from views.common import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    add_regime_shading,
    get_market_regime_data,
    preprocess,
    render_regime_legend,
    render_regime_status_panel,
    render_social_links,
)

# FRED identifiers for the U.S. Treasury curve (tenors + display labels).
TREASURY_SERIES = [
    ("US1M", "U.S. Treasury 1M", "DGS1MO"),
    ("US3M", "U.S. Treasury 3M", "DGS3MO"),
    ("US6M", "U.S. Treasury 6M", "DGS6MO"),
    ("US1Y", "U.S. Treasury 1Y", "DGS1"),
    ("US2Y", "U.S. Treasury 2Y", "DGS2"),
    ("US3Y", "U.S. Treasury 3Y", "DGS3"),
    ("US5Y", "U.S. Treasury 5Y", "DGS5"),
    ("US7Y", "U.S. Treasury 7Y", "DGS7"),
    ("US10Y", "U.S. Treasury 10Y", "DGS10"),
    ("US20Y", "U.S. Treasury 20Y", "DGS20"),
    ("US30Y", "U.S. Treasury 30Y", "DGS30"),
]

# OECD long-end series used to benchmark against international peers.
OECD_10Y_SERIES = [
    ("US10Y_OECD", "United States 10Y (OECD)", "IRLTLT01USM156N"),
    ("DE10Y", "Germany Bund 10Y", "IRLTLT01DEM156N"),
    ("FR10Y", "France Gov 10Y", "IRLTLT01FRM156N"),
    ("IT10Y", "Italy Gov 10Y", "IRLTLT01ITM156N"),
    ("GB10Y", "United Kingdom 10Y", "IRLTLT01GBM156N"),
    ("JP10Y", "Japan Gov 10Y", "IRLTLT01JPM156N"),
    ("ES10Y", "Spain Gov 10Y", "IRLTLT01ESM156N"),
    ("PT10Y", "Portugal Gov 10Y", "IRLTLT01PTM156N"),
    ("GR10Y", "Greece Gov 10Y", "IRLTLT01GRM156N"),
]


def _series_map(entries: Iterable[Tuple[str, str, str]]) -> Dict[str, Dict[str, str]]:
    return {symbol: {"label": label, "source": source} for symbol, label, source in entries}


# Unified lookup of every supported bond instrument.
BOND_SERIES: Dict[str, Dict[str, str]] = {
    **_series_map(TREASURY_SERIES),
    **_series_map(OECD_10Y_SERIES),
}

# Convenience tuple used when we need the full universe regardless of sidebar choices.
ALL_YIELD_SYMBOLS = tuple(symbol for symbol, _, _ in TREASURY_SERIES + OECD_10Y_SERIES)

# Explicit color overrides so Treasuries/OECD names stick to the house palette.
COLOR_OVERRIDES: Dict[str, str] = {
    "US1M": "#FFB000",
    "US3M": "#FF7C00",
    "US6M": "#CC5500",
    "US1Y": "#808080",
    "US2Y": "#A0A0A0",
    "US3Y": "#B8B8B8",
    "US5Y": "#D0D0D0",
    "US7Y": "#E0E0E0",
    "US10Y": "#FFD580",
    "US20Y": "#FFB000",
    "US30Y": "#FF7C00",
    "US10Y_OECD": "#CC5500",
    "DE10Y": "#808080",
    "FR10Y": "#A0A0A0",
    "IT10Y": "#B8B8B8",
    "GB10Y": "#D0D0D0",
    "JP10Y": "#E0E0E0",
    "ES10Y": "#FFD580",
    "PT10Y": "#FFB000",
    "GR10Y": "#FF7C00",
}

DEFAULT_COLOR_SEQUENCE = px.colors.qualitative.Plotly

# Shared grey→orange correlation gradient for heatmaps.
HEATMAP_COLOR_SCALE = [
    [0.0, "#4A4A4A"],
    [0.5, "#FFFFFF"],
    [1.0, "#FF962F"],
]

# Custom gradient for the 3D volatility surface (avoids pure white mid-tones).
VOL_SURFACE_COLOR_SCALE = [
    [0.0, "#4A4A4A"],
    [0.25, "#775D43"],
    [0.5, "#A4703C"],
    [0.75, "#D28336"],
    [1.0, "#FF962F"],
]

DEFAULT_SELECTION = ("US2Y", "US5Y", "US10Y", "US30Y")

# On-disk cache for individual FRED downloads.
YIELD_CACHE_DIR = Path("data_cache") / "yields"
YIELD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TREASURY_MATURITIES = [
    ("US1M", 1 / 12),
    ("US3M", 0.25),
    ("US6M", 0.5),
    ("US1Y", 1),
    ("US2Y", 2),
    ("US3Y", 3),
    ("US5Y", 5),
    ("US7Y", 7),
    ("US10Y", 10),
    ("US20Y", 20),
    ("US30Y", 30),
]

YIELD_CURVE_POINTS = {
    "United States": TREASURY_MATURITIES,
}


def _yield_cache_path(symbol: str) -> Path:
    return YIELD_CACHE_DIR / f"{symbol}.csv"


def _rgb_to_hex(rgb: Iterable[int]) -> str:
    r, g, b = rgb
    return "#{:02X}{:02X}{:02X}".format(r, g, b)


def _blend_with_white(base_rgb: Tuple[int, int, int], weight: float) -> Tuple[int, int, int]:
    """Blend a base RGB color toward white using the provided weight."""
    return tuple(int(round(channel + (255 - channel) * weight)) for channel in base_rgb)


def _orange_gradient(count: int) -> list[str]:
    """Return a smooth gradient based on the FF962F orange base color."""
    if count <= 0:
        return []
    base_rgb = tuple(hex_to_rgb("FF962F"))
    if count == 1:
        return [_rgb_to_hex(base_rgb)]
    weights = np.linspace(0.6, 0.0, count)
    return [_rgb_to_hex(_blend_with_white(base_rgb, float(weight))) for weight in weights]


def _prepare_series(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalize a data frame to a single numeric column labelled by symbol."""
    if frame.empty:
        return frame

    if symbol in frame.columns:
        series = frame[[symbol]]
    else:
        numeric_cols = [
            col for col in frame.columns if pd.api.types.is_numeric_dtype(frame[col])
        ]
        if not numeric_cols:
            return pd.DataFrame()
        series = frame[[numeric_cols[0]]].rename(columns={numeric_cols[0]: symbol})

    series[symbol] = pd.to_numeric(series[symbol], errors="coerce")
    return series.dropna(how="all")


def _download_fred_series(symbol: str, source: str, start: date, end: date) -> pd.DataFrame:
    """Fetch a FRED series using the public CSV endpoint."""
    params = {"id": source}
    if start:
        params["cosd"] = start.strftime("%Y-%m-%d")
    if end:
        params["coed"] = end.strftime("%Y-%m-%d")

    response = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv", params=params, timeout=20)
    response.raise_for_status()
    buffer = io.StringIO(response.text)
    frame = pd.read_csv(buffer)
    if frame.empty:
        return pd.DataFrame()
    date_column = None
    for candidate in ("observation_date", "date", "DATE"):
        if candidate in frame.columns:
            date_column = candidate
            break
    if not date_column:
        return pd.DataFrame()

    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column])

    value_column = source if source in frame.columns else None
    if value_column is None:
        numeric_cols = [
            col for col in frame.columns if col != date_column and pd.api.types.is_numeric_dtype(frame[col])
        ]
        if not numeric_cols:
            return pd.DataFrame()
        value_column = numeric_cols[0]

    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce")
    frame = frame.set_index(date_column).sort_index()
    frame = frame.rename(columns={value_column: symbol})
    return frame[[symbol]]


def _load_series(symbol: str, source: str, start: date, end: date, use_live: bool) -> pd.DataFrame:
    """Load a yield series with optional live refresh."""
    cache_path = _yield_cache_path(symbol)
    dataset = pd.DataFrame()

    if use_live:
        try:
            buffer_start = start - timedelta(days=365)
            dataset = _download_fred_series(symbol, source, buffer_start, end)
            if not dataset.empty:
                dataset.to_csv(cache_path)
        except Exception:
            dataset = pd.DataFrame()

    if dataset.empty and cache_path.exists():
        dataset = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        dataset.index = dataset.index.tz_localize(None)

    return _prepare_series(dataset, symbol)


@st.cache_data(ttl=12 * 60 * 60, show_spinner=False)
def load_bond_prices(symbols: Tuple[str, ...], start: date, end: date, use_live: bool) -> pd.DataFrame:
    """Retrieve sovereign yield series for the selected universe and date range."""
    if not symbols:
        return pd.DataFrame()

    frames = []
    for symbol in symbols:
        config = BOND_SERIES.get(symbol)
        if not config:
            continue
        series = _load_series(symbol, config["source"], start, end, use_live)
        if series.empty:
            continue
        frames.append(series)

    if not frames:
        return pd.DataFrame()

    dataset = pd.concat(frames, axis=1).sort_index()
    mask = (dataset.index >= pd.Timestamp(start)) & (dataset.index <= pd.Timestamp(end))
    subset = dataset.loc[mask]
    return preprocess(subset)


def format_labels(symbols: Iterable[str]) -> Dict[str, str]:
    """Return ticker labels for sidebar selection."""
    return {symbol: BOND_SERIES.get(symbol, {}).get("label", symbol) for symbol in symbols}


def build_color_map(labels: Iterable[str]) -> Dict[str, str]:
    """Assign stable colors across bond categories."""
    palette = itertools.cycle(DEFAULT_COLOR_SEQUENCE)
    return {label: COLOR_OVERRIDES.get(label, next(palette)) for label in labels}


def compute_metrics(prices: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Derive level, change, and curve diagnostics for yields."""
    if prices.empty:
        return {
            "levels": prices,
            "daily_change": prices,
            "metrics": pd.DataFrame(),
            "rolling_change_vol": prices,
            "drawdown": prices,
            "spreads": pd.DataFrame(),
        }

    # Daily changes in basis points
    daily_change = prices.diff() * 100  # bp
    weekly_change = prices.diff(5) * 100
    monthly_change = prices.diff(21) * 100
    rolling_change_vol = daily_change.rolling(10).std()

    metrics = pd.DataFrame(index=prices.columns)
    latest = prices.iloc[-1]
    metrics["Current Yield (%)"] = latest
    metrics["1D Change (bp)"] = daily_change.iloc[-1]
    metrics["5D Change (bp)"] = weekly_change.iloc[-1]
    metrics["20D Change (bp)"] = monthly_change.iloc[-1]
    metrics["10D Vol (bp)"] = rolling_change_vol.iloc[-1]

    trailing_window = prices.tail(63)
    if not trailing_window.empty:
        range_bp = (trailing_window.max() - trailing_window.min()) * 100
        metrics["63D Range (bp)"] = range_bp
    else:
        metrics["63D Range (bp)"] = np.nan

    # Curve diagnostics
    spread_specs = [
        ("US 10s-2s (bp)", "US10Y", "US2Y"),
        ("US 30s-5s (bp)", "US30Y", "US5Y"),
        ("US-DE 10Y (bp)", "US10Y", "DE10Y"),
    ]
    spread_data: Dict[str, pd.Series] = {}
    for label, long_leg, short_leg in spread_specs:
        if {long_leg, short_leg}.issubset(prices.columns):
            spread_data[label] = (prices[long_leg] - prices[short_leg]) * 100
    spreads = pd.DataFrame(spread_data) if spread_data else pd.DataFrame(index=prices.index)

    # Retain placeholders for compatibility
    return {
        "levels": prices,
        "daily_change": daily_change,
        "metrics": metrics,
        "rolling_change_vol": rolling_change_vol,
        "drawdown": daily_change,  # unused but kept for interface consistency
        "spreads": spreads,
    }


def render_top_gainers(
    metrics: pd.DataFrame,
    closing_prices: pd.Series,
    *,
    as_of: pd.Timestamp | None = None,
) -> None:
    """Display top-performing bonds using Streamlit metrics."""
    change_field = "1D Change (bp)"
    if change_field not in metrics.columns:
        st.info("Change data unavailable for movers view.")
        return

    daily_changes = metrics[change_field].dropna()
    if daily_changes.empty:
        st.info("No recent yield changes available to rank movers.")
        return

    count = min(5, len(daily_changes))
    top_movers = daily_changes.nlargest(count)
    label_map = format_labels(top_movers.index)

    st.subheader("Top Yield Movers")
    columns = st.columns(count, gap="medium")
    for column, (symbol, change) in zip(columns, top_movers.items()):
        display_name = label_map.get(symbol, symbol)
        last_price = closing_prices.get(symbol, np.nan)
        value_text = f"{last_price:.2f}%" if pd.notna(last_price) else "N/A"
        delta_text = f"{change:+.1f} bp"
        delta_color = "#16a34a" if change >= 0 else "#dc2626"
        arrow = "&#9650;" if change >= 0 else "&#9660;"
        column.markdown(
            (
                "<div style=\"border:1px solid rgba(148,163,184,0.45);border-radius:0.9rem;"
                "padding:0.85rem;background:rgba(15,23,42,0.25);min-height:105px;display:flex;"
                "flex-direction:column;justify-content:center;gap:0.6rem;\">"
                f"<div style=\"font-size:1.1rem;font-weight:600;color:#f8fafc;\">{display_name}</div>"
                "<div style=\"display:flex;align-items:flex-end;gap:0.75rem;\">"
                f"<span style=\"font-size:1.8rem;font-weight:600;color:#e2e8f0;line-height:1;\">{value_text}</span>"
                f"<span style=\"font-size:0.9rem;font-weight:600;color:{delta_color};line-height:1;align-self:flex-end;\">{arrow} {delta_text}</span>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_performance_chart(
    levels: pd.DataFrame,
    color_map: Dict[str, str],
    regime_window: pd.DataFrame,
    label_map: Dict[str, str],
) -> None:
    """Render the primary yield level chart."""
    fig = go.Figure()
    for column in levels.columns:
        display_name = label_map.get(column, column)
        fig.add_trace(
            go.Scatter(
                x=levels.index,
                y=levels[column],
                mode="lines",
                name=display_name,
                line=dict(width=2, color=color_map.get(column)),
                hovertemplate=f"%{{y:.2f}}%<br>%{{x|%Y-%m-%d}}<extra>{display_name}</extra>",
            )
        )

    fig.update_layout(
        title=(
            "Sovereign Yield Levels<br>"
            "<span style=\"font-size:0.85em;font-weight:400;\">"
            "Tracks absolute sovereign yields to compare level shifts across selected markets."
            "</span>"
        ),
        template="plotly_white",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        xaxis_title="Date",
        yaxis_title="Yield (%)",
        hovermode="x unified",
        margin=dict(t=60, b=40),
    )
    add_regime_shading(fig, regime_window)
    st.plotly_chart(fig, use_container_width=True)


def render_volatility_drawdown_tab(
    daily_change: pd.DataFrame,
    color_map: Dict[str, str],
    label_map: Dict[str, str],
    spreads: pd.DataFrame,
) -> None:
    """Render yield change and curve spread diagnostics."""
    changes = daily_change.dropna(how="all")
    if changes.empty:
        st.info("Yield change data requires at least two observations.")
    else:
        vol_fig = go.Figure()
        for column in changes.columns:
            display_name = label_map.get(column, column)
            vol_fig.add_trace(
                go.Scatter(
                    x=changes.index,
                    y=changes[column],
                    mode="lines",
                    name=display_name,
                    line=dict(width=2, color=color_map.get(column)),
                    hovertemplate=f"%{{y:.1f}} bp<br>%{{x|%Y-%m-%d}}<extra>{display_name}</extra>",
                )
            )
        vol_fig.update_layout(
            template="plotly_white",
            height=350,
            title=(
                "Daily Yield Change (bp)<br>"
                "<span style=\"font-size:0.85em;font-weight:400;\">"
                "Shows day-over-day moves in yields expressed in basis points for each maturity."
                "</span>"
            ),
            xaxis_title="Date",
            yaxis_title="Change (bp)",
            hovermode="x unified",
        )
        st.plotly_chart(vol_fig, use_container_width=True)

    spread_view = spreads.dropna(how="all")
    if spread_view.empty:
        st.info("Insufficient data to compute curve spreads.")
    else:
        dd_fig = go.Figure()
        colors = _orange_gradient(len(spread_view.columns))
        if not colors:
            palette_cycle = itertools.cycle(DEFAULT_COLOR_SEQUENCE)
            colors = [next(palette_cycle) for _ in spread_view.columns]
        for column, line_color in zip(spread_view.columns, colors):
            dd_fig.add_trace(
                go.Scatter(
                    x=spread_view.index,
                    y=spread_view[column],
                    mode="lines",
                    name=column,
                    line=dict(width=2, color=line_color),
                    hovertemplate="%{y:.1f} bp<br>%{x|%Y-%m-%d}<extra>%{fullData.name}</extra>",
                )
            )
        dd_fig.update_layout(
            template="plotly_white",
            height=350,
            title=(
                "Curve & Cross-Market Spreads (bp)<br>"
                "<span style=\"font-size:0.85em;font-weight:400;\">"
                "Monitors slope and cross-country differentials to spot curve inversions and spread shifts."
                "</span>"
            ),
            xaxis_title="Date",
            yaxis_title="Spread (bp)",
            hovermode="x unified",
        )
        st.plotly_chart(dd_fig, use_container_width=True)

    us_curve_points = YIELD_CURVE_POINTS.get("United States", ())
    ordered_symbols = [symbol for symbol, _ in us_curve_points if symbol in changes.columns]
    if len(ordered_symbols) >= 2:
        vol_surface = changes[ordered_symbols].rolling(21).std().dropna(how="all")
        if not vol_surface.empty:
            vol_surface = vol_surface.iloc[-180:]
            maturities = [maturity for symbol, maturity in us_curve_points if symbol in ordered_symbols]
            date_axis = [ts.strftime("%Y-%m-%d") for ts in vol_surface.index]
            surface_fig = go.Figure(
                data=[
                    go.Surface(
                        x=maturities,
                        y=date_axis,
                        z=vol_surface[ordered_symbols].values,
                        colorscale=VOL_SURFACE_COLOR_SCALE,
                        colorbar=dict(title="σ (bp)"),
                    )
                ]
            )
            surface_fig.update_layout(
                title=(
                    "U.S. Treasury Yield Volatility Surface (21D σ)<br>"
                    "<span style=\"font-size:0.85em;font-weight:400;\">"
                    "21-day rolling volatility of daily yield changes across the Treasury curve."
                    "</span>"
                ),
                scene=dict(
                    xaxis=dict(
                        title="Maturity (Years)",
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.15)",
                    ),
                    yaxis=dict(
                        title="Date",
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.15)",
                    ),
                    zaxis=dict(
                        title="Volatility (bp)",
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.15)",
                    ),
                ),
                height=520,
            )
            st.plotly_chart(surface_fig, use_container_width=True)


def render_correlation_tab(daily_change: pd.DataFrame, label_map: Dict[str, str]) -> None:
    """Display the correlation heatmap for daily yield changes."""
    if daily_change.empty or daily_change.shape[1] < 2:
        st.info("Correlation analysis requires at least two series with recent yield changes.")
        return

    renamed = daily_change.rename(columns=label_map)
    corr = renamed.corr()
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
            "Pairwise correlations of daily yield changes to highlight synchronized moves."
            "</span>"
        ),
        xaxis_title="",
        yaxis_title="",
        height=500,
    )
    st.plotly_chart(heatmap, use_container_width=True)


def render_yield_curve_tab(
    yield_data: pd.DataFrame,
    label_map: Dict[str, str],
    color_map: Dict[str, str],
) -> None:
    """Display OECD vs U.S. yield snapshots and their time series."""
    if yield_data.empty:
        st.info("Yield data unavailable for the selected date range.")
        return

    filled = yield_data.ffill()
    latest_valid = filled.dropna(how="all")
    if latest_valid.empty:
        st.info("Yield data unavailable for the selected date range.")
        return

    snapshot_date = latest_valid.index[-1]
    snapshot = filled.loc[snapshot_date]

    oecd_symbols = [symbol for symbol, _, _ in OECD_10Y_SERIES if symbol in snapshot.index]

    top_left, top_right = st.columns(2)

    def _bar_colors(count: int) -> list[str]:
        gradient = _orange_gradient(count)
        if gradient:
            return gradient
        palette = itertools.cycle(DEFAULT_COLOR_SEQUENCE)
        return [next(palette) for _ in range(count)]

    with top_left:
        data = snapshot[oecd_symbols].dropna().sort_values()
        if data.empty:
            st.info("OECD 10Y yield snapshot unavailable.")
        else:
            labels = [label_map.get(symbol, symbol) for symbol in data.index]
            bar_colors = _bar_colors(len(data))
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=labels,
                        y=[float(value) for value in data.values],
                        marker=dict(color=bar_colors),
                        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                    )
                ]
            )
            fig.update_layout(
                template="plotly_white",
                title=(
                    f"OECD 10-Year Government Bond Yields ({snapshot_date.strftime('%Y-%m-%d')})<br>"
                    "<span style=\"font-size:0.85em;font-weight:400;\">"
                    "Latest OECD 10-year benchmarks, sorted by yield for quick cross-country comparison."
                    "</span>"
                ),
                xaxis_title="Country",
                yaxis_title="Yield (%)",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

    with top_right:
        treasury_points = []
        for symbol, maturity in TREASURY_MATURITIES:
            if symbol not in snapshot.index:
                continue
            value = snapshot.get(symbol)
            if pd.isna(value):
                continue
            treasury_points.append((maturity, symbol, float(value)))

        if not treasury_points:
            st.info("U.S. Treasury yield curve snapshot unavailable.")
        else:
            treasury_points.sort(key=lambda item: item[2])
            maturities, symbols, yields = zip(*treasury_points)
            labels = [label_map.get(symbol, symbol) for symbol in symbols]

            def _format_maturity(value: float) -> str:
                if value < 1:
                    months = round(value * 12)
                    return f"{months}M"
                if float(value).is_integer():
                    return f"{int(value)}Y"
                return f"{value:.1f}Y"

            maturity_text = [_format_maturity(maturity) for maturity in maturities]
            bar_colors = _bar_colors(len(symbols))
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=labels,
                        y=yields,
                        customdata=np.array(maturity_text),
                        marker=dict(color=bar_colors),
                        hovertemplate="Maturity: %{customdata}<br>Yield: %{y:.2f}%<extra></extra>",
                    )
                ]
            )
            fig.update_layout(
                template="plotly_white",
                title=(
                    f"U.S. Treasury Yield Curve Snapshot ({snapshot_date.strftime('%Y-%m-%d')})<br>"
                    "<span style=\"font-size:0.85em;font-weight:400;\">"
                    "Displays the current Treasury curve by maturity using the latest available yields."
                    "</span>"
                ),
                xaxis_title="Maturity (Years)",
                yaxis_title="Yield (%)",
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

    bottom_left, bottom_right = st.columns(2)

    oecd_history_symbols = [symbol for symbol, _, _ in OECD_10Y_SERIES if symbol in yield_data.columns]
    treasury_history_symbols = [symbol for symbol, _, _ in TREASURY_SERIES if symbol in yield_data.columns]

    with bottom_left:
        history = yield_data[oecd_history_symbols].dropna(how="all")
        if history.empty:
            st.info("OECD 10Y time series unavailable.")
        else:
            fig = go.Figure()
            for symbol in history.columns:
                display_name = label_map.get(symbol, symbol)
                fig.add_trace(
                    go.Scatter(
                        x=history.index,
                        y=history[symbol],
                        mode="lines",
                        name=display_name,
                        line=dict(color=color_map.get(symbol)),
                        hovertemplate=f"%{{y:.2f}}%<br>%{{x|%Y-%m-%d}}<extra>{display_name}</extra>",
                    )
                )
            fig.update_layout(
                template="plotly_white",
                title=(
                    "OECD 10Y Yields Over Time<br>"
                    "<span style=\"font-size:0.85em;font-weight:400;\">"
                    "Historical path of OECD 10-year yields to assess medium-term trends."
                    "</span>"
                ),
                xaxis_title="Date",
                yaxis_title="Yield (%)",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

    with bottom_right:
        history = yield_data[treasury_history_symbols].dropna(how="all")
        if history.empty:
            st.info("U.S. Treasury time series unavailable.")
        else:
            fig = go.Figure()
            for symbol in history.columns:
                display_name = label_map.get(symbol, symbol)
                fig.add_trace(
                    go.Scatter(
                        x=history.index,
                        y=history[symbol],
                        mode="lines",
                        name=display_name,
                        line=dict(color=color_map.get(symbol)),
                        hovertemplate=f"%{{y:.2f}}%<br>%{{x|%Y-%m-%d}}<extra>{display_name}</extra>",
                    )
                )
            fig.update_layout(
                template="plotly_white",
                title=(
                    "U.S. Treasury Yields Over Time<br>"
                    "<span style=\"font-size:0.85em;font-weight:400;\">"
                    "Tracks U.S. curve dynamics through time for the selected maturities."
                    "</span>"
                ),
                xaxis_title="Date",
                yaxis_title="Yield (%)",
                height=420,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)


def render() -> None:
    """Entry point for the bonds tab."""
    try:
        st.set_page_config(
            page_title="Bonds | Cross-Asset Market Monitor",
            page_icon=":material/stacked_line_chart:",
            layout="wide",
        )
    except RuntimeError:
        # set_page_config can only be called once per app launch; ignore subsequent calls.
        pass

    st.header(":material/stacked_line_chart: Bonds")
    st.caption(
        "Interactive analytics focused on U.S. Treasuries and German Bunds — monitor level changes, risk metrics, and current yield curves."
    )

    default_start = max(DEFAULT_START_DATE, DEFAULT_END_DATE - timedelta(days=365))

    with st.sidebar:
        options = tuple(BOND_SERIES.keys())
        label_map = format_labels(options)
        default_selection = [symbol for symbol in DEFAULT_SELECTION if symbol in options]
        selected_symbols = st.multiselect(
            "Select bonds",
            options=options,
            default=default_selection,
            format_func=lambda symbol: label_map.get(symbol, symbol),
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
        st.info("Select at least one bond from the sidebar to begin the analysis.")
        return

    with st.spinner("Loading bond analytics..."):
        prices = load_bond_prices(tuple(selected_symbols), start_date, end_date, use_live=True)
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
    selected_label_map = format_labels(available_symbols)

    with st.spinner("Loading full yield curve..."):
        yield_universe = load_bond_prices(ALL_YIELD_SYMBOLS, start_date, end_date, use_live=True)
    yield_label_map = format_labels(ALL_YIELD_SYMBOLS)
    yield_color_map = build_color_map(ALL_YIELD_SYMBOLS)

    regime_scope = regime_df.loc[regime_df.index.intersection(prices.index)] if not regime_df.empty else pd.DataFrame()
    regime_window = regime_scope if show_regime_overlay else pd.DataFrame()

    analytics = compute_metrics(prices)
    levels = analytics["levels"]
    metrics = analytics["metrics"].sort_values("Current Yield (%)", ascending=False)

    color_map = build_color_map(available_symbols)

    last_observation = prices.index.max() if not prices.empty else None
    latest_row = prices.iloc[-1] if not prices.empty else pd.Series(dtype=float)
    render_top_gainers(metrics, latest_row, as_of=last_observation)
    render_performance_chart(levels, color_map, regime_window, selected_label_map)

    tab_yield, tab_risk, tab_correlation = st.tabs(
        ["Yield Curves", "Volatility & Drawdown", "Correlation Heatmap"]
    )

    with tab_yield:
        if yield_universe.empty:
            st.info("Yield data unavailable for the full curve selection.")
        else:
            render_yield_curve_tab(yield_universe, yield_label_map, yield_color_map)

    with tab_risk:
        render_volatility_drawdown_tab(
            analytics["daily_change"],
            color_map,
            selected_label_map,
            analytics["spreads"],
        )

    with tab_correlation:
        render_correlation_tab(analytics["daily_change"], selected_label_map)

    st.subheader("Performance & Risk Summary")
    st.dataframe(
        metrics.rename(index=selected_label_map).style.format(
            {
                "Current Yield (%)": "{:.2f}",
                "1D Change (bp)": "{:+.1f}",
                "5D Change (bp)": "{:+.1f}",
                "20D Change (bp)": "{:+.1f}",
                "10D Vol (bp)": "{:.1f}",
                "63D Range (bp)": "{:.1f}",
            }
        )
    )

    st.subheader("Raw Data")
    export_frame = prices.rename(columns=selected_label_map)
    st.dataframe(export_frame.style.format("{:.2f}%"))

    csv_bytes = export_frame.to_csv(index_label="Date").encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"bonds_raw_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_bonds_raw",
    )


if __name__ == "__main__":
    render()
