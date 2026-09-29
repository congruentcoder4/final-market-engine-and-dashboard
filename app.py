"""Market Psychology Dashboard.

Application flow:
1. fetch market data
2. convert raw market data into normalized signals
3. calculate the market regime
4. display the result
"""

from __future__ import annotations

import streamlit as st

from backend.market_data import MarketDataError, fetch_market_data, normalize_watchlist
from backend.regime_engine import RegimeEngineError, calculate_market_regime
from backend.signals import calculate_signals
from frontend.dashboard import (
    format_timestamp,
    render_futures,
    render_headlines,
    render_regime,
    render_ticker_details,
    render_watchlist,
)


DEFAULT_WATCHLIST = ["SPY", "QQQ", "NVDA", "TSLA", "RKLB"]

st.set_page_config(page_title="Market Psychology Dashboard", page_icon="📈", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def load_market_data(watchlist: tuple[str, ...]):
    return fetch_market_data(watchlist)


def initialize_state() -> None:
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = DEFAULT_WATCHLIST.copy()
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = None


def add_ticker_control() -> None:
    with st.popover("➕ Add ticker"):
        new_ticker = st.text_input("Ticker", placeholder="ASTS")
        if st.button("Add to watchlist", use_container_width=True):
            try:
                normalized = normalize_watchlist([new_ticker])
                if not normalized:
                    st.warning("Enter a ticker.")
                elif normalized[0] in st.session_state.watchlist:
                    st.warning("Ticker is already in your watchlist.")
                else:
                    st.session_state.watchlist.append(normalized[0])
                    st.session_state.selected_ticker = normalized[0]
                    st.cache_data.clear()
                    st.rerun()
            except MarketDataError as exc:
                st.warning(str(exc))


def main() -> None:
    initialize_state()

    title_col, refresh_col = st.columns([5, 1])
    with title_col:
        st.title("Market Psychology Dashboard")
        st.caption("Market data → normalized signals → weighted regime score")
    with refresh_col:
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    try:
        # 1. fetch market data
        data = load_market_data(tuple(st.session_state.watchlist))

        # 2. convert raw data into our five normalized signals
        analysis = calculate_signals(data.history, data.news_items)

        # 3. weight the signals and classify the type of market regime for the current market environment
        regime = calculate_market_regime(analysis.signals)

    except (MarketDataError, RegimeEngineError) as exc:
        st.error(f"Dashboard can not be calculated: {exc}")
        return

    st.caption(f"Live/delayed data fetched {format_timestamp(data.as_of)} · cached for five minutes")

    # 4. Display the results.
    render_regime(regime, analysis)
    st.divider()
    render_futures(data)
    st.divider()
    render_watchlist(data)
    add_ticker_control()
    render_ticker_details(data)
    render_headlines(analysis)

    all_warnings = (*data.warnings, *analysis.warnings)
    if all_warnings:
        with st.expander("Data-quality notices"):
            for warning in all_warnings:
                st.write(f"• {warning}")

    st.divider()
    st.caption(
        "Prices and headlines are taken from yfinance/Yahoo Finance and may be delayed. "
        "This project is only for research and education purposes."
    )


if __name__ == "__main__":
    main()
