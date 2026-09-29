"""
the backend stays seperate from Streamlit so the calculations can stay
simple to test
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st
from yfinance import ticker

from backend.market_data import FUTURES_SYMBOLS, MarketData, Quote
from backend.regime_engine import RegimeResult
from backend.signals import SignalAnalysis


def format_timestamp(value: str) -> str:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone().strftime("%b %d, %Y · %I:%M %p %Z")
    except (TypeError, ValueError):
        return value


def trend_label(quote: Quote) -> str:
    if quote.sma20 is None:
        return "Not enough history"
    if quote.price > quote.sma20:
        return "Above 20-day average"
    if quote.price < quote.sma20:
        return "Below 20-day average"
    return "At 20-day average"


def render_regime(result: RegimeResult, analysis: SignalAnalysis) -> None:
    st.subheader("Market Psychology Regime")
    regime_col, score_col, confidence_col = st.columns(3)
    regime_col.metric("Regime", result.regime.value)
    score_col.metric("Composite score", f"{result.score:+.2f}")
    confidence_col.metric("Confidence", f"{result.confidence:.0%}")

    st.progress((result.score + 1.0) / 2.0)
    st.caption(f"{result.bias} · {result.description}")

    with st.expander("Signal breakdown"):
        rows = [
            {
                "Signal": name.title(),
                "Score (-1 to +1)": value,
                "Weighted contribution": result.contributions.get(name),
            }
            for name, value in result.component_scores.items()
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

        cols = st.columns(4)
        cols[0].metric("VIX", _metric(analysis.metrics.get("vix_level"), ".2f"))
        cols[1].metric(
            "Sectors above 20-day avg",
            _metric(analysis.metrics.get("sectors_above_sma20_pct"), ".0f", "%"),
        )
        cols[2].metric(
            "SPY volume vs avg",
            _metric(analysis.metrics.get("spy_volume_ratio"), ".2f", "×"),
        )
        cols[3].metric("SPY/QQQ RSI", _metric(analysis.metrics.get("index_rsi14"), ".1f"))

        st.caption(
            f"{result.signal_count}/5 signals available · {result.coverage:.0%} weight coverage"
        )


def _metric(value: object, fmt: str, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{float(value):{fmt}}{suffix}"


def render_futures(data: MarketData) -> None:
    st.subheader("Index Futures")
    columns = st.columns(len(FUTURES_SYMBOLS))

    for column, (label, symbol) in zip(columns, FUTURES_SYMBOLS.items()):
        quote = data.futures.get(symbol)
        with column:
            if quote is None:
                st.metric(f"{label} Futures", "Unavailable")
            else:
                st.metric(
                    f"{label} Futures",
                    f"{quote.price:,.2f}",
                    f"{quote.change_percent:+.2f}%",
                )


def render_watchlist(data: MarketData) -> None:
    st.subheader("Watchlist")
    columns = st.columns(min(5, max(1, len(st.session_state.watchlist))))

    for index, ticker in enumerate(st.session_state.watchlist):
        column = columns[index % len(columns)]
        with column:
            quote = data.quotes.get(ticker)
            if quote:
                label = f"{ticker}\n${quote.price:,.2f}"
            else:
                label = ticker
            if st.button(label, key=f"ticker_{ticker}", use_container_width=True):
                st.session_state.selected_ticker = ticker
                st.rerun()


def render_ticker_details(data: MarketData) -> None:
    ticker = st.session_state.get("selected_ticker")
    if not ticker:
        return

    quote = data.quotes.get(ticker)
    st.divider()
    st.subheader(f"{ticker} Details")

    if quote is None:
        st.info(f"No usable market data was returned for {ticker}.")
        return

    cols = st.columns(4)
    cols[0].metric("Price", f"${quote.price:,.2f}")
    cols[1].metric("Daily change", f"{quote.change_percent:+.2f}%")
    cols[2].metric("20-day trend", trend_label(quote))
    cols[3].metric(
        "20-day return",
        "—" if quote.return20_percent is None else f"{quote.return20_percent:+.2f}%",
    )

    history = data.history.get(ticker)
    render_stock_chart(history, ticker)

    if st.button("Remove from watchlist", key=f"remove_{ticker}"):
        st.session_state.watchlist = [x for x in st.session_state.watchlist if x != ticker]
        st.session_state.selected_ticker = None
        st.cache_data.clear()
        st.rerun()


def render_headlines(analysis: SignalAnalysis) -> None:
    if not analysis.headlines:
        return

    with st.expander("Recent headlines used by the news signal"):
        for headline in analysis.headlines[:8]:
            source = f" — {headline.publisher}" if headline.publisher else ""
            tone = "positive" if headline.tone > 0 else "negative" if headline.tone < 0 else "neutral"
            if headline.link:
                st.markdown(f"[{headline.title}]({headline.link}){source} · {tone}")
            else:
                st.write(f"{headline.title}{source} · {tone}")

def render_stock_chart(history, ticker):
    st.subheader(f"{ticker} Price History")

    if history is not None and not history.empty:
        st.line_chart(history["Close"])
    else:
        st.info("No price history available.")

