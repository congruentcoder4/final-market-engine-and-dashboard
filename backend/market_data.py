"""
Handles getting and preparing the market data used by the dashboard.

This file pulls the raw market data and organizes it into a format the rest
of the project can easily use. It focuses only on collecting and cleaning up
the data.

The actual market signal calculations are handled in `signals.py`, while
`regime_engine.py` takes those signals and determines the current market
regime.

Keeping these steps separate makes the project easier to understand, test,
and update.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence
import re

import pandas as pd


FUTURES_SYMBOLS: dict[str, str] = {
    "Dow Jones": "YM=F",
    "S&P 500": "ES=F",
    "Nasdaq 100": "NQ=F",
}

SECTOR_ETFS: tuple[str, ...] = (
    "XLC", "XLY", "XLP", "XLE", "XLF", "XLV",
    "XLI", "XLB", "XLK", "XLRE", "XLU",
)

CORE_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ", "^VIX")
TICKER_PATTERN = re.compile(r"^[A-Z0-9.^=][A-Z0-9.\-^=]{0,14}$")


class MarketDataError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    label: str
    price: float
    previous_close: float
    change: float
    change_percent: float
    as_of: str
    volume: float | None = None
    sma20: float | None = None
    high20: float | None = None
    low20: float | None = None
    return20_percent: float | None = None


@dataclass(frozen=True, slots=True)
class MarketData:
    """Raw provider data plus simple quote information for the UI."""

    history: pd.DataFrame
    news_items: tuple[Mapping[str, Any], ...]
    futures: dict[str, Quote]
    quotes: dict[str, Quote]
    as_of: str
    warnings: tuple[str, ...] = ()


DownloadFunction = Callable[..., pd.DataFrame | None]
TickerFactory = Callable[[str], Any]


def normalize_watchlist(tickers: Iterable[str]) -> tuple[str, ...]:
    """Uppercase, validate, and de-duplicate ticker symbols."""

    result: list[str] = []
    seen: set[str] = set()

    for ticker in tickers:
        symbol = str(ticker).strip().upper()
        if not symbol:
            continue
        if not TICKER_PATTERN.fullmatch(symbol):
            raise MarketDataError(f"Invalid ticker: {ticker}")
        if symbol not in seen:
            seen.add(symbol)
            result.append(symbol)

    return tuple(result)


def ticker_frame(history: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extract one ticker from yfinance flat or MultiIndex output."""

    if history is None or history.empty:
        return pd.DataFrame()
    if not isinstance(history.columns, pd.MultiIndex):
        return history.copy()

    for level in range(history.columns.nlevels):
        if symbol in history.columns.get_level_values(level):
            frame = history.xs(symbol, axis=1, level=level, drop_level=True)
            if isinstance(frame, pd.Series):
                frame = frame.to_frame()
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(-1)
            return frame.copy()

    return pd.DataFrame()


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").dropna().astype(float)


def quote_from_frame(symbol: str, label: str, frame: pd.DataFrame) -> Quote | None:
    """Create a simple quote and 20-day reference levels from price history."""

    close = numeric_series(frame, "Close")
    if len(close) < 2:
        return None

    price = float(close.iloc[-1])
    previous = float(close.iloc[-2])
    change = price - previous
    change_percent = (change / previous * 100.0) if previous else 0.0

    volume_series = numeric_series(frame, "Volume")
    volume = float(volume_series.iloc[-1]) if not volume_series.empty else None
    sma20 = float(close.tail(20).mean()) if len(close) >= 20 else None

    high = numeric_series(frame, "High")
    low = numeric_series(frame, "Low")
    high20 = float(high.tail(20).max()) if not high.empty else float(close.tail(20).max())
    low20 = float(low.tail(20).min()) if not low.empty else float(close.tail(20).min())

    return20 = None
    if len(close) >= 21 and float(close.iloc[-21]) != 0:
        return20 = (price / float(close.iloc[-21]) - 1.0) * 100.0

    try:
        timestamp = pd.Timestamp(close.index[-1])
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        as_of = timestamp.isoformat()
    except Exception:
        as_of = str(close.index[-1])

    return Quote(
        symbol=symbol,
        label=label,
        price=round(price, 4),
        previous_close=round(previous, 4),
        change=round(change, 4),
        change_percent=round(change_percent, 4),
        as_of=as_of,
        volume=round(volume, 2) if volume is not None else None,
        sma20=round(sma20, 4) if sma20 is not None else None,
        high20=round(high20, 4),
        low20=round(low20, 4),
        return20_percent=round(return20, 4) if return20 is not None else None,
    )


def build_market_data(
    history: pd.DataFrame,
    *,
    watchlist: Iterable[str],
    news_items: Sequence[Mapping[str, Any]] | None = None,
    fetched_at: datetime | None = None,
) -> MarketData:
    """Turn provider output into the small data object used by the app."""

    if history is None or history.empty:
        raise MarketDataError("The market-data provider returned no price history.")

    watchlist = normalize_watchlist(watchlist)
    warnings: list[str] = []

    futures: dict[str, Quote] = {}
    for label, symbol in FUTURES_SYMBOLS.items():
        quote = quote_from_frame(symbol, label, ticker_frame(history, symbol))
        if quote:
            futures[symbol] = quote
        else:
            warnings.append(f"No usable data for {label} futures.")

    quotes: dict[str, Quote] = {}
    for symbol in watchlist:
        quote = quote_from_frame(symbol, symbol, ticker_frame(history, symbol))
        if quote:
            quotes[symbol] = quote
        else:
            warnings.append(f"No usable data for {symbol}.")

    timestamp = fetched_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    valid_news = tuple(item for item in (news_items or ()) if isinstance(item, Mapping))

    return MarketData(
        history=history,
        news_items=valid_news,
        futures=futures,
        quotes=quotes,
        as_of=timestamp.isoformat(),
        warnings=tuple(warnings),
    )


def fetch_market_data(
    watchlist: Iterable[str],
    *,
    download_function: DownloadFunction | None = None,
    ticker_factory: TickerFactory | None = None,
    timeout: float = 15.0,
) -> MarketData:
    """Download live/delayed Yahoo Finance data for the dashboard."""

    watchlist = normalize_watchlist(watchlist)

    if download_function is None or ticker_factory is None:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise MarketDataError("Install dependencies with: pip install -r requirements.txt") from exc

        download_function = download_function or yf.download
        ticker_factory = ticker_factory or yf.Ticker

    symbols = tuple(
        dict.fromkeys((*CORE_SYMBOLS, *SECTOR_ETFS, *FUTURES_SYMBOLS.values(), *watchlist))
    )

    try:
        history = download_function(
            tickers=list(symbols),
            period="1y",
            interval="1d",
            group_by="ticker",
            auto_adjust=True,
            actions=False,
            threads=True,
            progress=False,
            timeout=timeout,
            multi_level_index=True,
        )
    except Exception as exc:
        raise MarketDataError(f"Price download failed: {exc}") from exc

    news_items: Sequence[Mapping[str, Any]] | None = None
    try:
        spy = ticker_factory("SPY")
        if hasattr(spy, "get_news"):
            news_items = spy.get_news(count=20, tab="news")
        elif hasattr(spy, "news"):
            news_items = spy.news
    except Exception:
        # News is optional. Price-based signals can still run without it.
        news_items = None

    return build_market_data(
        history,
        watchlist=watchlist,
        news_items=news_items,
        fetched_at=datetime.now(timezone.utc),
    )
