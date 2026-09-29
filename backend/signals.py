"""Converting raw market data into five normalized signals.

-each signal returns a score from -1 (risk-off) to +1 (risk-on). 
-keeping all signal calculations here makes the project's central data flow easy to follow
-each signal also returns a dictionary of metrics that can be displayed in the dashboard
-all five signals are calculated in one top-level function, `calculate_signals`, 
(it returns a dataclass containing the five signal scores, metrics, and any warnings about missing data)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Mapping, Sequence
import re

import pandas as pd

from backend.market_data import SECTOR_ETFS, numeric_series, ticker_frame
from backend.regime_engine import MarketSignals


POSITIVE_WORDS = {
    "advance", "advances", "beat", "beats", "bullish", "gain", "gains",
    "growth", "optimism", "rally", "rebound", "record", "recover", "recovery",
    "rise", "rises", "strong", "surge", "surges", "upgrade", "upgrades",
}
NEGATIVE_WORDS = {
    "bearish", "crash", "crisis", "decline", "declines", "default",
    "downgrade", "downgrades", "drop", "drops", "fear", "fall", "falls",
    "inflation", "layoff", "layoffs", "miss", "misses", "plunge", "plunges",
    "recession", "risk", "selloff", "slump", "tariff", "tariffs", "weak",
}


@dataclass(frozen=True, slots=True)
class Headline:
    title: str
    publisher: str | None = None
    link: str | None = None
    published_at: str | None = None
    tone: float = 0.0


@dataclass(frozen=True, slots=True)
class SignalAnalysis:
    signals: MarketSignals
    metrics: dict[str, float | int | str | None]
    headlines: tuple[Headline, ...]
    warnings: tuple[str, ...]


def clamp(value: float) -> float:
    if not isfinite(float(value)):
        raise ValueError("Signal scores must be finite.")
    return max(-1.0, min(1.0, float(value)))


def calculate_vix_signal(frame: pd.DataFrame) -> tuple[float | None, dict[str, float | None]]:
    close = numeric_series(frame, "Close")
    if len(close) < 20:
        return None, {"vix_level": None, "vix_sma20": None}

    level = float(close.iloc[-1])
    sma20 = float(close.tail(20).mean())
    level_score = clamp((20.0 - level) / 15.0)
    trend_score = clamp((sma20 - level) / max(3.0, sma20 * 0.25))
    score = clamp(0.70 * level_score + 0.30 * trend_score)

    return round(score, 4), {
        "vix_level": round(level, 4),
        "vix_sma20": round(sma20, 4),
    }


def calculate_breadth_signal(
    history: pd.DataFrame,
    symbols: Sequence[str] = SECTOR_ETFS,
) -> tuple[float | None, dict[str, float | int | None]]:
    above_20 = advancing = positive_5d = usable = 0

    for symbol in symbols:
        close = numeric_series(ticker_frame(history, symbol), "Close")
        if len(close) < 20:
            continue

        usable += 1
        latest = float(close.iloc[-1])
        above_20 += latest > float(close.tail(20).mean())
        advancing += latest > float(close.iloc[-2])
        positive_5d += len(close) >= 6 and latest > float(close.iloc[-6])

    if usable < max(3, len(symbols) // 2):
        return None, {"breadth_universe_count": usable, "sectors_above_sma20_pct": None}

    pct_above = above_20 / usable
    pct_advancing = advancing / usable
    pct_positive_5d = positive_5d / usable
    score = clamp(
        0.50 * (2 * pct_above - 1)
        + 0.25 * (2 * pct_advancing - 1)
        + 0.25 * (2 * pct_positive_5d - 1)
    )

    return round(score, 4), {
        "breadth_universe_count": usable,
        "sectors_above_sma20_pct": round(pct_above * 100, 2),
        "sectors_advancing_1d_pct": round(pct_advancing * 100, 2),
        "sectors_positive_5d_pct": round(pct_positive_5d * 100, 2),
    }


def calculate_volume_signal(frame: pd.DataFrame) -> tuple[float | None, dict[str, float | None]]:
    close = numeric_series(frame, "Close")
    volume = numeric_series(frame, "Volume")
    joined = pd.concat([close.rename("close"), volume.rename("volume")], axis=1).dropna()

    if len(joined) < 21:
        return None, {"spy_volume_ratio": None, "signed_volume_pressure_5d": None}

    returns = joined["close"].pct_change()
    direction = returns.apply(lambda x: 1.0 if x > 0 else -1.0 if x < 0 else 0.0)
    recent = joined.tail(5)
    recent_direction = direction.reindex(recent.index).fillna(0.0)
    recent_volume = float(recent["volume"].sum())
    pressure = float((recent_direction * recent["volume"]).sum()) / recent_volume if recent_volume else 0.0

    average_volume = float(joined["volume"].iloc[-21:-1].mean())
    latest_volume = float(joined["volume"].iloc[-1])
    volume_ratio = latest_volume / average_volume if average_volume else 1.0
    latest_return = float(returns.iloc[-1]) if pd.notna(returns.iloc[-1]) else 0.0
    conviction = max(0.0, min(1.0, (volume_ratio - 0.50) / 1.00))
    score = clamp(0.75 * pressure + 0.25 * clamp(latest_return / 0.02) * conviction)

    return round(score, 4), {
        "spy_volume_ratio": round(volume_ratio, 4),
        "signed_volume_pressure_5d": round(pressure, 4),
        "spy_daily_return_pct": round(latest_return * 100, 4),
    }


def _rsi(close: pd.Series, period: int = 14) -> float | None:
    if len(close) < period + 1:
        return None

    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean().iloc[-1]

    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return None
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = float(avg_gain) / float(avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def calculate_sentiment_signal(
    history: pd.DataFrame,
    symbols: Sequence[str] = ("SPY", "QQQ"),
) -> tuple[float | None, dict[str, float | int | None]]:
    scores: list[float] = []
    returns_5d: list[float] = []
    returns_20d: list[float] = []
    rsi_values: list[float] = []

    for symbol in symbols:
        close = numeric_series(ticker_frame(history, symbol), "Close")
        if len(close) < 21:
            continue

        return_5d = float(close.iloc[-1] / close.iloc[-6] - 1)
        return_20d = float(close.iloc[-1] / close.iloc[-21] - 1)
        rsi = _rsi(close)
        if rsi is None:
            continue

        score = clamp(
            0.30 * clamp(return_5d / 0.05)
            + 0.40 * clamp(return_20d / 0.12)
            + 0.30 * clamp((rsi - 50.0) / 25.0)
        )
        scores.append(score)
        returns_5d.append(return_5d)
        returns_20d.append(return_20d)
        rsi_values.append(rsi)

    if not scores:
        return None, {
            "sentiment_universe_count": 0,
            "index_return_5d_pct": None,
            "index_return_20d_pct": None,
            "index_rsi14": None,
        }

    return round(sum(scores) / len(scores), 4), {
        "sentiment_universe_count": len(scores),
        "index_return_5d_pct": round(sum(returns_5d) / len(returns_5d) * 100, 4),
        "index_return_20d_pct": round(sum(returns_20d) / len(returns_20d) * 100, 4),
        "index_rsi14": round(sum(rsi_values) / len(rsi_values), 4),
    }


def score_headline(title: str) -> float:
    words = re.findall(r"[a-z]+", title.lower())
    positive = sum(word in POSITIVE_WORDS for word in words)
    negative = sum(word in NEGATIVE_WORDS for word in words)
    matches = positive + negative
    return 0.0 if matches == 0 else clamp((positive - negative) / matches)


def _headline_fields(item: Mapping[str, Any]) -> tuple[str, str | None, str | None, str | None]:
    content = item.get("content") if isinstance(item.get("content"), Mapping) else item
    title = str(content.get("title") or item.get("title") or "").strip()

    provider = content.get("provider") or item.get("publisher")
    if isinstance(provider, Mapping):
        provider = provider.get("displayName") or provider.get("name")
    publisher = str(provider).strip() if provider else None

    link = content.get("canonicalUrl") or content.get("clickThroughUrl") or item.get("link")
    if isinstance(link, Mapping):
        link = link.get("url")
    link = str(link).strip() if link else None

    published = content.get("pubDate") or item.get("providerPublishTime")
    if isinstance(published, (int, float)):
        try:
            published_at = datetime.fromtimestamp(float(published), tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            published_at = str(published)
    else:
        published_at = str(published) if published else None

    return title, publisher, link, published_at


def calculate_news_signal(
    news_items: Sequence[Mapping[str, Any]] | None,
) -> tuple[float | None, tuple[Headline, ...], dict[str, int]]:
    if not news_items:
        return None, (), {"headline_count": 0, "headline_keyword_match_count": 0}

    headlines: list[Headline] = []
    matched = 0

    for item in news_items:
        title, publisher, link, published_at = _headline_fields(item)
        if not title:
            continue
        tone = score_headline(title)
        matched += tone != 0
        headlines.append(Headline(title, publisher, link, published_at, round(tone, 4)))

    if not headlines:
        return None, (), {"headline_count": 0, "headline_keyword_match_count": 0}

    score = clamp(sum(item.tone for item in headlines) / len(headlines))
    return round(score, 4), tuple(headlines), {
        "headline_count": len(headlines),
        "headline_keyword_match_count": matched,
    }


def calculate_signals(
    history: pd.DataFrame,
    news_items: Sequence[Mapping[str, Any]] | None = None,
) -> SignalAnalysis:
    """Calculate all five signals in one clear, top-level function."""

    vix, vix_metrics = calculate_vix_signal(ticker_frame(history, "^VIX"))
    breadth, breadth_metrics = calculate_breadth_signal(history)
    volume, volume_metrics = calculate_volume_signal(ticker_frame(history, "SPY"))
    sentiment, sentiment_metrics = calculate_sentiment_signal(history)
    news, headlines, news_metrics = calculate_news_signal(news_items)

    signals = MarketSignals(
        vix=vix,
        breadth=breadth,
        volume=volume,
        sentiment=sentiment,
        news=news,
    )

    metrics: dict[str, float | int | str | None] = {}
    for values in (vix_metrics, breadth_metrics, volume_metrics, sentiment_metrics, news_metrics):
        metrics.update(values)

    warnings = tuple(
        f"{name.title()} signal is unavailable and will be reweighted out."
        for name, value in signals.as_dict().items()
        if value is None
    )

    return SignalAnalysis(signals, metrics, headlines, warnings)
