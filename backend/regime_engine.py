"""
Takes the market signals and uses them to figure out the current market regime.

This file receives the normalized signal scores from `signals.py`, applies
weights to each one, and combines them into a final market score. That score
is then used to classify the market into a regime, such as bullish, bearish,
or neutral.

It doesn't handle downloading market data or displaying anything on the
dashboard. Its main job is just to take the signals, combine them, and return
the final result.

Keeping this logic separate makes it easier to test and change without
affecting the rest of the project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from statistics import pstdev


class MarketRegime(str, Enum):
    PANIC = "Panic"
    FEAR = "Fear"
    CAUTION = "Caution"
    NEUTRAL = "Neutral"
    OPTIMISM = "Optimism"
    GREED = "Greed"
    EUPHORIA = "Euphoria"


@dataclass(frozen=True, slots=True)
class MarketSignals:
    """Normalized signals: -1 is bearish, 0 is neutral, +1 is bullish."""

    vix: float | None = None
    breadth: float | None = None
    volume: float | None = None
    sentiment: float | None = None
    news: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RegimeResult:
    score: float
    regime: MarketRegime
    bias: str
    confidence: float
    description: str
    component_scores: dict[str, float | None]
    contributions: dict[str, float]
    signal_count: int
    coverage: float


class RegimeEngineError(ValueError):
    pass


DEFAULT_WEIGHTS: dict[str, float] = {
    "vix": 0.25,
    "breadth": 0.25,
    "volume": 0.15,
    "sentiment": 0.20,
    "news": 0.15,
}

REGIME_DESCRIPTIONS: dict[MarketRegime, str] = {
    MarketRegime.PANIC: "Broad risk aversion and heavy selling pressure.",
    MarketRegime.FEAR: "Defensive positioning is dominating the market.",
    MarketRegime.CAUTION: "Risk appetite is weak and investors are cautious.",
    MarketRegime.NEUTRAL: "Signals are balanced or too mixed for a clear direction.",
    MarketRegime.OPTIMISM: "Risk appetite is improving with constructive participation.",
    MarketRegime.GREED: "Strong risk-taking is widespread across the market.",
    MarketRegime.EUPHORIA: "Risk appetite is extremely strong and potentially crowded.",
}


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _validate_signal(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RegimeEngineError(f"{name} must be a number or None.")
    if not isfinite(float(value)) or not -1.0 <= float(value) <= 1.0:
        raise RegimeEngineError(f"{name} must be between -1.0 and 1.0.")


def classify_regime(score: float) -> MarketRegime:
    """Map the composite score to a human-readable regime."""

    if score < -0.65:
        return MarketRegime.PANIC
    if score < -0.35:
        return MarketRegime.FEAR
    if score < -0.12:
        return MarketRegime.CAUTION
    if score <= 0.12:
        return MarketRegime.NEUTRAL
    if score <= 0.35:
        return MarketRegime.OPTIMISM
    if score <= 0.65:
        return MarketRegime.GREED
    return MarketRegime.EUPHORIA


def calculate_market_regime(
    signals: MarketSignals,
    *,
    weights: dict[str, float] | None = None,
    minimum_signals: int = 3,
) -> RegimeResult:
    """Weight available signals, calculate the score, and classify the regime."""

    scores = signals.as_dict()
    for name, value in scores.items():
        _validate_signal(name, value)

    configured_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise RegimeEngineError(f"Unknown weight: {', '.join(sorted(unknown))}")
        configured_weights.update(weights)

    for name, weight in configured_weights.items():
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise RegimeEngineError(f"{name} weight must be numeric.")
        if not isfinite(float(weight)) or weight < 0:
            raise RegimeEngineError(f"{name} weight must be non-negative and finite.")

    available = {
        name: float(value)
        for name, value in scores.items()
        if value is not None and configured_weights[name] > 0
    }
    if len(available) < minimum_signals:
        raise RegimeEngineError(
            f"At least {minimum_signals} market signals are required; "
            f"received {len(available)}."
        )

    total_weight = sum(configured_weights.values())
    available_weight = sum(configured_weights[name] for name in available)
    if total_weight <= 0 or available_weight <= 0:
        raise RegimeEngineError("Signal weights must add up to more than zero.")

    # Missing signals are removed, then the remaining weights are re-normalized.
    normalized_weights = {
        name: configured_weights[name] / available_weight for name in available
    }
    contributions = {
        name: available[name] * normalized_weights[name] for name in available
    }
    score = _clamp(sum(contributions.values()))

    dispersion = pstdev(available.values()) if len(available) > 1 else 0.0
    agreement = _clamp(1.0 - dispersion, 0.0, 1.0)
    coverage = available_weight / total_weight
    confidence = _clamp(
        0.10 + 0.55 * agreement + 0.25 * coverage + 0.10 * abs(score),
        0.0,
        1.0,
    )

    regime = classify_regime(score)
    bias = "Risk-on" if score > 0.12 else "Risk-off" if score < -0.12 else "Neutral"

    return RegimeResult(
        score=round(score, 4),
        regime=regime,
        bias=bias,
        confidence=round(confidence, 4),
        description=REGIME_DESCRIPTIONS[regime],
        component_scores=scores,
        contributions={name: round(value, 4) for name, value in contributions.items()},
        signal_count=len(available),
        coverage=round(coverage, 4),
    )
