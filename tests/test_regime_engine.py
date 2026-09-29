import pytest

from backend.regime_engine import MarketRegime, MarketSignals, RegimeEngineError, calculate_market_regime


def test_positive_signals_create_positive_regime() -> None:
    result = calculate_market_regime(MarketSignals(1, 1, 1, 1, 1))
    assert result.score == 1.0
    assert result.regime is MarketRegime.EUPHORIA


def test_missing_signals_are_reweighted() -> None:
    result = calculate_market_regime(MarketSignals(vix=0.4, breadth=0.2, volume=0.0))
    expected = (0.4 * 0.25 + 0.2 * 0.25) / 0.65
    assert result.score == pytest.approx(expected, abs=1e-4)
    assert result.signal_count == 3
    assert result.coverage == pytest.approx(0.65)


def test_too_few_signals_raise_clear_error() -> None:
    with pytest.raises(RegimeEngineError, match="At least 3"):
        calculate_market_regime(MarketSignals(vix=0.2, news=-0.1))
