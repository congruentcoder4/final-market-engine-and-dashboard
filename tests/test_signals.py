from backend.signals import calculate_signals, score_headline
from tests.helpers import make_history


def test_all_market_signals_are_normalized() -> None:
    analysis = calculate_signals(
        make_history(),
        news_items=[{"title": "Stocks rally on strong growth"}],
    )

    for score in analysis.signals.as_dict().values():
        assert score is not None
        assert -1.0 <= score <= 1.0

    assert analysis.metrics["breadth_universe_count"] == 11
    assert analysis.signals.news == 1.0


def test_headline_scoring_is_easy_to_explain() -> None:
    assert score_headline("Stocks rally on strong growth") == 1.0
    assert score_headline("Stocks plunge on recession fear") == -1.0
    assert score_headline("Markets open Wednesday") == 0.0
