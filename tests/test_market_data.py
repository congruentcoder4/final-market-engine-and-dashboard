from datetime import datetime, timezone

import pytest

from backend.market_data import MarketDataError, build_market_data, normalize_watchlist
from tests.helpers import make_history


def test_watchlist_is_normalized_and_deduplicated() -> None:
    assert normalize_watchlist([" spy ", "SPY", "brk-b"]) == ("SPY", "BRK-B")

    with pytest.raises(MarketDataError):
        normalize_watchlist(["BAD TICKER"])


def test_market_data_builds_quotes_and_futures() -> None:
    data = build_market_data(
        make_history(),
        watchlist=["SPY", "NVDA"],
        fetched_at=datetime(2026, 9, 28, tzinfo=timezone.utc),
    )

    assert set(data.quotes) == {"SPY", "NVDA"}
    assert len(data.futures) == 3
    assert data.quotes["NVDA"].sma20 is not None
    assert data.as_of == "2026-09-28T00:00:00+00:00"
