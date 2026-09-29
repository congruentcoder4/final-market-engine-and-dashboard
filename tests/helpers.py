from __future__ import annotations

import numpy as np
import pandas as pd

from backend.market_data import FUTURES_SYMBOLS, SECTOR_ETFS


def make_history() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=80, freq="B")
    symbols = ["SPY", "QQQ", "^VIX", *SECTOR_ETFS, *FUTURES_SYMBOLS.values(), "NVDA", "TSLA", "RKLB"]
    history = pd.DataFrame(index=dates)

    for position, symbol in enumerate(symbols):
        if symbol == "^VIX":
            close = np.linspace(24.0, 16.0, len(dates))
            volume = np.full(len(dates), 500_000)
        else:
            slope = 0.25 + position * 0.004
            close = 100.0 + np.arange(len(dates)) * slope
            volume = 1_000_000 + np.arange(len(dates)) * 2_000 + position

        history[(symbol, "Close")] = close
        history[(symbol, "Open")] = close - 0.2
        history[(symbol, "High")] = close + 0.5
        history[(symbol, "Low")] = close - 0.5
        history[(symbol, "Volume")] = volume

    history.columns = pd.MultiIndex.from_tuples(history.columns)
    return history
