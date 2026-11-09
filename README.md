# Market Psychology Dashboard — 

A Streamlit project that turns live or delayed market data into five normalized signals and combines them into a market-psychology regime.

## The entire project in four steps


1. Get market data
        
2. Calculate the 5 signals (-1 to +1)
        
3. Weight the signals and calculate the regime
        
4. Display the result in Streamlit


## Project structure

```text
market-dashboard-clean/
├── app.py                    # main application
├── backend/
│   ├── market_data.py        # Gets and prepares market data
│   ├── signals.py            # Calculates market signals like vix, breadth, etc
│   └── regime_engine.py     # calculates market regime
        llm_engine.py
├── frontend/
│   └── dashboard.py          # Streamlit display functions
├── tests/
│   ├── test_market_data.py
│   ├── test_signals.py
│   └── test_regime_engine.py
├── requirements.txt
└── pyproject.toml
```


How It Works
Yahoo Finance data → normalized signals → weighted regime score → Streamlit dashboard → optional GPT-OSS interpretation


## What Each File Does

### `backend/market_data.py`
Handles market data collection. It validates ticker symbols, downloads historical market data and news from Yahoo Finance, and builds quote information used by the application.

### `backend/signals.py`
Transforms raw market data into standardized signals on a `-1` to `+1` scale.

The five signals are:

- VIX
- Market breadth
- Volume pressure
- SPY/QQQ momentum and RSI sentiment
- Headline tone

Standardizing the signals allows indicators with different underlying measurements to be compared and combined consistently.

### `backend/regime_engine.py`
Combines the five signals using predefined weights to calculate the overall market regime.

If a signal is unavailable, its weight is removed and the remaining weights are automatically normalized so that missing data does not disproportionately affect the final result.

### 'backend/llm_engine.py' 
Market snapshot and GPT-OSS analysis (simple implementation of running a local llm)

The AI feature expects Ollama at http://localhost:11434. The dashboard's non-AI features do not require Ollama.

### `app.py`
Coordinates the application by connecting data collection, signal calculation, regime classification, and the Streamlit interface.

```python
data = load_market_data(...)
analysis = calculate_signals(data.history, data.news_items)
regime = calculate_market_regime(analysis.signals)
# render the result
```

The application follows a simple pipeline:

**Market Data → Signal Calculation → Weighted Regime Classification → User Interface**

## Run the Project

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```



## Run the Tests

```bash
pytest
```

For research purposes only. Market data can be delayed / not financial advice.
