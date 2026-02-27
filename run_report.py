import numpy as np
import pandas as pd
import yfinance as yf
from ingestion import validate_and_prepare
from backtesting import walk_forward_backtest

ticker = "SPY"
data = yf.download(ticker, start="2018-01-01", end="2024-01-01", progress=False)
prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
if isinstance(prices, pd.DataFrame): prices = prices.squeeze()
    
returns_series, _ = validate_and_prepare(prices)
ret_arr = returns_series.values.astype(np.float64)

# Run a small backtest
result = walk_forward_backtest(ret_arr, 1, 1, min_train=len(ret_arr)-50)

with open("backtest_report.txt", "w") as f:
    f.write(result.summary_report())
