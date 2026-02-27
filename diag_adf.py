import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

t = np.arange(1000)
# Trend in returns
log_returns = 0.01 * t + np.random.normal(0, 1, 1000)
res = adfuller(log_returns, autolag='AIC')
print(f"Stat: {res[0]}")
print(f"p-value: {res[1]}")
print(f"Crit: {res[4]}")
