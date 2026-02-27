import numpy as np
import pandas as pd
from ingestion import validate_and_prepare
from exceptions import NonStationaryDataError

t = np.arange(1000)
log_returns = 0.01 * t + np.random.normal(0, 1, 1000)
prices = np.exp(np.cumsum(log_returns))
prices_series = pd.Series(prices)

try:
    validate_and_prepare(prices_series)
    print("FAILED: No error raised")
except NonStationaryDataError as e:
    print(f"SUCCESS: Caught {type(e).__name__}: {e}")
except Exception as e:
    print(f"FAILED: Caught unexpected {type(e).__name__}: {e}")
