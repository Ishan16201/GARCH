import numpy as np
import pandas as pd
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from scipy.stats import chi2, norm

from estimation import fit_garch, GARCHResult
from forecasting import forecast
from exceptions import ClusteringRiskWarning

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """
    Container for GARCH out-of-sample backtest results.
    """
    forecasted_variance: np.ndarray
    realized_variance: np.ndarray
    returns: np.ndarray
    var_95: np.ndarray
    var_99: np.ndarray
    
    # Accuracy Metrics
    qlike: float
    mse: float
    mae: float
    
    # Statistical Tests
    kupiec_95: Dict[str, Any]
    kupiec_99: Dict[str, Any]
    christoffersen_95: Dict[str, Any]
    christoffersen_99: Dict[str, Any]

    def summary_report(self) -> str:
        """
        Generate a clean summary report of the backtest.
        """
        lines = [
            "--- GARCH Walk-Forward Backtest Report ---",
            f"OOS Observations: {len(self.returns)}",
            "-" * 42,
            f"{'Metric':<25} | {'Value':>12}",
            "-" * 42,
            f"{'Q-Likelihood (QLIKE)':<25} | {self.qlike:>12.6f}",
            f"{'MSE (Variance)':<25} | {self.mse:>12.8f}",
            f"{'MAE (Variance)':<25} | {self.mae:>12.8f}",
            "-" * 42,
            f"{'VaR Level':<10} | {'Breaches':<8} | {'Kupiec':<8} | {'Independence':<12}",
            f"{'95%':<10} | {self.kupiec_95['breaches']:<8} | "
            f"{'Pass' if self.kupiec_95['p_value'] > 0.05 else 'Fail':<8} | "
            f"{'Pass' if self.christoffersen_95['p_value'] > 0.05 else 'Fail':<12}",
            f"{'99%':<10} | {self.kupiec_99['breaches']:<8} | "
            f"{'Pass' if self.kupiec_99['p_value'] > 0.05 else 'Fail':<8} | "
            f"{'Pass' if self.christoffersen_99['p_value'] > 0.05 else 'Fail':<12}",
            "-" * 42,
        ]
        return "\n".join(lines)

def kupiec_pof_test(breaches: np.ndarray, p_star: float) -> Dict[str, Any]:
    """
    Perform Kupiec's Proportion of Failures (POF) test.
    
    Mathematical Notes
    ------------------
    H0: Observed breach rate = Theoretical breach rate (p*).
    LR_POF = -2 * ln[(1-p*)^(T-x) * p*^x / (1-p_hat)^(T-x) * p_hat^x]
    where x = number of breaches, T = total observations, p_hat = x/T.
    
    Returns
    -------
    Dict[str, Any]
        Statistic, p-value, and breaches count.
    """
    T = len(breaches)
    x = np.sum(breaches)
    p_hat = x / T
    
    # Avoid log(0)
    if x == 0:
        lr = -2 * (T * np.log(1 - p_star))
    elif x == T:
        lr = -2 * (T * np.log(p_star))
    else:
        num = (1 - p_star)**(T - x) * (p_star**x)
        den = (1 - p_hat)**(T - x) * (p_hat**x)
        lr = -2 * np.log(num / den)
        
    p_val = 1 - chi2.cdf(lr, df=1)
    
    return {
        "breaches": int(x),
        "actual_rate": p_hat,
        "lr_stat": lr,
        "p_value": p_val
    }

def christoffersen_test(breaches: np.ndarray) -> Dict[str, Any]:
    """
    Perform Christoffersen's Independence test.
    
    Mathematical Notes
    ------------------
    Tests for first-order Markov dependence in exceedances.
    H0: Exceedances are independent.
    
    Returns
    -------
    Dict[str, Any]
        Likelihood ratio and p-value.
    """
    T = len(breaches)
    # Binary sequence of hits
    I = breaches.astype(int)
    
    # Transitions
    # n_ij: number of times state j follows state i
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        if I[t-1] == 0 and I[t] == 0: n00 += 1
        elif I[t-1] == 0 and I[t] == 1: n01 += 1
        elif I[t-1] == 1 and I[t] == 0: n10 += 1
        elif I[t-1] == 1 and I[t] == 1: n11 += 1
        
    # Probabilities
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    
    # Null likelihood
    L_null = (1 - pi)**(n00 + n10) * pi**(n01 + n11)
    # Alternative likelihood
    L_alt = (1 - pi01)**n00 * pi01**n01 * (1 - pi11)**n10 * pi11**n11
    
    # Max to prevent numerical issues
    LR_ind = -2 * np.log(max(L_null / L_alt, 1e-10))
    p_val = 1 - chi2.cdf(LR_ind, df=1)
    
    return {
        "lr_stat": LR_ind,
        "p_value": p_val
    }

def walk_forward_backtest(returns: np.ndarray, p: int, q: int, min_train: int = 500) -> BacktestResult:
    """
    Perform walk-forward backtest with expanding window refit.
    """
    T = len(returns)
    oos_size = T - min_train
    
    forecasts = np.zeros(oos_size)
    actual_returns = returns[min_train:]
    realized_var = actual_returns**2
    
    print(f"Starting Walk-Forward Backtest (OOS size: {oos_size})...")
    
    for i in tqdm(range(oos_size)):
        t = min_train + i
        train_data = returns[:t]
        
        # Fit model on window [0..t]
        try:
            # Using fewer starts for backtest speed
            res = fit_garch(train_data, p, q, n_starts=2)
            # Forecast T+1
            fc = forecast(res, train_data, horizon=1, n_simulations=0)
            forecasts[i] = fc.point_forecast[0]
        except Exception as e:
            logger.warning(f"Refit at index {t} failed: {e}. Using previous forecast.")
            forecasts[i] = forecasts[i-1] if i > 0 else np.var(train_data)

    # 1. Accuracy Metrics
    # QLIKE: (1/T) * Σ[σ²_t/σ̂²_t - ln(σ²_t/σ̂²_t) - 1]
    ratio = realized_var / forecasts
    qlike = np.mean(ratio - np.log(ratio) - 1)
    mse = np.mean((realized_var - forecasts)**2)
    mae = np.mean(np.abs(realized_var - forecasts))
    
    # 2. VaR Backtesting
    # 95% level: z = -1.645
    # 99% level: z = -2.326
    sigma_oos = np.sqrt(forecasts)
    var_95 = -1.645 * sigma_oos
    var_99 = -2.326 * sigma_oos
    
    breaches_95 = actual_returns < var_95
    breaches_99 = actual_returns < var_99
    
    kupiec_95 = kupiec_pof_test(breaches_95, 0.05)
    kupiec_99 = kupiec_pof_test(breaches_99, 0.01)
    
    christoffersen_95 = christoffersen_test(breaches_95)
    christoffersen_99 = christoffersen_test(breaches_99)
    
    if christoffersen_95['p_value'] <= 0.05 or christoffersen_99['p_value'] <= 0.05:
        warnings.warn("VaR exceedances exhibit clustering. Volatility dynamics may be mis-specified.", ClusteringRiskWarning)

    return BacktestResult(
        forecasted_variance=forecasts,
        realized_variance=realized_var,
        returns=actual_returns,
        var_95=var_95,
        var_99=var_99,
        qlike=qlike,
        mse=mse,
        mae=mae,
        kupiec_95=kupiec_95,
        kupiec_99=kupiec_99,
        christoffersen_95=christoffersen_95,
        christoffersen_99=christoffersen_99
    )

if __name__ == "__main__":
    from ingestion import validate_and_prepare
    import yfinance as yf
    
    # Silence secondary logs
    logging.getLogger('yfinance').setLevel(logging.ERROR)
    logging.getLogger('estimation').setLevel(logging.ERROR)
    
    print("--- GARCH Pipeline: Walk-Forward Backtesting ---")
    ticker = "SP500" # Using a longer index history for backtest robustness
    # SPY usually has plenty of data
    ticker = "SPY"
    
    data = yf.download(ticker, start="2018-01-01", end="2024-01-01", progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    if isinstance(prices, pd.DataFrame): prices = prices.squeeze()
        
    returns_series, _ = validate_and_prepare(prices)
    ret_arr = returns_series.values.astype(np.float64)
    
    # Run backtest GARCH(1,1)
    # We use a short OOS size for the demonstration/demo code to run quickly
    # In production, this would be the full history
    result = walk_forward_backtest(ret_arr, 1, 1, min_train=len(ret_arr)-252) # 1 year OOS
    
    print("\nBacktest Complete.")
    print(result.summary_report())
