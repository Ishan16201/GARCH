import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import skew, kurtosis
from typing import Tuple, Dict, Any

from exceptions import NonStationaryDataError, ArchEffectsNotFoundWarning

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Compute log-returns from a price series.

    Mathematical Notes
    ------------------
    Log-returns are defined as:
        r_t = ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    
    Justification for GARCH modeling:
    1. Time Additivity: The log-return over multiple periods is the sum of 
       single-period log-returns: ln(P_T/P_0) = sum(ln(P_t/P_{t-1})).
    2. Numerical Stability: Log-transforms often stabilize variance and transform 
       exponential price growth into linear return space.
    3. Theoretical Fit: Many GARCH formulations assume normally distributed (or 
       Student-t) innovations in log-space, which are more likely to hold for 
       r_t than for simple returns (P_t - P_{t-1}) / P_{t-1}.

    Parameters
    ----------
    prices : pd.Series
        Historical price series (e.g., Adjusted Close).

    Returns
    -------
    pd.Series
        Series of logarithmic returns, with the first NaN dropped.
    """
    log_returns = np.log(prices).diff().dropna()
    return log_returns

def perform_stationarity_test(returns: pd.Series) -> Dict[str, Any]:
    """
    Perform the Augmented Dickey-Fuller (ADF) test for stationarity.

    Mathematical Notes
    ------------------
    Null Hypothesis (H0): The series has a unit root (is non-stationary).
    Alternative Hypothesis (H1): The series is stationary (no unit root).

    Significance Levels:
    - Enforced at 1%, 5%, and 10%. Rejection requires the test statistic 
      to be less than the critical value.

    Parameters
    ----------
    returns : pd.Series
        The log-return series to test.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing ADF statistic, p-value, and critical values.

    Raises
    ------
    NonStationaryDataError
        If H0 cannot be rejected at even the 10% significance level.
    """
    result = adfuller(returns, autolag='AIC')
    statistic = result[0]
    p_value = result[1]
    crit_values = result[4]
    print(f"DEBUG ADF: Stat={statistic}, Crit10%={crit_values['10%']}")

    # Check rejection at 10% (least strict)
    if statistic > crit_values['10%']:
        raise NonStationaryDataError(
            f"Series is non-stationary. ADF Statistic ({statistic:.4f}) > "
            f"10% Critical Value ({crit_values['10%']:.4f})."
        )

    return {
        "adf_statistic": statistic,
        "p_value": p_value,
        "critical_values": crit_values,
        "is_stationary": True
    }

def check_arch_effects(returns: pd.Series, lags: int = 10) -> Dict[str, Any]:
    """
    Test for conditional heteroskedasticity (ARCH effects) using Ljung-Box 
    on squared returns.

    Mathematical Notes
    ------------------
    Ljung-Box test on squared residuals (r_t^2) checks for autocorrelation 
    in the variance process.
    Q(m) = n(n+2) * sum_{k=1}^m (rho^2_k / (n-k))
    where rho_k is the autocorrelation of squared returns at lag k.

    Parameters
    ----------
    returns : pd.Series
        Log-returns.
    lags : int, default 10
        Number of lags to perform the test.

    Returns
    -------
    Dict[str, Any]
        Stats for the specified lag.

    Warnings
    --------
    ArchEffectsNotFoundWarning
        If the p-value for the Ljung-Box test is > 0.05 (failure to detect 
        heteroskedasticity).
    """
    squared_returns = returns**2
    lb_result = acorr_ljungbox(squared_returns, lags=[lags], return_df=True)
    p_val = lb_result.iloc[0]['lb_pvalue']
    stat = lb_result.iloc[0]['lb_stat']

    if p_val > 0.05:
        import warnings
        from exceptions import ArchEffectsNotFoundWarning
        warnings.warn(
            f"No significant ARCH effects detected (p-value: {p_val:.4f} > 0.05). "
            "GARCH modeling may not be appropriate.",
            ArchEffectsNotFoundWarning
        )

    return {
        "lb_stat": stat,
        "lb_pvalue": p_val,
        "arch_effects_detected": p_val <= 0.05
    }

def compute_descriptive_stats(returns: pd.Series) -> Dict[str, Any]:
    """
    Compute descriptive statistics and test for heavy tails (Leptokurtosis).

    Parameters
    ----------
    returns : pd.Series
        Log-returns.

    Returns
    -------
    Dict[str, Any]
        Suite of statistics including mean, variance, skewness, and 
        excess kurtosis.
    """
    mu = np.mean(returns)
    var = np.var(returns)
    sk = skew(returns)
    ku = kurtosis(returns) # scipy returns excess kurtosis by default (K - 3)

    is_leptokurtic = ku > 3

    return {
        "mean": mu,
        "variance": var,
        "skewness": sk,
        "excess_kurtosis": ku,
        "is_leptokurtic": is_leptokurtic,
        "distribution_flag": "leptokurtic" if is_leptokurtic else "standard"
    }

def validate_and_prepare(prices: pd.Series) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Primary orchestrator for GARCH data ingestion and validation.

    Parameters
    ----------
    prices : pd.Series
        Raw price series.

    Returns
    -------
    Tuple[pd.Series, Dict[str, Any]]
        Returns the clean log-return series and a metadata dictionary 
        containing all diagnostic results.
    """
    log_returns = compute_log_returns(prices)
    
    metadata = {}
    metadata["stationarity"] = perform_stationarity_test(log_returns)
    metadata["arch_effects"] = check_arch_effects(log_returns)
    metadata["descriptive_stats"] = compute_descriptive_stats(log_returns)

    return log_returns, metadata

if __name__ == "__main__":
    import yfinance as yf
    
    print("--- GARCH Pipeline: Data Ingestion & Validation ---")
    ticker = "SPY"
    print(f"Fetching data for {ticker}...")
    
    # Fetch more historical data for better statistical power
    data = yf.download(ticker, start="2018-01-01", end="2024-01-01", progress=False)
    
    # Handle possible MultiIndex from yfinance or missing Adj Close
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        # Fallback for multi-index or other structures
        prices = data.iloc[:, 0]
        
    # If it's still a DataFrame (e.g. multi-ticker or multi-index), squeeze to Series
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
        
    try:
        returns, meta = validate_and_prepare(prices)
        
        print(f"\nDiagnostic Report for {ticker}:")
        print("-" * 40)
        print(f"Stationarity (ADF p-value): {meta['stationarity']['p_value']:.6f}")
        print(f"ARCH Effects (Ljung-Box p-value): {meta['arch_effects']['lb_pvalue']:.6f}")
        print(f"Excess Kurtosis: {meta['descriptive_stats']['excess_kurtosis']:.4f}")
        print(f"Distribution Flag: {meta['descriptive_stats']['distribution_flag']}")
        
        if meta['descriptive_stats']['is_leptokurtic']:
            print("\nNOTE: Series exhibits leptokurtosis (heavy tails).")
            print("Consider using Student-t innovations in the GARCH model.")
            
    except NonStationaryDataError as e:
        print(f"\nCRITICAL ERROR: {e}")
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
