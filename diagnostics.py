import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, Any, List, Tuple
from scipy.stats import jarque_bera, norm, probplot
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

from estimation import GARCHResult, fit_garch
from engine import compute_variance_path, unpack_params
from exceptions import ModelMisspecificationWarning

# Setup logging
logger = logging.getLogger(__name__)

class GARCHDiagnostics:
    """
    Post-estimation validation suite for GARCH(p,q) models.
    """

    def __init__(self, result: GARCHResult, returns: np.ndarray):
        """
        Parameters
        ----------
        result : GARCHResult
            Fitted model results.
        returns : np.ndarray
            Original log-return series used for estimation.
        """
        self.result = result
        self.returns = returns
        self.T = len(returns)
        
        # Compute residuals and standardized residuals
        params = unpack_params(result.params, result.p, result.q)
        mu = params['mu']
        self.eps = returns - mu
        
        sigmas_sq = compute_variance_path(returns, result.params, result.p, result.q)
        self.sigmas = np.sqrt(sigmas_sq)
        self.std_residuals = self.eps / self.sigmas

    def test_normality(self) -> Dict[str, Any]:
        """
        Perform Jarque-Bera test for residual normality.

        Mathematical Notes
        ------------------
        JB = (T/6) * [S^2 + (K-3)^2 / 4]
        where S is skewness and K is kurtosis.
        Null Hypothesis (H0): Residuals are normally distributed.

        Returns
        -------
        Dict[str, Any]
            JB statistic, p-value, and pass/fail flag.
        """
        stat, p_val = jarque_bera(self.std_residuals)
        is_normal = p_val > 0.05
        
        return {
            "jb_stat": stat,
            "p_value": p_val,
            "is_normal": is_normal
        }

    def test_arch_lm(self, lags: int = 10) -> Dict[str, Any]:
        """
        Run Ljung-Box on squared standardized residuals (ARCH-LM test).

        Mathematical Notes
        ------------------
        Tests for remaining conditional heteroskedasticity. 
        If significant, the model order (p,q) is likely insufficient.

        Returns
        -------
        Dict[str, Any]
            Ljung-Box stats and misspecification flag.
        """
        res_sq = self.std_residuals**2
        lb_res = acorr_ljungbox(res_sq, lags=[lags], return_df=True)
        p_val = lb_res.iloc[0]['lb_pvalue']
        
        if p_val <= 0.05:
            warnings.warn(
                f"Significant ARCH effects remain in residuals (p={p_val:.4f}). "
                "The GARCH model may be misspecified.", 
                ModelMisspecificationWarning
            )

        return {
            "lb_stat": lb_res.iloc[0]['lb_stat'],
            "p_value": p_val,
            "adequately_specified": p_val > 0.05
        }

    def compute_acf_pacf(self, lags: int = 40) -> Dict[str, Any]:
        """
        Compute ACF and PACF with Bartlett's 95% confidence bands.

        Mathematical Notes
        ------------------
        Confidence Bands: ±1.96 / sqrt(T)
        Source: Bartlett (1946).

        Returns
        -------
        Dict[str, Any]
            ACF, PACF values and list of significant lags.
        """
        acf_vals = acf(self.std_residuals, nlags=lags, fft=True)
        pacf_vals = pacf(self.std_residuals, nlags=lags)
        
        band = 1.96 / np.sqrt(self.T)
        
        sig_acf = np.where(np.abs(acf_vals[1:]) > band)[0] + 1
        sig_pacf = np.where(np.abs(pacf_vals[1:]) > band)[0] + 1
        
        return {
            "acf": acf_vals,
            "pacf": pacf_vals,
            "conf_band": band,
            "sig_acf_lags": sig_acf.tolist(),
            "sig_pacf_lags": sig_pacf.tolist()
        }

    def get_qq_plot_data(self) -> Dict[str, np.ndarray]:
        """
        Prepare data for a QQ-plot (Theoretical vs Empirical Quantiles).

        Returns
        -------
        Dict[str, np.ndarray]
            'theoretical_quantiles', 'sample_quantiles'.
        """
        (osm, osr), (slope, intercept, r) = probplot(self.std_residuals, dist="norm")
        return {
            "theoretical_quantiles": osm,
            "sample_quantiles": osr
        }

    def run_all(self) -> Dict[str, Any]:
        """
        Execute all diagnostics and return a summary report.
        """
        report = {
            "normality": self.test_normality(),
            "arch_effects": self.test_arch_lm(),
            "autocorrelation": self.compute_acf_pacf(),
            "standardized_residuals_stats": {
                "mean": np.mean(self.std_residuals),
                "std": np.std(self.std_residuals)
            }
        }
        
        # Log summary
        if not report['normality']['is_normal']:
            logger.warning("Diagnostics: Residuals fail normality test (Jarque-Bera).")
        if not report['arch_effects']['adequately_specified']:
            logger.warning("Diagnostics: Significant ARCH effects remain (likely misspecified).")
            
        return report

def select_order(returns: np.ndarray, p_max: int = 3, q_max: int = 3) -> pd.DataFrame:
    """
    Grid search for optimal GARCH(p,q) order based on AIC and BIC.

    Parameters
    ----------
    returns : np.ndarray
        Log-returns.
    p_max, q_max : int
        Maximum orders to test.

    Returns
    -------
    pd.DataFrame
        Table of p, q, AIC, BIC.
    """
    results = []
    
    print(f"Starting grid search for (p,q) up to ({p_max}, {q_max})...")
    
    for p in range(1, p_max + 1):
        for q in range(1, q_max + 1):
            try:
                # Use fewer starts for speed in grid search
                res = fit_garch(returns, p, q, n_starts=3)
                results.append({
                    "p": p,
                    "q": q,
                    "AIC": res.aic,
                    "BIC": res.bic,
                    "LL": res.log_likelihood
                })
            except Exception as e:
                logger.error(f"Failed to fit GARCH({p},{q}): {e}")
                
    df = pd.DataFrame(results)
    
    # Identify optima
    best_aic = df.loc[df['AIC'].idxmin()]
    best_bic = df.loc[df['BIC'].idxmin()]
    
    print(f"\nOrder Selection Summary:")
    print(f"  Best AIC: GARCH({int(best_aic['p'])},{int(best_aic['q'])})")
    print(f"  Best BIC: GARCH({int(best_bic['p'])},{int(best_bic['q'])})")
    
    return df

if __name__ == "__main__":
    from ingestion import validate_and_prepare
    import yfinance as yf
    
    # Silence yfinance and scipy logs
    logging.getLogger('yfinance').setLevel(logging.ERROR)
    
    print("--- GARCH Pipeline: Diagnostics & Validation ---")
    ticker = "SPY"
    data = yf.download(ticker, start="2018-01-01", end="2024-01-01", progress=False)
    
    # Extract prices
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    if isinstance(prices, pd.DataFrame): prices = prices.squeeze()
        
    returns, _ = validate_and_prepare(prices)
    ret_arr = returns.values.astype(np.float64)
    
    # 1. Fit Baseline GARCH(1,1)
    print("\nFitting baseline GARCH(1,1)...")
    res_11 = fit_garch(ret_arr, 1, 1)
    
    # 2. Run Diagnostics
    print("\nRunning Model Diagnostics...")
    diag = GARCHDiagnostics(res_11, ret_arr)
    report = diag.run_all()
    
    print("-" * 30)
    print(f"Normality (JB p-value):    {report['normality']['p_value']:.6f}")
    print(f"ARCH-LM (LB p-value):      {report['arch_effects']['p_value']:.6f}")
    print(f"Residual Mean:             {report['standardized_residuals_stats']['mean']:.6f}")
    print(f"Residual Std:              {report['standardized_residuals_stats']['std']:.6f}")
    
    sig_lags = report['autocorrelation']['sig_acf_lags']
    if sig_lags:
        print(f"Significant ACF Lags:      {sig_lags}")
    else:
        print("No significant autocorrelation in standardized residuals.")
        
    # 3. Grid Search for Order Selection
    print("\nRunning Order Selection grid search...")
    grid = select_order(ret_arr, p_max=2, q_max=2)
    print("\nGrid Results:")
    print(grid.sort_values("BIC"))
