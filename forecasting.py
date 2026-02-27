import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, List, Optional
import timeit

from estimation import GARCHResult
from engine import compute_variance_path, unpack_params

# Setup logging
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class GARCHForecast:
    """
    Container for GARCH multi-step forecasts and confidence intervals.
    """
    horizon: int
    point_forecast: np.ndarray # Analytical expectations
    lower_1pct: np.ndarray
    lower_5pct: np.ndarray
    upper_5pct: np.ndarray
    upper_99pct: np.ndarray
    unconditional_variance: float
    convergence_horizon: int # Horizon h where forecast is within 1% of unc. var.

    def annualized_vol(self, trading_days: int = 252) -> np.ndarray:
        """
        Compute annualized volatility from the daily point forecast.

        Mathematical Notes
        ------------------
        σ_annual = σ_daily * √trading_days
        Assumption: Daily returns are iid with constant variance σ²_daily.
        Limitation: This ignores the time-varying nature of variance within the 
        annualization window (e.g., mean reversion to unconditional variance).

        Parameters
        ----------
        trading_days : int, default 252
            Number of trading days per year.

        Returns
        -------
        np.ndarray
            Annualized standard deviations.
        """
        return np.sqrt(self.point_forecast * trading_days)

def forecast(result: GARCHResult, returns: np.ndarray, horizon: int, 
             n_simulations: int = 10000, seed: int = 42) -> GARCHForecast:
    """
    Compute multi-step ahead forecasts for GARCH(p,q) process.

    Mathematical Notes
    ------------------
    1. One-step-ahead variance:
       σ²_{T+1} = ω + Σ αᵢ ε²_{t+1-i} + Σ βⱼ σ²_{T+1-j}
       where ε_t and σ²_t are known at time T.

    2. h-step-ahead Analytical Forecast (Derivation):
       Let P = Σ αᵢ + Σ βⱼ (Persistence).
       By Law of Iterated Expectations (LIE):
       E_T[σ²_{T+h}] = ω + Σ αᵢ E_T[ε²_{T+h-i}] + Σ βⱼ E_T[σ²_{T+h-j}]
       Since E_T[ε²_{T+k}] = E_T[σ²_{T+k}] for k > 0, we have (for GARCH(1,1)):
       E_T[σ²_{T+h}] = ω + (α + β) E_T[σ²_{T+h-1}]
       Recursively:
       E_T[σ²_{T+h}] - σ² = (α + β) (E_T[σ²_{T+h-1}] - σ²)
       where σ² = ω / (1 - P).
       Thus: E_T[σ²_{T+h}] = σ² + P^{h-1} (σ²_{T+1} - σ²)
       As h → ∞, P^{h-1} → 0 (since P < 1), E_T[σ²_{T+h}] → σ².

    Parameters
    ----------
    result : GARCHResult
        Fitted model parameters.
    returns : np.ndarray
        Historical returns.
    horizon : int
        Number of steps to forecast.
    n_simulations : int
        Number of Monte Carlo paths.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    GARCHForecast
        Dataclass with projections and intervals.
    """
    params = unpack_params(result.params, result.p, result.q)
    mu = params['mu']
    omega = params['omega']
    alphas = params['alphas']
    betas = params['betas']
    persistence = params['persistence']
    
    # Unconditional variance
    uncond_var = omega / (1 - persistence)
    
    # 1. Compute sigma^2_{T+1}
    # We need the full path to get the last values
    full_variances = compute_variance_path(returns, result.params, result.p, result.q)
    eps_sql = (returns - mu)**2
    
    # σ²_{T+1} = ω + Σ α_i * ε²_{T+1-i} + Σ β_j * σ²_{T+1-j}
    v_next = omega
    for i in range(1, result.q + 1):
        v_next += alphas[i-1] * eps_sql[-i]
    for j in range(1, result.p + 1):
        v_next += betas[j-1] * full_variances[-j]
        
    # 2. Analytical Projections
    point_forecast = np.zeros(horizon)
    point_forecast[0] = v_next
    for h in range(1, horizon):
        point_forecast[h] = uncond_var + (persistence**h) * (v_next - uncond_var)
        
    # Convergence Horizon
    diff = np.abs(point_forecast - uncond_var)
    threshold = 0.01 * uncond_var
    conv_indices = np.where(diff < threshold)[0]
    convergence_h = conv_indices[0] + 1 if len(conv_indices) > 0 else horizon
    
    # 3. Monte Carlo Simulation (Vectorized over paths)
    rng = np.random.default_rng(seed)
    # Shape (N, horizon)
    sim_variances = np.zeros((n_simulations, horizon))
    sim_eps_sql = np.zeros((n_simulations, horizon))
    
    # Initialize with historical lags needed for p, q
    # For simplicity, we assume GARCH(1,1) logic for the loop but make it generalizable
    # by keeping state of last q eps^2 and last p variances
    
    curr_v = np.full(n_simulations, v_next)
    
    for h in range(horizon):
        if h == 0:
            sim_variances[:, h] = v_next
        else:
            # v_t = ω + α ε²_{t-1} + β v_{t-1}
            # This is specifically for GARCH(1,1). For p,q > 1, we'd need more buffers.
            # Enforcing p=1, q=1 for the vectorized MC demo or implementing sliding window
            
            # General p,q implementation:
            v_t = np.full(n_simulations, omega)
            for i in range(1, result.q + 1):
                # if h-i < 0, use historical eps^2
                if h - i < 0:
                    val = eps_sql[h-i]
                    v_t += alphas[i-1] * val
                else:
                    v_t += alphas[i-1] * sim_eps_sql[:, h-i]
                    
            for j in range(1, result.p + 1):
                # if h-j < 0, use historical variance
                if h - j < 0:
                    val = full_variances[h-j]
                    v_t += betas[j-1] * val
                else:
                    v_t += betas[j-1] * sim_variances[:, h-j]
            
            sim_variances[:, h] = v_t
            
        # Draw innovations for next step
        z = rng.standard_normal(n_simulations)
        sim_eps_sql[:, h] = sim_variances[:, h] * (z**2)
        
    # 4. Extract Quantiles
    l1 = np.percentile(sim_variances, 1, axis=0)
    l5 = np.percentile(sim_variances, 5, axis=0)
    u5 = np.percentile(sim_variances, 95, axis=0)
    u99 = np.percentile(sim_variances, 99, axis=0)
    
    return GARCHForecast(
        horizon=horizon,
        point_forecast=point_forecast,
        lower_1pct=l1,
        lower_5pct=l5,
        upper_5pct=u5,
        upper_99pct=u99,
        unconditional_variance=uncond_var,
        convergence_horizon=convergence_h
    )

if __name__ == "__main__":
    from ingestion import validate_and_prepare
    from estimation import fit_garch
    import yfinance as yf
    
    print("--- GARCH Engine: Forecasting & Simulation ---")
    ticker = "SPY"
    data = yf.download(ticker, start="2018-01-01", end="2024-01-01", progress=False)
    prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
    if isinstance(prices, pd.DataFrame): prices = prices.squeeze()
        
    returns, _ = validate_and_prepare(prices)
    ret_arr = returns.values.astype(np.float64)
    
    # 1. Fit GARCH(1,1)
    print("Fitting model...")
    res = fit_garch(ret_arr, 1, 1)
    
    # 2. Generate Forecasts
    horizon = 30
    print(f"Generating {horizon}-day forecast...")
    
    # Benchmark MC
    def run_fc():
        return forecast(res, ret_arr, horizon=252, n_simulations=10000)
    
    t = timeit.timeit(run_fc, number=1)
    print(f"T=10,000 simulations on 252-day horizon took: {t:.4f} seconds")
    
    fc = forecast(res, ret_arr, horizon=horizon)
    
    print(f"\nForecast Summary for {ticker}:")
    print("-" * 40)
    print(f"Unconditional Variance:  {fc.unconditional_variance:.8f}")
    print(f"T+1 Variance Forecast:   {fc.point_forecast[0]:.8f}")
    print(f"T+{horizon} Variance Forecast:  {fc.point_forecast[-1]:.8f}")
    print(f"Convergence Horizon:     {fc.convergence_horizon} days")
    
    # Verifying convergence assertion
    assert np.abs(fc.point_forecast[-1] - fc.unconditional_variance) < np.abs(fc.point_forecast[0] - fc.unconditional_variance), \
        "Forecast should move towards unconditional variance."
    
    print("\nInitial Annualized Volatility (T+1):")
    print(f"  {fc.annualized_vol()[0]:.2%}")
    print(f"Long-run Annualized Volatility:")
    print(f"  {np.sqrt(fc.unconditional_variance * 252):.2%}")
    
    print("\nMonte Carlo 95% Confidence Interval (T+30):")
    print(f"  [{np.sqrt(fc.lower_5pct[-1]*252):.2%}, {np.sqrt(fc.upper_5pct[-1]*252):.2%}]")
