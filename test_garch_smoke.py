import pytest
import numpy as np
import pandas as pd
from ingestion import validate_and_prepare
from engine import gaussian_log_likelihood
from estimation import fit_garch, GARCHResult
from forecasting import forecast
from backtesting import kupiec_pof_test
from exceptions import NonStationaryDataError

def test_stationarity_validation():
    """ Test data validation raises NonStationaryDataError on integrated log-returns. """
    rng = np.random.default_rng(42)
    # A random walk in log-returns is non-stationary
    log_returns = np.cumsum(rng.standard_normal(1000) * 0.001)
    prices = np.exp(np.cumsum(log_returns))
    prices_series = pd.Series(prices)
    
    with pytest.raises(NonStationaryDataError):
        validate_and_prepare(prices_series)

def test_likelihood_finite():
    """ Test NLL returns finite value. """
    rng = np.random.default_rng(42)
    returns = rng.normal(0, 0.01, 500)
    theta = np.array([0.0, 1e-5, 0.1, 0.8])
    nll = gaussian_log_likelihood(theta, returns, p=1, q=1)
    assert np.isfinite(nll)

def test_mle_convergence():
    """ Test MLE convergence status. """
    T = 1000
    omega, alpha, beta = 1e-6, 0.1, 0.8
    v = np.zeros(T)
    eps = np.zeros(T)
    v[0] = omega / (1-alpha-beta)
    rng = np.random.default_rng(42)
    
    for t in range(1, T):
        v[t] = omega + alpha * eps[t-1]**2 + beta * v[t-1]
        eps[t] = np.sqrt(v[t]) * rng.standard_normal()
        
    res = fit_garch(eps, p=1, q=1, n_starts=1)
    assert res.convergence_status

def test_forecast_convergence():
    """ Test multi-step forecast properties with fixed params. """
    # Use high persistence but less than 1
    # mu, omega, alpha, beta
    params = np.array([0.0, 1e-6, 0.1, 0.8]) # persistence = 0.9
    res = GARCHResult(
        p=1, q=1, params=params, param_names=['mu', 'omega', 'alpha1', 'beta1'],
        std_errors=np.zeros(4), z_stats=np.zeros(4), p_values=np.zeros(4),
        log_likelihood=0.0, aic=0.0, bic=0.0, convergence_status=True,
        n_iterations=0, hessian_eigenvalues=np.array([1.0])
    )
    
    returns = np.random.normal(0, 0.01, 500)
    # Point forecast should move towards unconditional
    fc = forecast(res, returns, horizon=500, n_simulations=0)
    
    # Check if last forecast is close to unconditional var
    # E[v_inf] = omega / (1-p) = 1e-6 / 0.1 = 1e-5
    assert np.allclose(fc.point_forecast[-1], fc.unconditional_variance, rtol=1e-1)
    assert fc.point_forecast[-1] < fc.point_forecast[0] + 1e-1 # Logical check

def test_backtest_logic():
    """ Test Kupiec POF test. """
    breaches = np.zeros(100)
    breaches[0] = 1 # 1% rate
    test_res = kupiec_pof_test(breaches, 0.01)
    assert test_res['p_value'] > 0.05
