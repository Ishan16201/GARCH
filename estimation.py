import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List, Optional
from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.stats import norm

from engine import gaussian_log_likelihood, unpack_params, compute_variance_path
from exceptions import ConvergenceFailedError, SingularHessianWarning

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class GARCHResult:
    """
    Immutable container for GARCH model estimation results.
    """
    p: int
    q: int
    params: np.ndarray
    param_names: List[str]
    std_errors: np.ndarray
    z_stats: np.ndarray
    p_values: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    convergence_status: bool
    n_iterations: int
    hessian_eigenvalues: np.ndarray

    def __repr__(self) -> str:
        """
        Return a formatted table of results similar to statsmodels output.
        """
        T = len(self.param_names)
        header = f"{'Parameter':<12} | {'Value':>12} | {'Std.Error':>12} | {'z-stat':>12} | {'p-value':>12}"
        sep = "-" * len(header)
        lines = [sep, header, sep]
        
        for i in range(T):
            lines.append(
                f"{self.param_names[i]:<12} | {self.params[i]:>12.6f} | "
                f"{self.std_errors[i]:>12.6f} | {self.z_stats[i]:>12.4f} | "
                f"{self.p_values[i]:>12.4f}"
            )
        
        infos = [
            sep,
            f"Log-Likelihood: {self.log_likelihood:.4f}",
            f"AIC:            {self.aic:.4f}",
            f"BIC:            {self.bic:.4f}",
            f"Converged:      {self.convergence_status}",
            f"Iterations:     {self.n_iterations}",
            sep
        ]
        return "\n".join(lines + infos)

def compute_hessian(func, x, args, eps=1e-4) -> np.ndarray:
    """
    Compute the Hessian matrix using finite differences (central difference).
    Improved stability for GARCH likelihoods.
    """
    n = len(x)
    hessian = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                # Diagonal: (f(x+h) - 2f(x) + f(x-h)) / h^2
                x_p = x.copy(); x_p[i] += eps
                x_m = x.copy(); x_m[i] -= eps
                
                f_p = func(x_p, *args)
                f_m = func(x_m, *args)
                f_x = func(x, *args)
                
                # Handle infs from constraints
                if not (np.isfinite(f_p) and np.isfinite(f_m) and np.isfinite(f_x)):
                    hessian[i, i] = np.nan
                else:
                    hessian[i, i] = (f_p - 2*f_x + f_m) / (eps**2)
            else:
                # Off-diagonal
                x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
                x_pn = x.copy(); x_pn[i] += eps; x_pn[j] -= eps
                x_np = x.copy(); x_np[i] -= eps; x_np[j] += eps
                x_nn = x.copy(); x_nn[i] -= eps; x_nn[j] -= eps
                
                f_pp = func(x_pp, *args)
                f_pn = func(x_pn, *args)
                f_np = func(x_np, *args)
                f_nn = func(x_nn, *args)
                
                if not all(np.isfinite([f_pp, f_pn, f_np, f_nn])):
                    hessian[i, j] = np.nan
                else:
                    hessian[i, j] = (f_pp - f_pn - f_np + f_nn) / (4 * eps**2)
                hessian[j, i] = hessian[i, j]
                
    return hessian

def generate_starting_points(p: int, q: int, returns: np.ndarray, n: int = 10) -> List[np.ndarray]:
    """
    Generate n starting points using Dirichlet sampling for stationarity.
    """
    starts = []
    mu_init = np.mean(returns)
    vol_init = np.var(returns)
    
    # Add a 'sensible' start based on typical GARCH params
    typical = np.concatenate(([mu_init, vol_init * 0.1], [0.05]*q, [0.85/p]*p))
    starts.append(typical)
    
    for _ in range(n - 1):
        # Dirichlet ensures sum(alphas + betas) < 1
        dirichlet_samples = np.random.dirichlet(np.ones(q + p + 1))
        alphas_betas = dirichlet_samples[:-1] * 0.95 # Keep persistence < 1
        
        # Scale omega to be a fraction of total variance
        omega_init = vol_init * (1 - np.sum(alphas_betas)) * np.random.uniform(0.01, 0.2)
        
        theta = np.concatenate(([mu_init, omega_init], alphas_betas))
        starts.append(theta)
        
    return starts

def multi_start_mle(returns: np.ndarray, p: int, q: int, n_starts: int = 10) -> Tuple[Any, float]:
    """
    Execute multi-start optimization to find global MLE.
    """
    args = (returns, p, q)
    bounds = [(None, None), (1e-6, None)] + [(0, 1)] * (q + p)
    A = np.zeros(1 + 1 + q + p)
    A[2:] = 1.0
    stationarity_constraint = LinearConstraint(A, 0.0, 0.999)
    
    starts = generate_starting_points(p, q, returns, n_starts)
    best_opt = None
    min_nll = np.inf
    
    for i, start in enumerate(starts):
        try:
            res = minimize(
                gaussian_log_likelihood,
                start,
                args=args,
                method='SLSQP',
                bounds=bounds,
                constraints=[stationarity_constraint],
                options={'ftol': 1e-8, 'maxiter': 200}
            )
            
            if res.success and res.fun < min_nll:
                min_nll = res.fun
                best_opt = res
        except Exception as e:
            logger.debug(f"Start {i+1} failed: {e}")
            
    if best_opt is None:
        raise ConvergenceFailedError("MLE estimation failed: No convergence across all starting points.")
        
    return best_opt, min_nll

def fit_garch(returns: np.ndarray, p: int, q: int, n_starts: int = 10) -> GARCHResult:
    """
    Fit a GARCH(p,q) model using Maximum Likelihood Estimation.
    """
    T = len(returns)
    args = (returns, p, q)
    
    # 1. MLE with Multi-Start
    best_opt, nll_star = multi_start_mle(returns, p, q, n_starts)
    theta_star = best_opt.x
    
    # 2. Parameter Uncertainty (Hessian)
    logger.info("Computing Hessian for standard errors...")
    hessian = compute_hessian(gaussian_log_likelihood, theta_star, args)
    
    # Check for NaNs or Infs in Hessian
    if not np.all(np.isfinite(hessian)):
        import warnings
        warnings.warn("Hessian contains non-finite values. Standard errors cannot be computed.", SingularHessianWarning)
        eigvals = np.array([np.nan])
        std_errors = np.array([np.nan] * len(theta_star))
    else:
        try:
            # Check for PSD
            eigvals = np.linalg.eigvalsh(hessian)
            if np.any(eigvals <= 0):
                import warnings
                warnings.warn("Hessian is not positive-definite. Standard errors may be unreliable.", SingularHessianWarning)
            
            # Use pseudo-inverse for near-singular matrices
            cov_matrix = np.linalg.pinv(hessian)
            std_errors = np.sqrt(np.maximum(np.diag(cov_matrix), 0))
        except Exception as e:
            logger.error(f"Hessian analysis failed: {e}")
            eigvals = np.array([np.nan])
            std_errors = np.array([np.nan] * len(theta_star))

    # 3. Diagnostics
    z_stats = theta_star / std_errors
    p_values = 2 * (1 - norm.cdf(np.abs(z_stats)))
    
    k = len(theta_star)
    aic = 2 * k + 2 * nll_star
    bic = k * np.log(T) + 2 * nll_star
    
    param_names = ['mu', 'omega'] + [f'alpha_{i+1}' for i in range(q)] + [f'beta_{j+1}' for j in range(p)]
    
    return GARCHResult(
        p=p,
        q=q,
        params=theta_star,
        param_names=param_names,
        std_errors=std_errors,
        z_stats=z_stats,
        p_values=p_values,
        log_likelihood=-nll_star,
        aic=aic,
        bic=bic,
        convergence_status=best_opt.success,
        n_iterations=best_opt.nit,
        hessian_eigenvalues=eigvals
    )

if __name__ == "__main__":
    from ingestion import validate_and_prepare
    import yfinance as yf
    
    print("--- GARCH Engine: Maximum Likelihood Estimation ---")
    ticker = "SPY"
    print(f"Fetching and validating data for {ticker}...")
    
    data = yf.download(ticker, start="2018-01-01", end="2024-01-01", progress=False)
    # Extract price series robustly
    if 'Adj Close' in data.columns:
        prices = data['Adj Close']
    elif 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data.iloc[:, 0]
    
    # Step 1: Ingestion
    returns, meta = validate_and_prepare(prices)
    
    # Step 2: Estimation GARCH(1,1)
    p, q = 1, 1
    print(f"\nEstimating GARCH({p},{q}) model...")
    
    try:
        result = fit_garch(returns.values.astype(np.float64), p, q)
        print("\nEstimation Complete.")
        print(result)
        
        # Check for heavy distribution recommendation
        if result.p_values[0] > 0.05:
            print("\nAdvice: Mean parameter (mu) is not significant. Consider a zero-mean GARCH if applicable.")
            
    except Exception as e:
        print(f"\nERROR during estimation: {e}")
        import traceback
        traceback.print_exc()
