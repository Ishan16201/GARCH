import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from exceptions import ConstraintViolationError

def unpack_params(theta: np.ndarray, p: int, q: int) -> Dict[str, Any]:
    """
    Extract and validate GARCH(p,q) parameters from the parameter vector θ.

    Mathematical Notes
    ------------------
    Constraints for GARCH(p,q):
    1. Positivity: ω > 0, α_i ≥ 0, β_j ≥ 0.
       Required to ensure σ²_t > 0 for all t.
    2. Covariance Stationarity: Σ(α_i) + Σ(β_j) < 1.
       Required for the unconditional variance σ² = ω / (1 - Σα - Σβ) to be 
       finite and positive. If Σα + Σβ ≥ 1, the variance process is 
       non-stationary (Integrated GARCH or worse), leading to explosive 
       volatility.

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector [μ, ω, α_1...α_q, β_1...β_p].
    p : int
        Order of GARCH terms (σ²).
    q : int
        Order of ARCH terms (ε²).

    Returns
    -------
    Dict[str, Any]
        Dictionary of named parameters.

    Raises
    ------
    ConstraintViolationError
        If any positivity or stationarity constraints are breached.
    """
    expected_len = 1 + 1 + q + p # mu + omega + alphas + betas
    if len(theta) != expected_len:
        raise ValueError(f"Expected theta of length {expected_len}, got {len(theta)}")

    mu = theta[0]
    omega = theta[1]
    alphas = theta[2:2+q]
    betas = theta[2+q:]

    # Positivity Checks
    if omega <= 0:
        raise ConstraintViolationError(f"Positivity violated: ω ({omega}) must be > 0")
    if np.any(alphas < 0):
        idx = np.where(alphas < 0)[0][0]
        raise ConstraintViolationError(f"Positivity violated: α_{idx+1} ({alphas[idx]}) < 0")
    if np.any(betas < 0):
        idx = np.where(betas < 0)[0][0]
        raise ConstraintViolationError(f"Positivity violated: β_{idx+1} ({betas[idx]}) < 0")

    # Stationarity Check
    persistence = np.sum(alphas) + np.sum(betas)
    if persistence >= 1.0:
        raise ConstraintViolationError(
            f"Stationarity violated: Σα + Σβ ({persistence:.4f}) must be < 1. "
            "The variance process is non-stationary."
        )

    return {
        "mu": mu,
        "omega": omega,
        "alphas": alphas,
        "betas": betas,
        "persistence": persistence
    }

def compute_variance_path(returns: np.ndarray, theta: np.ndarray, p: int, q: int) -> np.ndarray:
    """
    Compute the conditional variance path σ²_t using GARCH(p,q) recursion.

    Mathematical Notes
    ------------------
    Recursion:
    σ²_t = ω + Σ(i=1..q) α_i * ε²_{t-i} + Σ(j=1..p) β_j * σ²_{t-j}
    where ε_t = r_t - μ.

    Initialization:
    σ²_0 is set to the long-run (unconditional) variance:
    σ² = ω / (1 - Σα - Σβ)

    Parameters
    ----------
    returns : np.ndarray
        Array of log-returns (float64).
    theta : np.ndarray
        Parameter vector.
    p : int
        GARCH order.
    q : int
        ARCH order.

    Returns
    -------
    np.ndarray
        Conditional variances σ²_t, same length as returns.
    """
    T = len(returns)
    params = unpack_params(theta, p, q)
    mu, omega = params['mu'], params['omega']
    alphas, betas = params['alphas'], params['betas']
    
    eps_sql = (returns - mu)**2
    variances = np.zeros(T, dtype=np.float64)
    
    # Initialize with unconditional variance
    uncond_var = omega / (1 - params['persistence'])
    
    # We need a buffer for past residuals and variances to handle lags p, q
    # For t < max(p, q), we use uncond_var for those unknown past values
    for t in range(T):
        # ARCH part: Σ alpha_i * eps^2_{t-i}
        arch_sum = 0.0
        for i in range(1, q + 1):
            if t - i < 0:
                arch_sum += alphas[i-1] * uncond_var
            else:
                arch_sum += alphas[i-1] * eps_sql[t-i]
        
        # GARCH part: Σ beta_j * sigma^2_{t-j}
        garch_sum = 0.0
        for j in range(1, p + 1):
            if t - j < 0:
                garch_sum += betas[j-1] * uncond_var
            else:
                garch_sum += betas[j-1] * variances[t-j]
                
        variances[t] = omega + arch_sum + garch_sum
        
    return variances

def gaussian_log_likelihood(theta: np.ndarray, returns: np.ndarray, p: int, q: int) -> float:
    """
    Compute the negative log-likelihood of a GARCH(p,q) model under Gaussian 
    innovations.

    Mathematical Notes
    ------------------
    The log-likelihood for observation t is:
    ℓ_t = -0.5 * [ ln(2π) + ln(σ²_t) + (r_t - μ)² / σ²_t ]
    
    Full Log-Likelihood:
    ℓ(θ) = Σ_{t=1}^T ℓ_t

    Parameters
    ----------
    theta : np.ndarray
        Parameters to evaluate.
    returns : np.ndarray
        Data series.
    p : int, q : int
        Model orders.

    Returns
    -------
    float
        Negative log-likelihood (NLL). Returns np.inf for invalid parameters.
    """
    try:
        variances = compute_variance_path(returns, theta, p, q)
    except ConstraintViolationError:
        return np.inf
    
    if np.any(variances <= 0):
        return np.inf
        
    mu = theta[0]
    eps_sql = (returns - mu)**2
    
    # Log-likelihood terms
    # Constant term: -0.5 * ln(2*pi) * T
    # Variance term: -0.5 * Σ ln(sigma^2_t)
    # Residual term: -0.5 * Σ (eps^2_t / sigma^2_t)
    
    T = len(returns)
    log_2pi = np.log(2.0 * np.pi)
    
    ll = -0.5 * (T * log_2pi + np.sum(np.log(variances)) + np.sum(eps_sql / variances))
    
    return -ll # Return Negative LL for minimization

def numerical_gradient_check(returns: np.ndarray, theta: np.ndarray, p: int, q: int, eps: float = 1e-6) -> np.ndarray:
    """
    Compute the numerical gradient of the NLL using finite differences.

    Parameters
    ----------
    returns : np.ndarray
        Data.
    theta : np.ndarray
        Initial parameters.
    p, q : int
        Orders.
    eps : float, default 1e-6
        Perturbation size.

    Returns
    -------
    np.ndarray
        Gradient vector.
    """
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        
        theta_plus[i] += eps
        theta_minus[i] -= eps
        
        lp = gaussian_log_likelihood(theta_plus, returns, p, q)
        lm = gaussian_log_likelihood(theta_minus, returns, p, q)
        
        grad[i] = (lp - lm) / (2 * eps)
        
    return grad

if __name__ == "__main__":
    import timeit
    
    print("--- GARCH Engine: Likelihood & Verification ---")
    
    # Generate synthetic returns
    np.random.seed(42)
    T = 10000
    # Simulate a GARCH(1,1) like process
    # theta = [mu, omega, alpha, beta]
    true_theta = np.array([0.0001, 0.1, 0.1, 0.8], dtype=np.float64)
    p, q = 1, 1
    
    # Simple simulation for testing
    sim_returns = np.random.normal(0, 1, T) * 0.5
    
    # Verify parameter unpacking
    print("\n1. Verifying Parameter Unpacking...")
    try:
        unpacked = unpack_params(true_theta, p, q)
        print(f"Unpacked: {unpacked}")
    except Exception as e:
        print(f"Unpacking Error: {e}")
        
    # Benchmark variance path
    print(f"\n2. Benchmarking Variance Path Recursion (T={T})...")
    time_taken = timeit.timeit(lambda: compute_variance_path(sim_returns, true_theta, p, q), number=10)
    print(f"Average execution time: {time_taken/10:.6f} seconds")
    
    # Calculate NLL
    print("\n3. Calculating Gaussian Log-Likelihood...")
    nll = gaussian_log_likelihood(true_theta, sim_returns, p, q)
    print(f"NLL (Negative Log-Likelihood): {nll:.4f}")
    
    # Gradient Check
    print("\n4. Numerical Gradient Check...")
    grad = numerical_gradient_check(sim_returns, true_theta, p, q)
    param_names = ['mu', 'omega'] + [f'alpha_{i+1}' for i in range(q)] + [f'beta_{j+1}' for j in range(p)]
    for name, g in zip(param_names, grad):
        print(f"  ∂ℓ/∂{name}: {g:.8f}")

    # Stress test with T=100,000
    T_stress = 100000
    print(f"\n5. Stress Test (T={T_stress})...")
    stress_returns = np.random.normal(0, 1, T_stress)
    time_stress = timeit.timeit(lambda: gaussian_log_likelihood(true_theta, stress_returns, p, q), number=1)
    print(f"T=100k NLL calculation time: {time_stress:.4f} seconds")
