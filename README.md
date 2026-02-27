# GARCH Volatility Modeling Pipeline

A production-grade Python pipeline for estimating, diagnosing, forecasting, and backtesting Generalized Autoregressive Conditional Heteroskedasticity (GARCH) models. Designed to meet Tier-1 asset management standards for risk modeling and quantitative research.


1. **Robust Estimation:** Implements Multi-Start Maximum Likelihood Estimation (MLE) using Dirichlet sampling for initial parameters to avoid local minima, enforcing strict covariance stationarity constraints ($\sum\alpha + \sum\beta < 1$). It includes stabilized finite-difference Hessian approximations for reliable standard errors.
2. **Data Pipeline Integrity:** Strictly validates assumptions via ADF (stationarity) and Ljung-Box (ARCH effects) tests *before* fitting, raising custom exceptions on invalid data.
3. **Rigorous Diagnostics:** Models are not just fit, but statistically audited post-estimation via Jarque-Bera (normality), ARCH-LM (remaining heteroskedasticity), and ACF/PACF bounds.
4. **Walk-Forward Validation:** Provides an expanding-window backtesting engine with specialized tests for Value-at-Risk (VaR) compliance, including Kupiec's POF (frequency) and Christoffersen's (independence/clustering) tests.
5. **System Architecture:** Fully orchestrated via a unified `GARCHPipeline` class with an `argparse` CLI, Pydantic-validated YAML configurations, comprehensive logging, and robust exception handling.

## System Architecture

The project is modularized into the following quantitative layers:

* **`ingestion.py`**: Validates raw price data, computes log-returns, tests for unit roots (ADF), and checks for ARCH effects (Ljung-Box).
* **`engine.py`**: The mathematical core. Implements vectorized conditional variance recursion and the Gaussian log-likelihood objective function.
* **`estimation.py`**: The optimizer. Executes SLSQP multi-start MLE, handles parameter bounds, computes standard errors via the Hessian, and calculates AIC/BIC.
* **`diagnostics.py`**: Post-estimation auditor. Checks standardized residuals for normality and independence. Includes grid search for optimal $(p,q)$ order selection.
* **`forecasting.py`**: Generates analytical multi-step $E[\sigma^2_{T+h}]$ forecasts converging to the unconditional variance, and utilizes high-performance vectorized Monte Carlo simulations for non-linear confidence intervals.
* **`backtesting.py`**: Implements walk-forward (expanding window) out-of-sample backtesting, calculating Value-at-Risk (VaR) limit breaches, QLIKE, and MSE metrics.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd GARCH

# Install strict dependencies
pip install -r requirements.txt
```

## Configuration

Configuration is managed via `config.yaml` and validated strictly by Pydantic (`config.py`).

```yaml
ticker: "SPY"
start_date: "2018-01-01"
end_date: "2024-01-01"
p: 1
q: 1
horizon: 30
n_simulations: 10000
min_train_window: 500
p_max: 2
q_max: 2
```

## CLI Usage

The system exposes a unified command-line orchestrator via `pipeline.py`.

```bash
# Run the pipeline with default YAML configuration
python pipeline.py

# Override settings via CLI arguments
python pipeline.py --ticker AAPL --p 1 --q 2 --horizon 60 --output aapl_results.json

# Run automatic AIC/BIC grid search for optimal order selection before fitting
python pipeline.py --select-order

# Disable the walk-forward backtest (for faster pure forecasting)
python pipeline.py --no-backtest
```

## Output

The pipeline generates a `PipelineReport` exported as JSON. This includes:
* Estimated parameters, standard errors, z-stats, and p-values.
* Diagnostic test values (ADF, LB, JB) and pass/fail booleans.
* Analytical point forecasts and Monte Carlo confidence intervals (1%, 5%, 95%, 99%).
* Walk-forward out-of-sample performance metrics (QLIKE, Kupiec POF, Christoffersen).

## Testing

A `pytest` smoke test suite validates mathematical correctness and module integration.

```bash
pytest test_garch_smoke.py -v
```
