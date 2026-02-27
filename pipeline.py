"""
GARCH Production Modeling Pipeline
==================================
Version: 1.0.0
Author: Senior Quantitative Developer

Architecture:
- Ingestion: Log-return computation and stationarity testing.
- Engine: Vectorized log-likelihood and variance recursion.
- Estimation: Multi-start MLE with robust Hessian uncertainty.
- Diagnostics: Post-estimation validation suite (JB, ARCH-LM).
- Forecasting: Analytical multi-step and Monte Carlo intervals.
- Backtesting: Walk-forward expanding window validation.

This module provides a unified interface for the entire research-to-production flow.
"""

import os
import sys
import json
import logging
import argparse
import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Internal Imports
from config import load_config, GARCHConfig
from ingestion import validate_and_prepare
from estimation import fit_garch, GARCHResult
from diagnostics import GARCHDiagnostics, select_order
from forecasting import forecast, GARCHForecast
from backtesting import walk_forward_backtest, BacktestResult
import exceptions

# VERSION
__version__ = "1.0.0"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GARCHPipeline")

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for NumPy types and Dataclasses. """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return super(NumpyEncoder, self).default(obj)

@dataclass
class PipelineReport:
    """
    Unified container for GARCH pipeline artifacts.
    """
    ticker: str
    p: int
    q: int
    fit_result: GARCHResult
    diagnostics: Dict[str, Any]
    forecast: GARCHForecast
    backtest: Optional[BacktestResult]
    duration_seconds: float

    def to_json(self, indent: int = 4) -> str:
        """ Serializes the entire pipeline output to a JSON string. """
        return json.dumps(asdict(self), cls=NumpyEncoder, indent=indent)

    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """ Converts components into a dictionary of DataFrames. """
        dfs = {}
        
        # Fit Results
        params = self.fit_result.params
        se = self.fit_result.std_errors
        dfs["params"] = pd.DataFrame({
            "value": params,
            "std_error": se,
            "z_stat": self.fit_result.z_stats,
            "p_value": self.fit_result.p_values
        })
        
        # Forecast
        # Combining analytical and MC intervals
        h = self.forecast.horizon
        dfs["forecast"] = pd.DataFrame({
            "horizon": np.arange(1, h + 1),
            "variance": self.forecast.point_forecast,
            "vol_annual": self.forecast.annualized_vol(),
            "lower_1pct": self.forecast.lower_1pct,
            "upper_99pct": self.forecast.upper_99pct
        })
        
        if self.backtest:
            dfs["backtest_oos"] = pd.DataFrame({
                "realized_var": self.backtest.realized_variance,
                "forecasted_var": self.backtest.forecasted_variance,
                "returns": self.backtest.returns
            })
            
        return dfs

class GARCHPipeline:
    """
    Orchestrator for the end-to-end GARCH pipeline.
    """
    
    def __init__(self, config: Optional[GARCHConfig] = None):
        self.config = config or load_config()

    def run(self, prices: pd.Series, p: Optional[int] = None, q: Optional[int] = None, 
            run_backtest: bool = True) -> PipelineReport:
        """
        Executes the full pipeline.
        """
        start_time = time.time()
        p = p or self.config.p
        q = q or self.config.q
        ticker = getattr(prices, 'name', self.config.ticker) or "Unknown"

        logger.info(f"Starting GARCH({p},{q}) Pipeline for {ticker}")

        # 1. Ingestion
        t_start = time.time()
        returns, metadata = validate_and_prepare(prices)
        ret_arr = returns.values.astype(np.float64)
        logger.info(f"Ingestion complete in {time.time()-t_start:.4f}s")

        # 2. Estimation
        t_start = time.time()
        logger.info(f"Fitting GARCH({p},{q}) model...")
        fit_res = fit_garch(ret_arr, p, q)
        logger.info(f"Estimation complete in {time.time()-t_start:.4f}s")

        # 3. Diagnostics
        t_start = time.time()
        diag = GARCHDiagnostics(fit_res, ret_arr)
        diag_report = diag.run_all()
        logger.info(f"Diagnostics complete in {time.time()-t_start:.4f}s")

        # 4. Forecasting
        t_start = time.time()
        logger.info(f"Generating {self.config.horizon}-day forecast...")
        fc = forecast(fit_res, ret_arr, horizon=self.config.horizon, 
                     n_simulations=self.config.n_simulations)
        logger.info(f"Forecasting complete in {time.time()-t_start:.4f}s")

        # 5. Backtesting (Optional)
        bt_res = None
        if run_backtest:
            t_start = time.time()
            logger.info("Starting walk-forward backtest (expanding window)...")
            bt_res = walk_forward_backtest(ret_arr, p, q, min_train=self.config.min_train_window)
            logger.info(f"Backtesting complete in {time.time()-t_start:.4f}s")

        total_duration = time.time() - start_time
        logger.info(f"Pipeline finished successfully in {total_duration:.2f}s")

        return PipelineReport(
            ticker=ticker,
            p=p,
            q=q,
            fit_result=fit_res,
            diagnostics=diag_report,
            forecast=fc,
            backtest=bt_res,
            duration_seconds=total_duration
        )

def main():
    parser = argparse.ArgumentParser(description="Unified GARCH Production Pipeline CLI")
    parser.add_argument("--ticker", type=str, default=None, help="Asset ticker")
    parser.add_argument("--p", type=int, default=None, help="GARCH order")
    parser.add_argument("--q", type=int, default=None, help="ARCH order")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon")
    parser.add_argument("--select-order", action="store_true", help="Run AIC/BIC grid search first")
    parser.add_argument("--no-backtest", action="store_true", help="Skip backtesting")
    parser.add_argument("--output", type=str, default="results.json", help="Output file path")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG, INFO, WARNING, ERROR")
    
    args = parser.parse_args()
    
    # Configure logging level
    logger.parent.setLevel(getattr(logging, args.log_level.upper()))
    
    try:
        # 1. Load Config
        config = load_config(args.config)
        # Override with CLI args if provided
        if args.ticker: config.ticker = args.ticker
        if args.p: config.p = args.p
        if args.q: config.q = args.q
        if args.horizon: config.horizon = args.horizon

        # 2. Fetch Data
        import yfinance as yf
        logger.info(f"Fetching data for {config.ticker} from {config.start_date} to {config.end_date}...")
        data = yf.download(config.ticker, start=config.start_date, end=config.end_date, progress=False)
        if data.empty:
            raise ValueError(f"No data found for ticker {config.ticker}")
        
        prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
        if isinstance(prices, pd.DataFrame): prices = prices.squeeze()
        prices.name = config.ticker

        # 3. Order Selection (Optional)
        p, q = config.p, config.q
        if args.select_order:
            logger.info("Executing AIC/BIC order selection...")
            from ingestion import validate_and_prepare as pre
            rets, _ = pre(prices)
            grid = select_order(rets.values, p_max=config.p_max, q_max=config.q_max)
            best_bic = grid.loc[grid['BIC'].idxmin()]
            p, q = int(best_bic['p']), int(best_bic['q'])
            logger.info(f"Selected optimal order: GARCH({p},{q}) via BIC")

        # 4. Orchestrate Pipeline
        pipeline = GARCHPipeline(config)
        report = pipeline.run(prices, p=p, q=q, run_backtest=not args.no_backtest)

        # 5. Save Results
        with open(args.output, "w") as f:
            f.write(report.to_json())
        logger.info(f"Full report saved to {args.output}")

    except Exception:
        logger.exception("FATAL: Pipeline failed encountered a critical error.")
        sys.exit(1)

if __name__ == "__main__":
    main()
