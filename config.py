from pydantic import BaseModel, Field, field_validator
from typing import Optional
import yaml
import os

class GARCHConfig(BaseModel):
    """
    Configuration model for GARCH Pipeline with strict validation.
    """
    ticker: str = Field(..., description="The asset ticker symbol (e.g., SPY)")
    start_date: str = Field("2018-01-01", description="Start date for data fetching (YYYY-MM-DD)")
    end_date: str = Field("2024-01-01", description="End date for data fetching (YYYY-MM-DD)")
    
    # Model parameters
    p: int = Field(1, ge=1, le=5, description="GARCH order")
    q: int = Field(1, ge=1, le=5, description="ARCH order")
    
    # Forecasting
    horizon: int = Field(30, ge=1, le=252, description="Forecast horizon in days")
    n_simulations: int = Field(10000, ge=1000, description="Number of Monte Carlo simulations")
    
    # Backtesting
    min_train_window: int = Field(500, ge=252, description="Minimum observations for training")
    
    # Grid Search
    p_max: int = Field(2, ge=1, le=5)
    q_max: int = Field(2, ge=1, le=5)

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_dates(cls, v: str) -> str:
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Dates must be in YYYY-MM-DD format")

def load_config(path: str = "config.yaml") -> GARCHConfig:
    """
    Load and validate configuration from a YAML file.
    """
    if not os.path.exists(path):
        logger_warn = "Config file not found. Using defaults."
        return GARCHConfig(ticker="SPY") # Minimum required
        
    with open(path, "r") as f:
        data = yaml.safe_load(f)
        return GARCHConfig(**data)
