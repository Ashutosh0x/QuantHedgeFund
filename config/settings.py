"""
Quant Hedge Fund - Global Settings

Pydantic-based settings management with environment variable support.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,  # Ensure robust loading
    )
    
    # ===================
    # API Keys
    # ===================
    fmp_api_key: str = Field(default="", description="Financial Modeling Prep API key")
    datalink_api_key: str = Field(default="", description="Datalink API key")
    openai_api_key: str = Field(default="", description="OpenAI API key for AI assistant")
    groq_api_key: str = Field(default="", description="Groq API key for LLM features")
    
    # ===================
    # LLM Settings
    # ===================
    groq_model: str = Field(default="openai/gpt-oss-120b", description="Groq LLM model to use")
    
    # ===================
    # Database Settings
    # ===================
    duckdb_path: Path = Field(
        default=Path("./data/quant.duckdb"),
        description="Path to DuckDB database file"
    )
    cache_dir: Path = Field(
        default=Path("./data/cache"),
        description="Directory for cached parquet files"
    )
    
    # ===================
    # MLflow Settings
    # ===================
    mlflow_tracking_uri: str = Field(
        default="http://127.0.0.1:5050",
        description="MLflow tracking server URI"
    )
    mlflow_experiment_name: str = Field(
        default="Nightly Backtest",
        description="Default MLflow experiment name"
    )
    
    # ===================
    # Interactive Brokers
    # ===================
    ib_host: str = Field(default="127.0.0.1", description="IB Gateway host")
    ib_port: int = Field(default=7497, description="IB Gateway port")
    ib_client_id: int = Field(default=1, description="IB client ID")
    ib_paper_trading: bool = Field(default=True, description="Use paper trading mode")
    
    # ===================
    # Risk Management
    # ===================
    max_daily_loss_pct: float = Field(default=0.02, description="Max daily loss as % of portfolio (e.g. 0.02 = 2%)")
    max_symbol_exposure_pct: float = Field(default=0.20, description="Max exposure per symbol (e.g. 0.20 = 20%)")
    max_leverage: float = Field(default=1.5, description="Max portfolio leverage")
    min_order_threshold_usd: float = Field(default=100.0, description="Minimum order size in USD")
    
    # ===================
    # Dashboard Settings
    # ===================
    dashboard_port: int = Field(default=8501, description="Streamlit dashboard port")
    dashboard_data_dir: Path = Field(
        default=Path("./data/outputs"),
        description="Dashboard data directory"
    )
    
    # ===================
    # OpenAI Settings
    # ===================
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model to use")
    
    # ===================
    # Orchestration
    # ===================
    luigi_port: int = Field(default=8082, description="Luigi scheduler port")
    prefect_api_url: str = Field(
        default="http://127.0.0.1:4200",
        description="Prefect API URL"
    )
    
    # ===================
    # Logging
    # ===================
    log_level: str = Field(default="INFO", description="Logging level")
    log_dir: Path = Field(default=Path("./logs"), description="Log directory")
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_data_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.ensure_directories()
    return settings
