"""
QS Research - Factor Strategy Algorithms

Trading algorithms based on factor signals.
"""

from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from loguru import logger


def use_factor_as_signal(
    df: pd.DataFrame,
    factor_column: str = "close_qsmom_21_252_126",
    symbol_column: str = "symbol",
    date_column: str = "date",
    top_n: int = 20,
    threshold: Optional[float] = None,
    universe_column: Optional[str] = "in_universe",
    weighting: str = "slot",
) -> pd.DataFrame:
    """
    Use a factor column as the trading signal.
    
    Selects top N stocks by factor value for each rebalancing period.
    
    When fewer than top_n symbols qualify (thin universe, threshold), the
    weighting decides what happens to the spare capital:
    - "slot" (default): every pick gets 1 / top_n and unfilled slots stay in
      cash, so a shrinking universe never concentrates the book.
    - "equal": the picks share the whole book (1 / number of picks), so a
      universe of 4 names means 25% positions.
    
    Args:
        df: DataFrame with factor values
        factor_column: Column containing the factor values
        symbol_column: Symbol column name
        date_column: Date column name
        top_n: Number of top stocks to select
        threshold: Optional minimum factor value threshold
        universe_column: Boolean point-in-time eligibility column (written by
            universe_screener); when present only flagged rows are ranked
        weighting: "slot" or "equal" (see above)
        
    Returns:
        DataFrame with trading signals
    """
    if factor_column not in df.columns:
        raise ValueError(f"Factor column '{factor_column}' not found in DataFrame")
    if weighting not in ("slot", "equal"):
        raise ValueError(f"weighting must be 'slot' or 'equal', got {weighting!r}")
    
    logger.info(f"Generating signals using {factor_column}, top {top_n}")
    
    df = df.copy()
    
    # Get unique dates for rebalancing
    dates = df[date_column].unique()
    
    # Only rank symbols that are in the universe on that date
    if universe_column and universe_column in df.columns:
        df = df[df[universe_column].eq(True)]
    
    signal_records = []
    
    for date in dates:
        # Symbols without a factor value (e.g. still in the lookback warm-up) can't
        # be ranked; nlargest would otherwise return them when all values are NaN
        day_data = df[(df[date_column] == date) & df[factor_column].notna()].copy()
        
        # Apply threshold filter if specified
        if threshold is not None:
            day_data = day_data[day_data[factor_column] >= threshold]
        
        # Rank by factor (descending)
        day_data = day_data.nlargest(top_n, factor_column)
        
        # Equal weight the selected stocks
        if weighting == "slot":
            weight = 1.0 / top_n
        else:
            weight = 1.0 / len(day_data) if len(day_data) > 0 else 0
        
        for _, row in day_data.iterrows():
            signal_records.append({
                "date": date,
                "symbol": row[symbol_column],
                "factor_value": row[factor_column],
                "signal": 1,  # Long signal
                "weight": weight,
            })
    
    signals = pd.DataFrame(signal_records)
    logger.info(f"Generated {len(signals)} trading signals")
    
    return signals


def train_and_predict_xgboost(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "forward_return_21d",
    model_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Train XGBoost model and predict factor values.
    
    Args:
        train_df: Training data
        predict_df: Data to generate predictions for
        feature_columns: List of feature column names
        target_column: Target variable column name
        model_params: XGBoost hyperparameters
        
    Returns:
        predict_df with added 'ml_prediction' column
    """
    try:
        import xgboost as xgb
    except ImportError:
        logger.error("XGBoost not installed")
        raise
    
    if model_params is None:
        model_params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
    
    # Prepare training data
    X_train = train_df[feature_columns].fillna(0)
    y_train = train_df[target_column].fillna(0)
    
    # Train model
    model = xgb.XGBRegressor(**model_params)
    model.fit(X_train, y_train)
    
    # Generate predictions
    X_pred = predict_df[feature_columns].fillna(0)
    predict_df = predict_df.copy()
    predict_df["ml_prediction"] = model.predict(X_pred)
    
    logger.info(f"XGBoost model trained and predictions generated")
    
    return predict_df


def long_short_equal_weight_portfolio(
    signals: pd.DataFrame,
    num_long_positions: int = 20,
    num_short_positions: int = 20,
    long_threshold: float = 1.0,
    short_threshold: float = -1.0,
) -> pd.DataFrame:
    """
    Construct a long/short equal-weight portfolio.
    
    Args:
        signals: DataFrame with factor signals
        num_long_positions: Number of long positions
        num_short_positions: Number of short positions
        long_threshold: Minimum factor value for long
        short_threshold: Maximum factor value for short
        
    Returns:
        DataFrame with portfolio weights
    """
    signals = signals.copy()
    
    # Group by date
    portfolios = []
    
    for date in signals["date"].unique():
        day_signals = signals[signals["date"] == date].copy()
        
        # Long positions (highest factor values)
        long_positions = day_signals.nlargest(num_long_positions, "factor_value")
        long_positions["weight"] = 1.0 / num_long_positions
        long_positions["side"] = "long"
        
        # Short positions (lowest factor values) - if enabled
        if num_short_positions > 0:
            short_positions = day_signals.nsmallest(num_short_positions, "factor_value")
            short_positions["weight"] = -1.0 / num_short_positions
            short_positions["side"] = "short"
            portfolios.append(short_positions)
        
        portfolios.append(long_positions)
    
    portfolio = pd.concat(portfolios, ignore_index=True)
    
    logger.info(
        f"Constructed portfolio with {num_long_positions} long, "
        f"{num_short_positions} short positions"
    )
    
    return portfolio
