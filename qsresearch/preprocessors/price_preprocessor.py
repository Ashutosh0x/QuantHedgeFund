"""
QS Research - Price Data Preprocessor

Cleans and prepares price data for backtesting.
"""

from typing import Optional, List
import pandas as pd
from loguru import logger


def preprocess_price_data(
    df: pd.DataFrame,
    min_trading_days: int = 504,
    remove_low_trading_days: bool = True,
    remove_large_gaps: bool = True,
    remove_low_volume: bool = True,
    symbol_column: str = "symbol",
    date_column: str = "date",
    open_column: str = "open",
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
    volume_column: str = "volume",
    engine: str = "polars",
    point_in_time: bool = True,
    volume_lookback_days: int = 252,
    universe_column: str = "in_universe",
) -> pd.DataFrame:
    """
    Preprocess price data for backtesting.
    
    Performs the following cleaning steps:
    1. Remove symbols with insufficient trading history
    2. Remove large price gaps (corporate actions, errors)
    3. Remove low volume periods
    4. Forward fill missing prices
    
    With point_in_time=True (default, for backtests) every step only uses data
    up to the row's own date:
    - Insufficient history does not drop a symbol. Each row gets a boolean
      `universe_column` that is True once the symbol has min_trading_days of
      history up to that date (ANDed with any existing flag). Symbols that
      never reach it are dropped. Strategies should only select flagged rows.
    - Low volume is judged against the average of the previous
      volume_lookback_days bars, not the whole sample.
    With point_in_time=False the original whole-sample rules apply: symbols
    with fewer than min_trading_days rows in total are dropped and volume is
    compared to the full-sample average. Both use future data, which biases
    backtests.
    
    Args:
        df: Raw price DataFrame
        min_trading_days: Minimum required trading days (default: 504 = 2 years)
        remove_low_trading_days: Filter symbols with insufficient history
        remove_large_gaps: Remove suspicious price jumps
        remove_low_volume: Filter out low volume periods
        symbol_column: Name of symbol column
        date_column: Name of date column
        open_column: Name of open price column
        high_column: Name of high price column
        low_column: Name of low price column
        close_column: Name of close price column
        volume_column: Name of volume column
        engine: Processing engine
        point_in_time: Use only past data in each step (see above)
        volume_lookback_days: Point-in-time only: bars in the trailing
            average that low volume is judged against
        universe_column: Point-in-time only: eligibility column to write
        
    Returns:
        Cleaned DataFrame
    """
    initial_count = len(df)
    initial_symbols = df[symbol_column].nunique()
    
    logger.info(f"Preprocessing {initial_count} rows, {initial_symbols} symbols")
    
    df = df.copy()
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by symbol and date
    df = df.sort_values([symbol_column, date_column])
    
    # 1. Remove symbols with insufficient trading days
    if remove_low_trading_days and point_in_time:
        # Trading days of history up to and including each row's date
        history_days = df.groupby(symbol_column).cumcount() + 1
        has_history = history_days >= min_trading_days
        if universe_column in df.columns:
            has_history &= df[universe_column].eq(True)
        df[universe_column] = has_history
        
        ever_ready = df.groupby(symbol_column)[universe_column].transform("any")
        df = df[ever_ready]
        
        removed = initial_symbols - df[symbol_column].nunique()
        logger.info(
            f"Flagged rows with < {min_trading_days} trading days of history; "
            f"removed {removed} symbols that never reach it"
        )
    elif remove_low_trading_days:
        trading_days = df.groupby(symbol_column).size()
        valid_symbols = trading_days[trading_days >= min_trading_days].index
        df = df[df[symbol_column].isin(valid_symbols)]
        
        removed = initial_symbols - df[symbol_column].nunique()
        logger.info(f"Removed {removed} symbols with < {min_trading_days} trading days")
    
    # 2. Remove large price gaps
    if remove_large_gaps:
        # Calculate daily returns
        df["_returns"] = df.groupby(symbol_column)[close_column].pct_change()
        
        # Flag extreme returns (> 100% or < -50%)
        extreme_mask = (df["_returns"].abs() > 1.0) | (df["_returns"] < -0.5)
        extreme_count = extreme_mask.sum()
        
        if extreme_count > 0:
            # Remove these rows
            df = df[~extreme_mask]
            logger.info(f"Removed {extreme_count} rows with extreme price gaps")
        
        df = df.drop(columns=["_returns"])
    
    # 3. Remove low volume periods
    if remove_low_volume:
        if point_in_time:
            # Average volume of the previous volume_lookback_days bars only
            df["_avg_vol"] = df.groupby(symbol_column)[volume_column].transform(
                lambda v: v.shift(1).rolling(volume_lookback_days, min_periods=21).mean()
            )
        else:
            # Calculate average volume per symbol
            df["_avg_vol"] = df.groupby(symbol_column)[volume_column].transform("mean")
        
        # Remove days where volume is < 10% of average
        low_vol_mask = df[volume_column] < (df["_avg_vol"] * 0.1)
        low_vol_count = low_vol_mask.sum()
        
        if low_vol_count > 0:
            df = df[~low_vol_mask]
            logger.info(f"Removed {low_vol_count} low volume rows")
        
        df = df.drop(columns=["_avg_vol"])
    
    # 4. Handle missing values
    # Forward fill prices within each symbol
    price_cols = [open_column, high_column, low_column, close_column]
    for col in price_cols:
        if col in df.columns:
            df[col] = df.groupby(symbol_column)[col].ffill()
    
    # Fill missing volume with 0
    if volume_column in df.columns:
        df[volume_column] = df[volume_column].fillna(0)
    
    # Drop any remaining NaN rows
    df = df.dropna(subset=[close_column])
    
    final_count = len(df)
    final_symbols = df[symbol_column].nunique()
    
    logger.info(
        f"Preprocessed data: {final_count} rows ({final_count/initial_count:.1%}), "
        f"{final_symbols} symbols ({final_symbols/initial_symbols:.1%})"
    )
    
    return df.reset_index(drop=True)
