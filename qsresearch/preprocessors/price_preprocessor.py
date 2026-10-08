"""
QS Research - Price Data Preprocessor

Cleans and prepares price data for backtesting.
"""

from typing import Optional, List
import pandas as pd
from loguru import logger


DEFAULT_MIN_TRADING_DAYS = 504


def preprocess_price_data(
    df: pd.DataFrame,
    min_trading_days: int = DEFAULT_MIN_TRADING_DAYS,
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
    universe_column: str = "in_universe",
) -> pd.DataFrame:
    """
    Preprocess price data for backtesting.
    
    Performs the following cleaning steps:
    1. Flag rows where the symbol does not yet have enough trading history
    2. Remove large price gaps (corporate actions, errors)
    3. Remove low volume periods (vs. the symbol's average volume to date)
    4. Forward fill missing prices
    
    Args:
        df: Raw price DataFrame
        min_trading_days: Minimum required trading days (default: 504 = 2 years)
        remove_low_trading_days: Filter rows whose symbol has fewer than
            min_trading_days rows up to and including that date. The rows are
            kept (factors need the history) and flagged False in
            universe_column; symbols that never reach min_trading_days in the
            data are dropped, since none of their rows could be traded.
        remove_large_gaps: Remove suspicious price jumps
        remove_low_volume: Remove days with volume below 10% of the symbol's
            average volume up to and including that day
        symbol_column: Name of symbol column
        date_column: Name of date column
        open_column: Name of open price column
        high_column: Name of high price column
        low_column: Name of low price column
        close_column: Name of close price column
        volume_column: Name of volume column
        engine: Processing engine
        universe_column: Boolean eligibility column to add (ANDed with an
            existing one); universe_screener narrows it further
        
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
    
    # 1. Flag rows before each symbol has min_trading_days of history, counting
    # only rows up to and including that date (the total over the whole sample
    # would let future data decide which symbols are tradable early on)
    if remove_low_trading_days:
        history_days = df.groupby(symbol_column).cumcount().to_numpy() + 1
        has_history = history_days >= min_trading_days
        if universe_column in df.columns:
            has_history &= df[universe_column].eq(True).to_numpy()
        df[universe_column] = has_history
        
        ever_valid = df.loc[df[universe_column], symbol_column].unique()
        df = df[df[symbol_column].isin(ever_valid)]
        
        removed = initial_symbols - df[symbol_column].nunique()
        young = (~df[universe_column]).sum()
        logger.info(
            f"Removed {removed} symbols that never reach {min_trading_days} trading days; "
            f"flagged {young} rows before their symbol reaches it"
        )
    
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
        # Average volume per symbol up to and including each day; a whole-sample
        # average would let later volume decide which earlier days are dropped.
        # df is sorted by symbol, so the grouped result lines up by position.
        df["_avg_vol"] = (
            df.groupby(symbol_column, sort=False, dropna=False)[volume_column]
            .expanding()
            .mean()
            .to_numpy()
        )
        
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
