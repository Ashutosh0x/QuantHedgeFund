"""
QS Research - Universe Screener

Filters the investment universe based on liquidity, volatility, and other criteria.

Membership is point-in-time: each row is flagged using only that symbol's data up
to and including the row's date, so a backtest never trades a universe chosen
with knowledge of the future.
"""

import math
from typing import Optional, List
import pandas as pd
import numpy as np
from loguru import logger


DEFAULT_LOOKBACK_DAYS = 730
LOOKBACK_CALENDAR_BUFFER = 1.5  # lookback_days * 1.5 calendar days, buffer for weekends
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365


def lookback_trading_days(lookback_days: Optional[int]) -> int:
    """
    Trading days of history the screener's trailing window spans.
    
    The window is lookback_days * 1.5 calendar days, converted at 252 trading
    days per 365 calendar days. A falsy lookback_days screens on all history
    (expanding window), which no finite warm-up can cover, so it returns 0.
    """
    if not lookback_days:
        return 0
    calendar_days = lookback_days * LOOKBACK_CALENDAR_BUFFER
    return math.ceil(calendar_days * TRADING_DAYS_PER_YEAR / CALENDAR_DAYS_PER_YEAR)


def universe_screener(
    df: pd.DataFrame,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    volume_top_n: Optional[int] = 500,
    momentum_top_n: Optional[int] = None,
    percent_change_filter: bool = False,
    max_percent_change: float = 0.35,
    volatility_filter: bool = True,
    max_volatility: float = 0.25,
    min_avg_volume: int = 100_000,
    min_avg_price: float = 4.0,
    min_last_price: float = 5.0,
    symbol_column: str = "symbol",
    date_column: str = "date",
    close_column: str = "close",
    volume_column: str = "volume",
    universe_column: str = "in_universe",
) -> pd.DataFrame:
    """
    Screen the investment universe based on multiple criteria, point-in-time.
    
    This screener helps avoid:
    - Illiquid stocks that are hard to trade
    - Penny stocks with high manipulation risk
    - Highly volatile stocks that add noise
    - Stocks with suspicious price movements
    
    No rows are dropped. Each row gets a boolean `universe_column` that is True
    when the symbol passes every filter on that date, using metrics over the
    trailing window (date - lookback_days * 1.5 calendar days, inclusive, up to
    and including the date). An existing universe_column (e.g. the history flag
    from preprocess_price_data) is ANDed in first. volume_top_n ranks the
    symbols that pass the other filters by trailing average volume across each
    date. Signal generators such as use_factor_as_signal rank only the rows
    where the flag is True.
    
    Args:
        df: Price DataFrame
        lookback_days: Period for calculating metrics (trailing window of
            lookback_days * 1.5 calendar days; falsy uses all history to date)
        volume_top_n: Keep only top N by average volume on each date
        momentum_top_n: Accepted for config compatibility; not applied
        percent_change_filter: Accepted for config compatibility; not applied
        max_percent_change: Accepted for config compatibility; not applied
        volatility_filter: Filter high volatility stocks
        max_volatility: Maximum allowed annualized volatility
        min_avg_volume: Minimum average daily volume
        min_avg_price: Minimum average price
        min_last_price: Minimum last traded price
        symbol_column: Name of symbol column
        date_column: Name of date column
        close_column: Name of close price column
        volume_column: Name of volume column
        universe_column: Name of the eligibility column to add (ANDed with
            an existing column of that name)
    
    Returns:
        The input rows with an added boolean universe_column
    """
    initial_symbols = df[symbol_column].nunique()
    logger.info(f"Screening universe from {initial_symbols} symbols (point-in-time)")
    
    df = df.copy()
    
    # Rows already excluded upstream stay excluded
    if universe_column in df.columns:
        prior = df[universe_column].eq(True).to_numpy()
    else:
        prior = np.ones(len(df), dtype=bool)
    
    # Work on a (symbol, date)-sorted copy; results go back by position
    work = pd.DataFrame({
        "symbol": df[symbol_column].to_numpy(),
        "date": pd.to_datetime(df[date_column]).to_numpy(),
        "close": df[close_column].to_numpy(dtype=float),
        "volume": df[volume_column].to_numpy(dtype=float),
        "prior": prior,
        "row": np.arange(len(df)),
    })
    work = work.sort_values(["symbol", "date", "row"], kind="mergesort").reset_index(drop=True)
    
    closes = work.groupby("symbol", sort=False)["close"]
    # Same as the window's last non-null close, i.e. the last traded price
    work["last_price"] = closes.ffill()
    work["daily_return"] = closes.pct_change(fill_method=None)
    
    # Calculate trailing screening metrics per symbol, as of each row's date
    by_symbol = work.groupby("symbol", sort=False)
    metric_columns = ["close", "volume", "daily_return"]
    if lookback_days:
        window = pd.Timedelta(days=lookback_days * LOOKBACK_CALENDAR_BUFFER)
        trailing = by_symbol.rolling(window, on="date", closed="both")[metric_columns]
    else:
        trailing = by_symbol[metric_columns].expanding()
    means = trailing.mean()
    work["avg_price"] = means["close"].to_numpy()
    work["avg_volume"] = means["volume"].to_numpy()
    
    # Calculate annualized volatility
    work["volatility"] = trailing.std()["daily_return"].to_numpy() * np.sqrt(252)
    
    # Apply filters; a NaN metric fails its filter
    eligible = work["prior"].copy()
    
    # Minimum average volume
    if min_avg_volume:
        eligible &= work["avg_volume"] >= min_avg_volume
    
    # Minimum average price
    if min_avg_price:
        eligible &= work["avg_price"] >= min_avg_price
    
    # Minimum last price
    if min_last_price:
        eligible &= work["last_price"] >= min_last_price
    
    # Volatility filter
    if volatility_filter:
        eligible &= work["volatility"] <= max_volatility
    
    # Top N by volume among the symbols passing the filters on each date;
    # ties go to the earlier symbol, as nlargest did
    if volume_top_n:
        volume_rank = (
            work["avg_volume"]
            .where(eligible)
            .groupby(work["date"])
            .rank(method="first", ascending=False)
        )
        eligible &= volume_rank <= volume_top_n
    
    flags = np.zeros(len(df), dtype=bool)
    flags[work["row"].to_numpy()] = eligible.to_numpy()
    df[universe_column] = flags
    
    latest = work["date"] == work["date"].max()
    ever = work.loc[eligible, "symbol"].nunique()
    daily = eligible.groupby(work["date"]).sum()
    logger.info(
        f"Screened universe: {daily.mean():.0f} eligible symbols per date on average, "
        f"{int(eligible[latest].sum())} on the last date, {ever} of {initial_symbols} "
        f"eligible at some point"
    )
    
    return df
