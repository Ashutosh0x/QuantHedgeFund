"""
QS Research - Universe Screener

Filters the investment universe based on liquidity, volatility, and other criteria.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
from loguru import logger


def universe_screener(
    df: pd.DataFrame,
    lookback_days: int = 730,
    volume_top_n: Optional[int] = 500,
    momentum_top_n: Optional[int] = None,
    percent_change_filter: bool = False,
    max_percent_change: float = 0.35,
    volatility_filter: bool = True,
    max_volatility: float = 0.25,
    max_volatility_percentile: Optional[float] = None,
    min_avg_volume: int = 100_000,
    min_avg_price: float = 4.0,
    min_last_price: float = 5.0,
    symbol_column: str = "symbol",
    date_column: str = "date",
    close_column: str = "close",
    volume_column: str = "volume",
    point_in_time: bool = True,
    min_history_days: int = 21,
    universe_column: str = "in_universe",
) -> pd.DataFrame:
    """
    Screen the investment universe based on multiple criteria.
    
    This screener helps avoid:
    - Illiquid stocks that are hard to trade
    - Penny stocks with high manipulation risk
    - Highly volatile stocks that add noise
    - Stocks with suspicious price movements
    
    Two modes:
    - point_in_time=True (default, for backtests): every row is kept and gets a
      boolean `universe_column` saying whether the symbol passes the screen on
      that date, using only data up to and including that date (trailing
      window of lookback_days * 1.5 calendar days), ANDed with any existing
      flag. Symbols that never pass are dropped. Strategies should only
      select rows where the flag is True.
    - point_in_time=False (for picking today's universe): the screen is run
      once on the latest window and passing symbols are kept with their full
      history. Using this in a backtest picks the universe with knowledge of
      the future (look-ahead / survivorship bias).
    
    Args:
        df: Price DataFrame
        lookback_days: Period for calculating metrics
        volume_top_n: Keep only top N by average volume
        momentum_top_n: Keep only top N by momentum
        percent_change_filter: Filter extreme daily changes
        max_percent_change: Maximum allowed daily change
        volatility_filter: Filter high volatility stocks
        max_volatility: Maximum allowed annualized volatility
        max_volatility_percentile: Relative cap, e.g. 0.8 drops the most
            volatile 20% of the remaining symbols on each date. Unlike the
            absolute cap it keeps the universe size stable when volatility
            rises market-wide.
        min_avg_volume: Minimum average daily volume
        min_avg_price: Minimum average price
        min_last_price: Minimum last traded price
        symbol_column: Name of symbol column
        date_column: Name of date column
        close_column: Name of close price column
        volume_column: Name of volume column
        point_in_time: Screen each date on trailing data only (see above)
        min_history_days: Point-in-time only: trading days of history a
            symbol needs before it can be eligible
        universe_column: Point-in-time only: name of the eligibility column

    Returns:
        Filtered DataFrame
    """
    initial_symbols = df[symbol_column].nunique()
    logger.info(f"Screening universe from {initial_symbols} symbols")
    
    df = df.copy()
    
    if point_in_time:
        return _point_in_time_screen(
            df,
            lookback_days=lookback_days,
            volume_top_n=volume_top_n,
            volatility_filter=volatility_filter,
            max_volatility=max_volatility,
            max_volatility_percentile=max_volatility_percentile,
            min_avg_volume=min_avg_volume,
            min_avg_price=min_avg_price,
            min_last_price=min_last_price,
            min_history_days=min_history_days,
            symbol_column=symbol_column,
            date_column=date_column,
            close_column=close_column,
            volume_column=volume_column,
            universe_column=universe_column,
        )
    
    # Screening metrics use the latest lookback_days of data; the full history
    # of the symbols that pass is returned (backtests need it for factor warm-up)
    window = df
    if lookback_days:
        max_date = df[date_column].max()
        min_date = max_date - pd.Timedelta(days=lookback_days * 1.5)  # Buffer for weekends
        window = df[df[date_column] >= min_date]

    # Calculate screening metrics per symbol
    metrics = window.groupby(symbol_column).agg({
        close_column: ["mean", "std", "last"],
        volume_column: "mean",
    })
    metrics.columns = ["avg_price", "price_std", "last_price", "avg_volume"]

    # Calculate annualized volatility
    volatility = window.groupby(symbol_column).apply(
        lambda x: x[close_column].pct_change().std() * np.sqrt(252)
    )
    metrics["volatility"] = volatility
    
    # Apply filters
    valid_symbols = metrics.index.tolist()
    
    # Minimum average volume
    if min_avg_volume:
        vol_filter = metrics["avg_volume"] >= min_avg_volume
        filtered_out = (~vol_filter).sum()
        valid_symbols = metrics[vol_filter].index.tolist()
        logger.debug(f"Volume filter removed {filtered_out} symbols")
    
    # Minimum average price
    if min_avg_price:
        price_filter = metrics.loc[valid_symbols, "avg_price"] >= min_avg_price
        filtered_out = (~price_filter).sum()
        valid_symbols = [s for s, v in price_filter.items() if v]
        logger.debug(f"Avg price filter removed {filtered_out} symbols")
    
    # Minimum last price
    if min_last_price:
        last_filter = metrics.loc[valid_symbols, "last_price"] >= min_last_price
        filtered_out = (~last_filter).sum()
        valid_symbols = [s for s, v in last_filter.items() if v]
        logger.debug(f"Last price filter removed {filtered_out} symbols")
    
    # Volatility filter
    if volatility_filter:
        vol_filter = metrics.loc[valid_symbols, "volatility"] <= max_volatility
        filtered_out = (~vol_filter).sum()
        valid_symbols = [s for s, v in vol_filter.items() if v]
        logger.debug(f"Volatility filter removed {filtered_out} symbols")
    
    # Relative volatility cap
    if max_volatility_percentile is not None and valid_symbols:
        vol_pct = metrics.loc[valid_symbols, "volatility"].rank(pct=True)
        valid_symbols = [s for s, v in (vol_pct <= max_volatility_percentile).items() if v]
    
    # Top N by volume
    if volume_top_n and len(valid_symbols) > volume_top_n:
        top_by_volume = (
            metrics.loc[valid_symbols]
            .nlargest(volume_top_n, "avg_volume")
            .index.tolist()
        )
        valid_symbols = top_by_volume
        logger.debug(f"Volume top_n reduced to {volume_top_n} symbols")
    
    # Filter DataFrame
    df = df[df[symbol_column].isin(valid_symbols)]
    
    final_symbols = df[symbol_column].nunique()
    logger.info(
        f"Screened universe: {final_symbols} symbols "
        f"({final_symbols/initial_symbols:.1%} retained)"
    )
    
    return df


def _point_in_time_screen(
    df: pd.DataFrame,
    lookback_days: Optional[int],
    volume_top_n: Optional[int],
    volatility_filter: bool,
    max_volatility: float,
    max_volatility_percentile: Optional[float],
    min_avg_volume: Optional[float],
    min_avg_price: Optional[float],
    min_last_price: Optional[float],
    min_history_days: int,
    symbol_column: str,
    date_column: str,
    close_column: str,
    volume_column: str,
    universe_column: str,
) -> pd.DataFrame:
    """
    Flag, per date, the symbols that pass the screen on trailing data only.
    
    Each metric at date t uses rows dated <= t within a trailing window of
    lookback_days * 1.5 calendar days (the same span the latest-window screen
    uses), or all prior rows if lookback_days is falsy.
    """
    initial_symbols = df[symbol_column].nunique()
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    df = df.sort_values([symbol_column, date_column]).reset_index(drop=True)
    
    trailing = pd.DataFrame({
        symbol_column: df[symbol_column],
        date_column: df[date_column],
        "price": df[close_column],
        "volume": df[volume_column],
        "ret": df.groupby(symbol_column)[close_column].pct_change(),
    }).set_index(date_column)
    grouped = trailing.groupby(symbol_column, sort=True)
    if lookback_days:
        window = f"{int(lookback_days * 1.5)}D"
        rolling = grouped.rolling(window, min_periods=min_history_days)
    else:
        rolling = grouped.expanding(min_periods=min_history_days)
    
    # groupby(sort=True) walks symbols in sorted order and rows in date order,
    # which is exactly df's order after the sort above
    avg_price = rolling["price"].mean().to_numpy()
    avg_volume = rolling["volume"].mean().to_numpy()
    volatility = rolling["ret"].std().to_numpy() * np.sqrt(252)
    
    eligible = ~np.isnan(avg_price)  # min_history_days reached
    if min_avg_volume:
        eligible &= avg_volume >= min_avg_volume
    if min_avg_price:
        eligible &= avg_price >= min_avg_price
    if min_last_price:
        eligible &= df[close_column].to_numpy() >= min_last_price
    if volatility_filter:
        eligible &= volatility <= max_volatility
    # Respect earlier point-in-time flags (e.g. price_preprocessor's history rule)
    if universe_column in df.columns:
        eligible &= df[universe_column].eq(True).to_numpy()
    
    # Relative volatility cap among that date's eligible symbols
    if max_volatility_percentile is not None:
        eligible_vol = pd.Series(np.where(eligible, volatility, np.nan), index=df.index)
        vol_pct = eligible_vol.groupby(df[date_column]).rank(pct=True)
        eligible &= (vol_pct <= max_volatility_percentile).to_numpy()
    
    # Top N by trailing average volume among that date's eligible symbols
    if volume_top_n:
        ranked_volume = pd.Series(np.where(eligible, avg_volume, np.nan), index=df.index)
        volume_rank = ranked_volume.groupby(df[date_column]).rank(ascending=False, method="first")
        eligible &= (volume_rank <= volume_top_n).to_numpy()
    
    df[universe_column] = eligible
    
    # Symbols that are never eligible can never be traded; drop them
    ever_eligible = df.groupby(symbol_column)[universe_column].transform("any")
    df = df[ever_eligible].reset_index(drop=True)
    
    per_date = df[df[universe_column]].groupby(date_column)[symbol_column].nunique()
    logger.info(
        f"Point-in-time universe: {df[symbol_column].nunique()} of {initial_symbols} symbols "
        f"eligible at some point, {per_date.median() if len(per_date) else 0:.0f} per date (median)"
    )
    
    return df
