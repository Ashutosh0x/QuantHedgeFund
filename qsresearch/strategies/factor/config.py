"""
QS Research - Momentum Factor Strategy Configuration

This is the default momentum factor strategy configuration.
"""

# Momentum Factor Strategy Configuration
MOMENTUM_FACTOR_CONFIG = {
    # MLflow settings
    "experiment_name": "Momentum Factor Strategy",
    "run_name": "qsmom_equal_weight_long_only",
    
    # Backtest parameters
    "bundle_name": "historical_prices_fmp",
    "start_date": "2015-01-01",
    "end_date": "2025-02-14",
    "capital_base": 1_000_000,
    
    # Preprocessing pipeline
    "preprocessing": [
        {
            "name": "price_preprocessor",
            "func": "qsresearch.preprocessors.price_preprocessor:preprocess_price_data",
            "params": {
                "min_trading_days": 504,
                "remove_low_trading_days": True,
                "remove_large_gaps": True,
                "remove_low_volume": True,
                "symbol_column": "symbol",
                "date_column": "date",
                "open_column": "open",
                "high_column": "high",
                "low_column": "low",
                "close_column": "close",
                "volume_column": "volume",
                "engine": "polars",
            },
        },
        {
            "name": "universe_screener",
            "func": "qsresearch.preprocessors.universe_screener:universe_screener",
            "params": {
                "lookback_days": 730,
                "volume_top_n": 500,
                "momentum_top_n": None,
                "percent_change_filter": False,
                "max_percent_change": 0.35,
                # Relative cap: drop the most volatile 20% of names on each date.
                # An absolute cap (volatility_filter=True, max_volatility=0.25)
                # empties the universe when volatility rises market-wide (6-14
                # of 50 large caps were eligible in 2020-2022).
                "volatility_filter": False,
                "max_volatility": 0.25,
                "max_volatility_percentile": 0.8,
                "min_avg_volume": 100000,
                "min_avg_price": 4.0,
                "min_last_price": 5.0,
                "symbol_column": "symbol",
                "date_column": "date",
                "close_column": "close",
                "volume_column": "volume",
            },
        },
    ],
    
    # Factor calculation
    "factors": [
        {
            "name": "momentum_factor",
            "func": "qsresearch.features.momentum:add_qsmom_features",
            "params": {
                "fast_period": 21,
                "slow_period": 252,
                "signal_period": 126,
                "symbol_column": "symbol",
                "date_column": "date",
                "close_column": "close",
            },
        },
        {
            "name": "forward_returns",
            "func": "qsresearch.features.forward_returns:add_forward_returns",
            "params": {
                "forward_periods": [21],
                "symbol_column": "symbol",
                "date_column": "date",
                "close_column": "close",
                "engine": "polars",
            },
        },
    ],
    
    # Algorithm settings
    "algorithm": {
        "callable": "qsresearch.strategies.factor.algorithms:use_factor_as_signal",
        "params": {
            "factor_column": "close_qsmom_21_252_126",
            "top_n": 20,
            "threshold": None,
            "weighting": "slot",  # 1/top_n per pick; unfilled slots stay in cash
        },
    },
    
    # Portfolio construction
    "portfolio_strategy": {
        "func": "long_short_equal_weight_portfolio",
        "params": {
            "num_long_positions": 20,
            "num_short_positions": 0,  # Long only
            "long_threshold": 1.0,
        },
    },
    
    # Risk settings
    "stop_loss_enabled": False,
    "stop_loss_pct": 0.15,
    
    # Benchmark
    "benchmark_symbol": "SPY",
}

# Value Factor Strategy
VALUE_FACTOR_CONFIG = {
    "experiment_name": "Value Factor Strategy",
    "run_name": "value_pb_ratio",
    "bundle_name": "historical_prices_fmp",
    "start_date": "2015-01-01",
    "capital_base": 1_000_000,
    
    "preprocessing": [
        {
            "name": "price_preprocessor",
            "func": "qsresearch.preprocessors.price_preprocessor:preprocess_price_data",
            "params": {"min_trading_days": 504},
        },
        {
            "name": "universe_screener", 
            "func": "qsresearch.preprocessors.universe_screener:universe_screener",
            "params": {"volume_top_n": 500},
        },
    ],
    
    "algorithm": {
        "callable": "qsresearch.strategies.factor.algorithms:use_factor_as_signal",
        "params": {
            "factor_column": "pb_ratio_rank",
            "top_n": 30,
        },
    },
    
    "portfolio_strategy": {
        "func": "long_short_equal_weight_portfolio",
        "params": {
            "num_long_positions": 30,
            "num_short_positions": 0,
        },
    },
}
