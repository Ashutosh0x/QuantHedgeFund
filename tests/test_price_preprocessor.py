"""
Tests for the point-in-time history filter in qsresearch.preprocessors.price_preprocessor.

A row may only count a symbol's trading days up to and including its own date.
The old filter counted each symbol's rows over the whole sample, so a symbol
that went on to trade for min_trading_days was eligible from its first day,
before it had any of that history.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

from qsresearch.preprocessors import preprocess_price_data, universe_screener


DATES = pd.bdate_range("2019-01-01", "2023-12-29")


def _frame(symbol: str, dates, close=50.0, volume=1_000_000.0) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        "date": dates,
        "symbol": symbol,
        "close": np.broadcast_to(np.asarray(close, dtype=float), n).copy(),
        "volume": np.broadcast_to(np.asarray(volume, dtype=float), n).copy(),
    })


def _preprocess(df: pd.DataFrame, **overrides) -> pd.DataFrame:
    kwargs = dict(min_trading_days=504, remove_large_gaps=False, remove_low_volume=False)
    return preprocess_price_data(df, **{**kwargs, **overrides})


def _flags(processed: pd.DataFrame, symbol: str) -> pd.Series:
    rows = processed[processed["symbol"] == symbol]
    return rows.set_index("date")["in_universe"]


class TestPointInTimeHistory:
    def test_rows_before_min_history_are_flagged_not_dropped(self):
        # LISTED trades 700 days from 2021, so the old whole-sample count (700 >= 504)
        # made it eligible on its first day
        listed = DATES[DATES >= pd.Timestamp("2021-01-01")][:700]
        prices = pd.concat([_frame("OLD", DATES), _frame("LISTED", listed)], ignore_index=True)
        
        processed = _preprocess(prices)
        flags = _flags(processed, "LISTED")
        
        assert len(flags) == 700
        assert not flags.iloc[:503].any()
        assert flags.iloc[503:].all()
        assert flags.index[503] == listed[503]
        assert _flags(processed, "OLD").iloc[503:].all()
    
    def test_symbol_that_never_reaches_min_history_is_dropped(self):
        short = DATES[:300]
        prices = pd.concat([_frame("OLD", DATES), _frame("SHORT", short)], ignore_index=True)
        
        processed = _preprocess(prices)
        
        assert set(processed["symbol"]) == {"OLD"}
    
    def test_history_ignores_future_rows(self):
        """Flags up to any date are identical whether or not later data exists."""
        rng = np.random.default_rng(3)
        frames = []
        for i in range(10):
            # Random listing and delisting dates, so symbols cross 504 days at different times
            first, last = sorted(rng.choice(len(DATES), size=2, replace=False))
            frames.append(_frame(f"S{i}", DATES[first:last + 1]))
        prices = pd.concat(frames, ignore_index=True)
        
        full = _preprocess(prices).set_index(["symbol", "date"])["in_universe"]
        assert 0.05 < full.mean() < 0.95
        
        for cut in ["2020-06-30", "2021-09-15", "2023-01-31"]:
            truncated = _preprocess(prices[prices["date"] <= cut]).set_index(["symbol", "date"])["in_universe"]
            # Rows dropped from the truncated run belong to symbols not yet eligible
            expected = full[full.index.get_level_values("date") <= cut]
            assert truncated.index.isin(expected.index).all()
            got = truncated.reindex(expected.index, fill_value=False)
            pd.testing.assert_series_equal(got, expected)
    
    def test_filter_disabled_adds_no_flag(self):
        prices = pd.concat([_frame("OLD", DATES), _frame("SHORT", DATES[:300])], ignore_index=True)
        
        processed = _preprocess(prices, remove_low_trading_days=False)
        
        assert "in_universe" not in processed.columns
        assert set(processed["symbol"]) == {"OLD", "SHORT"}



class TestPointInTimeLowVolume:
    def _run(self, prices):
        return _preprocess(prices, remove_low_trading_days=False, remove_low_volume=True)
    
    def test_early_days_are_judged_on_volume_to_date(self):
        # GROWER trades 10k shares a day, then 1M from 2021. Its whole-sample average
        # (about 600k) put every early day under the 10% cut, so the old filter
        # deleted its first two years using volume it had not traded yet
        prices = _frame("GROWER", DATES, volume=np.where(DATES < pd.Timestamp("2021-01-01"), 1e4, 1e6))
        
        processed = self._run(prices)
        
        assert len(processed) == len(DATES)
    
    def test_thin_day_is_still_removed(self):
        volume = np.full(len(DATES), 1e6)
        volume[600] = 5e3
        prices = _frame("A", DATES, volume=volume)
        
        processed = self._run(prices)
        
        assert len(processed) == len(DATES) - 1
        assert DATES[600] not in set(processed["date"])
    
    def test_removed_days_ignore_future_rows(self):
        rng = np.random.default_rng(11)
        frames = []
        for i in range(6):
            # Volume trends up or down over the sample, with occasional thin days
            trend = np.exp(np.linspace(0.0, rng.uniform(-3.0, 3.0), len(DATES)))
            volume = 1e5 * trend * rng.lognormal(0.0, 1.0, len(DATES))
            frames.append(_frame(f"S{i}", DATES, volume=volume))
        prices = pd.concat(frames, ignore_index=True)
        keys = lambda df: set(zip(df["symbol"], df["date"]))
        
        full = keys(self._run(prices))
        assert 0 < len(full) < len(prices)
        
        for cut in ["2020-03-31", "2021-12-31", "2023-06-30"]:
            truncated = keys(self._run(prices[prices["date"] <= cut]))
            assert truncated == {k for k in full if k[1] <= pd.Timestamp(cut)}

class TestScreenerKeepsHistoryFlag:
    def test_young_symbol_does_not_take_a_volume_top_n_slot(self):
        # YOUNG trades five times OLD's volume but lists in 2021; with volume_top_n=1
        # OLD must keep the slot until YOUNG has 504 days of history
        young_dates = DATES[DATES >= pd.Timestamp("2021-01-01")]
        prices = pd.concat([
            _frame("OLD", DATES),
            _frame("YOUNG", young_dates, volume=5_000_000.0),
        ], ignore_index=True)
        
        processed = _preprocess(prices)
        screened = universe_screener(processed, lookback_days=30, volume_top_n=1, volatility_filter=False)
        old, young = _flags(screened, "OLD"), _flags(screened, "YOUNG")
        
        assert not young.iloc[:503].any()
        assert old[young.index[:503]].all()
        assert young.iloc[503:].all()
        assert not old[young.index[503:]].any()


class TestHistoryWarmup:
    def test_infer_warmup_includes_min_trading_days(self):
        from qsresearch.backtest.run_backtest import _infer_warmup_days
        from qsresearch.strategies.factor.config import MOMENTUM_FACTOR_CONFIG
        
        step = lambda params: [{"name": "price_preprocessor", "params": params}]
        assert _infer_warmup_days([], step({"min_trading_days": 504})) == 504
        # Omitted min_trading_days falls back to the default (504)
        assert _infer_warmup_days([], step({})) == 504
        assert _infer_warmup_days([], step({"min_trading_days": 504, "remove_low_trading_days": False})) == 0
        # The default momentum config's screener window (756) is still the longest
        momentum = MOMENTUM_FACTOR_CONFIG
        assert _infer_warmup_days(momentum["factors"], momentum["preprocessing"]) == 756
    
    def test_run_backtest_waits_for_min_history(self, monkeypatch, tmp_path):
        module = importlib.import_module("qsresearch.backtest.run_backtest")
        
        rng = np.random.default_rng(5)
        young_dates = DATES[DATES >= pd.Timestamp("2021-03-01")]
        frames = []
        # YOUNG has by far the best momentum; its 252-day factor is ready about a year
        # after listing, but it only has 400 days of history about 19 months after listing
        specs = {"YOUNG": (young_dates, 0.003), **{f"S{i}": (DATES, 0.0001 * i) for i in range(5)}}
        for symbol, (dates, drift) in specs.items():
            returns = rng.normal(drift, 0.01, len(dates))
            returns[0] = 0.0
            frames.append(_frame(symbol, dates, close=100.0 * np.cumprod(1.0 + returns)))
        prices = pd.concat(frames, ignore_index=True)
        
        monkeypatch.setattr(
            module, "_load_price_data",
            lambda b, s, e: prices[(prices.date >= s) & (prices.date <= e)].reset_index(drop=True),
        )
        captured = {}
        simulate = module._simulate_portfolio
        
        def capture_signals(signals, *args, **kwargs):
            captured["signals"] = signals
            return simulate(signals, *args, **kwargs)
        
        monkeypatch.setattr(module, "_simulate_portfolio", capture_signals)
        config = {
            "start_date": "2022-01-03",
            "end_date": "2023-12-29",
            "simulation": {"rebalance_frequency": "D"},
            "preprocessing": [{"name": "price_preprocessor", "params": {
                "min_trading_days": 400, "remove_large_gaps": False, "remove_low_volume": False,
            }}],
            "factors": [{"name": "momentum_factor", "func": "qsresearch.features.momentum:add_qsmom_features",
                         "params": {"fast_period": 21, "slow_period": 252, "signal_period": 126}}],
            "algorithm": {"params": {"factor_column": "close_qsmom_21_252_126", "top_n": 3}},
        }
        
        results = module.run_backtest(config, output_dir=tmp_path, log_to_mlflow=False)
        signals = captured["signals"]
        young_signals = pd.to_datetime(signals.loc[signals["symbol"] == "YOUNG", "date"])
        
        assert results["warmup_days"] == 400
        # The old whole-sample count let YOUNG trade as soon as its factor was ready
        assert young_signals.min() == young_dates[399]
        assert young_signals.max() == DATES[-1]
        # The warm-up covers 400 days, so long-listed symbols are tradable on start_date
        first_day = signals[pd.to_datetime(signals["date"]) == pd.Timestamp("2022-01-03")]
        assert len(first_day) == 3
        assert results["performance"]["turnover"].iloc[0] == pytest.approx(1.0)
