"""
Tests for point-in-time universe membership (qsresearch.preprocessors.universe_screener).

Membership on a date may only use data up to and including that date. The old
screener judged every symbol on the window ending at the last date of the data
and kept that set for the whole backtest, so a symbol that only became liquid,
pricier or calmer late in the sample was traded from day one, and a symbol that
failed only at the end was never traded at all.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

from qsresearch.preprocessors import universe_screener
from qsresearch.preprocessors.universe_screener import lookback_trading_days
from qsresearch.strategies.factor.algorithms import use_factor_as_signal


DATES = pd.bdate_range("2018-01-01", "2023-12-29")
SWITCH = pd.Timestamp("2021-01-01")   # LATE starts passing, EARLY starts failing
EARLY_CHECK = pd.Timestamp("2020-06-30")
WINDOW_FILLED = pd.Timestamp("2018-09-01")


def _frame(symbol: str, close, volume) -> pd.DataFrame:
    n = len(DATES)
    return pd.DataFrame({
        "date": DATES,
        "symbol": symbol,
        "close": np.broadcast_to(np.asarray(close, dtype=float), n).copy(),
        "volume": np.broadcast_to(np.asarray(volume, dtype=float), n).copy(),
    })


def _switching(before, after) -> np.ndarray:
    return np.where(DATES < SWITCH, before, after).astype(float)


def _alternating_close(start: float, step: float, active: np.ndarray) -> np.ndarray:
    """Close that moves +step/-step on alternate days where active, flat elsewhere."""
    signs = np.where(np.arange(len(DATES)) % 2 == 0, 1.0, -1.0)
    returns = np.where(active, signs * step, 0.0)
    returns[0] = 0.0
    return start * np.cumprod(1.0 + returns)


def _liquidity_case():
    late = _frame("LATE", 50.0, _switching(10_000, 1_000_000))
    early = _frame("EARLY", 50.0, _switching(1_000_000, 10_000))
    return late, early, {}


def _price_case():
    # Smooth geometric paths (constant daily return, so zero volatility):
    # LATE climbs from $0.50 to $50, EARLY falls from $50 to $0.50
    growth = 100.0 ** (np.arange(len(DATES)) / (len(DATES) - 1))
    late = _frame("LATE", 0.5 * growth, 1_000_000)
    early = _frame("EARLY", 50.0 / growth, 1_000_000)
    return late, early, {}


def _volatility_case():
    # +/-3% on alternate days is about 48% annualized volatility
    late = _frame("LATE", _alternating_close(50.0, 0.03, DATES < SWITCH), 1_000_000)
    early = _frame("EARLY", _alternating_close(50.0, 0.03, DATES >= SWITCH), 1_000_000)
    return late, early, {}


def _volume_rank_case():
    # With volume_top_n=2, STEADY (1M) is always in the top two; LATE and EARLY trade places
    late = _frame("LATE", 50.0, _switching(500_000, 2_000_000))
    early = _frame("EARLY", 50.0, _switching(2_000_000, 500_000))
    return late, early, {"volume_top_n": 2}


CASES = {
    "min_avg_volume": _liquidity_case,
    "min_price": _price_case,
    "max_volatility": _volatility_case,
    "volume_top_n": _volume_rank_case,
}

SCREEN_KWARGS = dict(
    lookback_days=100,
    volume_top_n=None,
    volatility_filter=True,
    max_volatility=0.25,
    min_avg_volume=100_000,
    min_avg_price=4.0,
    min_last_price=5.0,
)


def _screen(df: pd.DataFrame, **overrides) -> pd.DataFrame:
    return universe_screener(df, **{**SCREEN_KWARGS, **overrides})


def _flags(screened: pd.DataFrame, symbol: str) -> pd.Series:
    rows = screened[screened["symbol"] == symbol]
    return rows.set_index("date")["in_universe"]


class TestPointInTimeMembership:
    @pytest.mark.parametrize("case", CASES, ids=list(CASES))
    def test_symbol_that_passes_only_late_is_not_eligible_early(self, case):
        late, early, overrides = CASES[case]()
        steady = _frame("STEADY", 50.0, 1_000_000)
        screened = _screen(pd.concat([late, early, steady], ignore_index=True), **overrides)
        late_flags = _flags(screened, "LATE")
        
        assert not late_flags[EARLY_CHECK]
        assert not late_flags[late_flags.index < pd.Timestamp("2020-07-01")].any()
        # It does pass at the end, which is all the old end-of-sample screen looked at
        assert late_flags.iloc[-1]
    
    @pytest.mark.parametrize("case", CASES, ids=list(CASES))
    def test_symbol_that_fails_only_late_is_eligible_early(self, case):
        late, early, overrides = CASES[case]()
        steady = _frame("STEADY", 50.0, 1_000_000)
        screened = _screen(pd.concat([late, early, steady], ignore_index=True), **overrides)
        early_flags = _flags(screened, "EARLY")
        steady_flags = _flags(screened, "STEADY")
        
        assert early_flags[EARLY_CHECK]
        span = (early_flags.index >= WINDOW_FILLED) & (early_flags.index < pd.Timestamp("2020-07-01"))
        assert early_flags[span].all()
        # The old end-of-sample screen dropped it from the whole backtest
        assert not early_flags.iloc[-1]
        assert steady_flags[span].all()
    
    @pytest.mark.parametrize("lookback_days", [100, None])
    def test_membership_ignores_future_rows(self, lookback_days):
        """Flags up to any date are identical whether or not later data exists."""
        rng = np.random.default_rng(7)
        frames = []
        for i in range(8):
            # Prices around $5, volume around 100k and volatility around 25%,
            # so membership flips often and every filter matters
            returns = rng.normal(0.0, rng.uniform(0.01, 0.018), len(DATES))
            close = rng.uniform(4.0, 12.0) * np.cumprod(1.0 + returns)
            volume = rng.lognormal(np.log(rng.uniform(8e4, 3e5)), 0.5, len(DATES))
            frames.append(_frame(f"S{i}", close, volume))
        prices = pd.concat(frames, ignore_index=True)
        kwargs = dict(lookback_days=lookback_days, volume_top_n=4)
        
        full = _screen(prices, **kwargs).set_index(["symbol", "date"])["in_universe"]
        assert 0.05 < full.mean() < 0.95
        
        for cut in ["2019-03-15", "2020-11-02", "2022-06-30"]:
            truncated = _screen(prices[prices["date"] <= cut], **kwargs)
            truncated = truncated.set_index(["symbol", "date"])["in_universe"]
            pd.testing.assert_series_equal(truncated, full.loc[truncated.index])


class TestScreenerOutput:
    def test_rows_are_flagged_not_dropped(self):
        late, early, _ = _liquidity_case()
        # Shuffled input: the output keeps the input's rows, order and columns
        prices = pd.concat([late, early], ignore_index=True).sample(frac=1.0, random_state=0)
        original = prices.copy()
        
        screened = _screen(prices)
        
        pd.testing.assert_frame_equal(screened.drop(columns="in_universe"), original)
        pd.testing.assert_frame_equal(prices, original)
        assert screened["in_universe"].dtype == bool
    
    def test_lookback_trading_days(self):
        # 730 * 1.5 = 1095 calendar days = 756 trading days
        assert lookback_trading_days(730) == 756
        assert lookback_trading_days(100) == 104
        assert lookback_trading_days(None) == 0
        assert lookback_trading_days(0) == 0


class TestFactorSignalUniverse:
    def _df(self):
        dates = pd.bdate_range("2024-01-01", periods=2)
        return pd.DataFrame({
            "date": np.repeat(dates, 3),
            "symbol": ["A", "B", "C"] * 2,
            "factor": [3.0, 2.0, 1.0] * 2,
            # A has the best factor but only joins the universe on day 1
            "in_universe": [False, True, True, True, True, True],
        }), dates
    
    def test_only_eligible_rows_are_ranked(self):
        df, dates = self._df()
        
        signals = use_factor_as_signal(df, factor_column="factor", top_n=1)
        
        assert list(signals.loc[signals["date"] == dates[0], "symbol"]) == ["B"]
        assert list(signals.loc[signals["date"] == dates[1], "symbol"]) == ["A"]
    
    def test_universe_is_optional(self):
        df, dates = self._df()
        
        without_column = use_factor_as_signal(df.drop(columns="in_universe"), factor_column="factor", top_n=1)
        disabled = use_factor_as_signal(df, factor_column="factor", top_n=1, universe_column=None)
        
        assert list(without_column["symbol"]) == ["A", "A"]
        assert list(disabled["symbol"]) == ["A", "A"]


class TestScreenerWarmup:
    def test_infer_warmup_includes_screener_lookback(self):
        from qsresearch.backtest.run_backtest import _infer_warmup_days
        from qsresearch.strategies.factor.config import MOMENTUM_FACTOR_CONFIG, VALUE_FACTOR_CONFIG
        
        momentum = MOMENTUM_FACTOR_CONFIG
        # lookback_days=730 -> 1095 calendar days -> 756 trading days, longer than slow_period=252
        assert _infer_warmup_days(momentum["factors"], momentum["preprocessing"]) == 756
        # VALUE_FACTOR_CONFIG leaves lookback_days at the screener default (730)
        assert _infer_warmup_days([], VALUE_FACTOR_CONFIG["preprocessing"]) == 756
        
        screener = lambda params: [{"name": "universe_screener", "params": params}]
        assert _infer_warmup_days(momentum["factors"], screener({"lookback_days": 100})) == 252
        assert _infer_warmup_days([], screener({"lookback_days": None})) == 0
        # Factor-only calls are unchanged
        assert _infer_warmup_days(momentum["factors"]) == 252
    
    def test_run_backtest_never_trades_a_symbol_before_it_joins(self, monkeypatch, tmp_path):
        module = importlib.import_module("qsresearch.backtest.run_backtest")
        
        all_dates = pd.bdate_range("2019-01-01", "2023-06-30")
        rng = np.random.default_rng(1)
        frames = []
        # STAR has by far the best momentum but trades 10k shares a day until 2023
        drifts = {"STAR": 0.003, **{f"S{i}": 0.0001 * i for i in range(5)}}
        for symbol, drift in drifts.items():
            returns = rng.normal(drift, 0.01, len(all_dates))
            returns[0] = 0.0
            volume = np.full(len(all_dates), 1_000_000.0)
            if symbol == "STAR":
                volume[all_dates < pd.Timestamp("2023-01-01")] = 10_000.0
            frames.append(pd.DataFrame({
                "date": all_dates,
                "symbol": symbol,
                "close": 100.0 * np.cumprod(1.0 + returns),
                "volume": volume,
            }))
        prices = pd.concat(frames, ignore_index=True)
        requested = {}
        
        def fake_load(bundle_name, start_date, end_date):
            requested["start"] = start_date
            return prices[(prices.date >= start_date) & (prices.date <= end_date)].reset_index(drop=True)
        
        captured = {}
        simulate = module._simulate_portfolio
        
        def capture_signals(signals, *args, **kwargs):
            captured["signals"] = signals
            return simulate(signals, *args, **kwargs)
        
        monkeypatch.setattr(module, "_load_price_data", fake_load)
        monkeypatch.setattr(module, "_simulate_portfolio", capture_signals)
        config = {
            "start_date": "2022-07-01",
            "end_date": "2023-06-30",
            "simulation": {"rebalance_frequency": "D"},
            "preprocessing": [{"name": "universe_screener", "params": {
                "lookback_days": 400, "volume_top_n": None, "volatility_filter": False,
                "min_avg_volume": 100_000, "min_avg_price": 4.0, "min_last_price": 5.0,
            }}],
            "factors": [{"name": "momentum_factor", "func": "qsresearch.features.momentum:add_qsmom_features",
                         "params": {"fast_period": 21, "slow_period": 252, "signal_period": 126}}],
            "algorithm": {"params": {"factor_column": "close_qsmom_21_252_126", "top_n": 3}},
        }
        
        results = module.run_backtest(config, output_dir=tmp_path, log_to_mlflow=False)
        signals = captured["signals"]
        star_dates = pd.to_datetime(signals.loc[signals["symbol"] == "STAR", "date"])
        in_window = pd.to_datetime(signals["date"]) >= pd.Timestamp("2022-07-01")
        
        # 400 * 1.5 = 600 calendar days of screening history = 415 trading days of warm-up
        assert results["warmup_days"] == 415
        assert len(pd.bdate_range(requested["start"], "2022-06-30")) >= 415
        # The old screener saw STAR's 2023 volume and traded it from the first day
        assert (star_dates >= pd.Timestamp("2023-01-01")).all()
        assert star_dates.max() == pd.Timestamp("2023-06-30")
        # Before STAR joins, the book is filled from the symbols that were eligible
        assert signals.loc[in_window, "date"].min() == pd.Timestamp("2022-07-01")
