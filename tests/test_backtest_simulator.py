"""
Tests for the backtest portfolio simulator (qsresearch.backtest.run_backtest).
"""

import importlib

import numpy as np
import pandas as pd
import pytest

from qsresearch.backtest.run_backtest import BacktestDataError, _simulate_portfolio
from qsresearch.strategies.factor.algorithms import use_factor_as_signal


def _prices_from_returns(daily_returns: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Build a long price frame from per-symbol constant or per-day returns."""
    frames = []
    for symbol, rets in daily_returns.items():
        rets = np.broadcast_to(np.asarray(rets, dtype=float), len(dates)).copy()
        rets[0] = 0.0
        close = 100.0 * np.cumprod(1.0 + rets)
        frames.append(pd.DataFrame({"date": dates, "symbol": symbol, "close": close}))
    return pd.concat(frames, ignore_index=True)


class TestSignalWeights:
    """The simulator must hold the signal weights, not the universe average."""

    def test_top_n_differs_from_universe_average(self):
        dates = pd.bdate_range("2024-01-01", periods=250)
        winners = {f"W{i}": 0.003 for i in range(3)}
        losers = {f"L{i}": -0.001 for i in range(7)}
        prices = _prices_from_returns({**winners, **losers}, dates)
        prices["factor"] = prices["symbol"].str.startswith("W").astype(float)

        top_n = use_factor_as_signal(prices, factor_column="factor", top_n=3)
        universe = prices[["date", "symbol"]].assign(weight=1.0 / 10)

        kwargs = dict(
            capital_base=1_000_000,
            start_date=str(dates[0].date()),
            end_date=str(dates[-1].date()),
            transaction_cost_bps=0.0,
        )
        top_perf = _simulate_portfolio(top_n, prices, **kwargs)
        universe_perf = _simulate_portfolio(universe, prices, **kwargs)

        assert set(top_n["symbol"]) == set(winners)
        # Held from day 0's close, so it earns every return from day 1 on
        expected_top = 1_000_000 * 1.003 ** (len(dates) - 1)
        assert top_perf["portfolio_value"].iloc[-1] == pytest.approx(expected_top, rel=1e-9)
        assert universe_perf["portfolio_value"].iloc[-1] < 1_000_000 * 1.001 ** len(dates)
        assert top_perf["portfolio_value"].iloc[-1] > 1.5 * universe_perf["portfolio_value"].iloc[-1]

    def test_cash_when_weights_sum_below_one(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        prices = _prices_from_returns({"A": 0.01}, dates)
        signals = pd.DataFrame({"date": [dates[0]], "symbol": ["A"], "weight": [0.5]})

        perf = _simulate_portfolio(
            signals, prices, 100.0, str(dates[0].date()), str(dates[-1].date()),
            transaction_cost_bps=0.0,
        )

        # Half invested: day-1 return is exactly half the asset's return
        assert perf["returns"].iloc[1] == pytest.approx(0.005)


class TestNoLookAhead:
    """A weight decided on day t must earn day t+1's return, not day t's."""

    def test_weight_on_day_t_uses_day_t_plus_1_return(self):
        dates = pd.bdate_range("2024-01-01", periods=6)
        t = 2
        a_returns = np.zeros(len(dates))
        a_returns[t] = 0.10      # move on the signal day itself: must NOT be earned
        a_returns[t + 1] = 0.05  # move on the next day: must be earned
        prices = _prices_from_returns({"A": a_returns, "B": 0.0}, dates)
        signals = pd.DataFrame({"date": [dates[t]], "symbol": ["A"], "weight": [1.0]})

        perf = _simulate_portfolio(
            signals, prices, 100.0, str(dates[0].date()), str(dates[-1].date()),
            rebalance_frequency="D", transaction_cost_bps=0.0,
        )
        returns = perf.set_index("date")["returns"]

        assert returns[dates[t]] == pytest.approx(0.0)
        assert returns[dates[t + 1]] == pytest.approx(0.05)
        assert perf["portfolio_value"].iloc[-1] == pytest.approx(105.0)


class TestRebalancingAndCosts:
    def test_monthly_rebalance_ignores_mid_month_signals(self):
        dates = pd.bdate_range("2024-01-01", "2024-02-29")
        prices = _prices_from_returns({"A": 0.001, "B": -0.001}, dates)
        # Signal flips to B mid-January; monthly rebalancing must keep A until February
        signals = pd.DataFrame({
            "date": dates,
            "symbol": ["A" if d < pd.Timestamp("2024-01-15") else "B" for d in dates],
            "weight": 1.0,
        })

        perf = _simulate_portfolio(
            signals, prices, 100.0, "2024-01-01", "2024-02-29",
            rebalance_frequency="M", transaction_cost_bps=0.0,
        ).set_index("date")

        traded = perf.index[perf["turnover"] > 0]
        assert list(traded) == [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")]
        assert perf.loc["2024-01-31", "returns"] == pytest.approx(0.001)
        assert perf.loc["2024-02-02", "returns"] == pytest.approx(-0.001)

    def test_cost_charged_on_turnover(self):
        dates = pd.bdate_range("2024-01-01", "2024-02-29")
        prices = _prices_from_returns({"A": 0.0, "B": 0.0}, dates)
        signals = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-02-01")],
            "symbol": ["A", "B"],
            "weight": [1.0, 1.0],
        })

        perf = _simulate_portfolio(
            signals, prices, 100.0, "2024-01-01", "2024-02-29",
            rebalance_frequency="M", transaction_cost_bps=10.0,
        ).set_index("date")

        # Buy A (turnover 1), then switch A -> B (turnover 2)
        assert perf.loc["2024-01-01", "turnover"] == pytest.approx(1.0)
        assert perf.loc["2024-02-01", "turnover"] == pytest.approx(2.0)
        assert perf["portfolio_value"].iloc[-1] == pytest.approx(100.0 * (1 - 0.001) * (1 - 0.002))


class TestMissingData:
    """Missing data must raise, never produce a synthetic curve."""

    def _signals(self):
        return pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "symbol": ["A"], "weight": [1.0]})

    def test_empty_prices_raise(self):
        empty = pd.DataFrame(columns=["date", "symbol", "close"])
        with pytest.raises(BacktestDataError):
            _simulate_portfolio(self._signals(), empty, 1_000_000, "2024-01-01", "2024-12-31")

    def test_missing_price_columns_raise(self):
        prices = pd.DataFrame({"date": [pd.Timestamp("2024-01-02")], "symbol": ["A"]})
        with pytest.raises(BacktestDataError, match="close"):
            _simulate_portfolio(self._signals(), prices, 1_000_000, "2024-01-01", "2024-12-31")

    def test_empty_signals_raise(self):
        dates = pd.bdate_range("2024-01-01", periods=5)
        prices = _prices_from_returns({"A": 0.01}, dates)
        with pytest.raises(BacktestDataError):
            _simulate_portfolio(pd.DataFrame(), prices, 1_000_000, "2024-01-01", "2024-12-31")

    def test_no_prices_in_window_raise(self):
        dates = pd.bdate_range("2020-01-01", periods=5)
        prices = _prices_from_returns({"A": 0.01}, dates)
        with pytest.raises(BacktestDataError):
            _simulate_portfolio(self._signals(), prices, 1_000_000, "2024-01-01", "2024-12-31")

    def test_random_fallback_is_gone(self):
        module = importlib.import_module("qsresearch.backtest.run_backtest")

        assert not hasattr(module, "_fallback_simulation")


class TestWarmup:
    """History before start_date is loaded so factors are ready on day one."""

    def test_infer_warmup_from_factor_lookbacks(self):
        from qsresearch.backtest.run_backtest import _infer_warmup_days
        from qsresearch.strategies.factor.config import MOMENTUM_FACTOR_CONFIG

        # slow_period=252 dominates; forward_periods=[21] is forward-looking and ignored
        assert _infer_warmup_days(MOMENTUM_FACTOR_CONFIG["factors"]) == 252
        assert _infer_warmup_days([{"name": "x", "params": {"periods": [5, 63]}}]) == 0
        assert _infer_warmup_days([{"name": "x", "params": {"roc_periods": [5, 63]}}]) == 63
        assert _infer_warmup_days([]) == 0

    def test_warmup_start_covers_requested_trading_days(self):
        from qsresearch.backtest.run_backtest import _warmup_start_date

        data_start = pd.Timestamp(_warmup_start_date("2015-01-02", 252))
        assert len(pd.bdate_range(data_start, "2015-01-01")) >= 252
        assert _warmup_start_date("2015-01-02", 0) == "2015-01-02"
        with pytest.raises(ValueError):
            _warmup_start_date("2015-01-02", -1)

    def test_run_backtest_trades_from_start_date(self, monkeypatch, tmp_path):
        module = importlib.import_module("qsresearch.backtest.run_backtest")

        all_dates = pd.bdate_range("2021-01-01", "2023-06-30")
        rng = np.random.default_rng(0)
        prices = _prices_from_returns(
            {f"S{i}": rng.normal(0.0004 * (i - 4), 0.01, len(all_dates)) for i in range(10)},
            all_dates,
        )
        requested = {}

        def fake_load(bundle_name, start_date, end_date):
            requested["start"] = start_date
            window = prices[(prices.date >= start_date) & (prices.date <= end_date)]
            return window.reset_index(drop=True)

        monkeypatch.setattr(module, "_load_price_data", fake_load)
        config = {
            "start_date": "2022-07-01",
            "end_date": "2023-06-30",
            "factors": [{"name": "momentum_factor", "func": "qsresearch.features.momentum:add_qsmom_features", "params": {
                "fast_period": 21, "slow_period": 252, "signal_period": 126,
            }}],
            "algorithm": {
                "callable": "qsresearch.strategies.factor.algorithms:use_factor_as_signal",
                "params": {"factor_column": "close_qsmom_21_252_126", "top_n": 3},
            },
        }

        results = module.run_backtest(config, output_dir=tmp_path, log_to_mlflow=False)
        perf = results["performance"]

        assert pd.Timestamp(requested["start"]) < pd.Timestamp("2021-07-01")
        assert results["warmup_days"] == 252
        assert perf["date"].iloc[0] == pd.Timestamp("2022-07-01")
        # Factors are ready, so the book is bought on the first day of the window
        assert perf["turnover"].iloc[0] == pytest.approx(1.0)

    def test_without_warmup_portfolio_idles_a_year(self, monkeypatch, tmp_path):
        """The old behaviour, kept reachable via warmup_days=0, for contrast."""
        module = importlib.import_module("qsresearch.backtest.run_backtest")

        all_dates = pd.bdate_range("2021-01-01", "2023-06-30")
        prices = _prices_from_returns({f"S{i}": 0.0005 * i for i in range(5)}, all_dates)
        monkeypatch.setattr(
            module, "_load_price_data",
            lambda b, s, e: prices[(prices.date >= s) & (prices.date <= e)].reset_index(drop=True),
        )
        config = {
            "start_date": "2022-01-03",
            "end_date": "2023-06-30",
            "simulation": {"warmup_days": 0},
            "factors": [{"name": "momentum_factor", "func": "qsresearch.features.momentum:add_qsmom_features",
                         "params": {"slow_period": 252}}],
            "algorithm": {"params": {"factor_column": "close_qsmom_21_252_126", "top_n": 2}},
        }

        perf = module.run_backtest(config, output_dir=tmp_path, log_to_mlflow=False)["performance"]
        first_trade = perf.loc[perf["turnover"] > 0, "date"].iloc[0]

        # The 252-bar lookback is only filled about a year into the window
        assert first_trade >= pd.Timestamp("2022-12-01")
        assert (perf["date"] < first_trade).sum() >= 240


class TestFactorSignal:
    def test_symbols_without_factor_value_are_not_selected(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        df = pd.DataFrame({
            "date": np.repeat(dates, 3),
            "symbol": ["A", "B", "C"] * 3,
            # Day 0: no factor values yet; day 1: only C; day 2: all three
            "factor": [np.nan, np.nan, np.nan, np.nan, np.nan, 1.0, 3.0, 2.0, 1.0],
        })

        signals = use_factor_as_signal(df, factor_column="factor", top_n=2)

        assert dates[0] not in set(signals["date"])
        day1 = signals[signals["date"] == dates[1]]
        # One qualifying name fills one of two slots; the other half stays cash
        assert list(day1["symbol"]) == ["C"] and day1["weight"].iloc[0] == pytest.approx(0.5)
        assert set(signals.loc[signals["date"] == dates[2], "symbol"]) == {"A", "B"}
        assert signals["factor_value"].notna().all()


class TestUniverseScreener:
    def test_latest_window_screen_keeps_full_history_of_passing_symbols(self):
        from qsresearch.preprocessors import universe_screener

        dates = pd.bdate_range("2015-01-01", "2024-12-31")
        prices = _prices_from_returns({"KEEP": 0.0003, "CHEAP": -0.002}, dates)
        prices["volume"] = 1_000_000

        screened = universe_screener(
            prices, lookback_days=730, volume_top_n=None, volatility_filter=False,
            min_avg_volume=100_000, min_avg_price=4.0, min_last_price=5.0,
            point_in_time=False,
        )

        # CHEAP ends far below $5 and is dropped; KEEP keeps all ten years, not
        # just the screening window, so factors can warm up before the backtest
        assert set(screened["symbol"]) == {"KEEP"}
        assert screened["date"].min() == dates[0]
        assert len(screened) == len(dates)


class TestPointInTimeScreener:
    """Universe membership on each date may only use data up to that date."""

    @staticmethod
    def _screen(prices, **kwargs):
        from qsresearch.preprocessors import universe_screener

        params = dict(
            lookback_days=730, volume_top_n=None, volatility_filter=False,
            min_avg_volume=100_000, min_avg_price=4.0, min_last_price=5.0,
        )
        params.update(kwargs)
        return universe_screener(prices, **params)

    @staticmethod
    def _price_path(dates, levels):
        """Piecewise-constant close: list of (start_date, price)."""
        close = pd.Series(np.nan, index=dates)
        for start, price in levels:
            close[close.index >= pd.Timestamp(start)] = price
        return close.to_numpy()

    def test_symbol_that_only_qualifies_late_is_not_eligible_early(self):
        dates = pd.bdate_range("2018-01-01", "2023-12-29")
        prices = pd.DataFrame({
            "date": dates, "symbol": "LATE", "volume": 1_000_000,
            "close": self._price_path(dates, [("2018-01-01", 2.0), ("2022-01-03", 50.0)]),
        })

        flags = self._screen(prices).set_index("date")["in_universe"]

        # A penny stock until 2022: the end-of-sample screen would have kept it all along
        assert not flags[:"2021-12-31"].any()
        assert flags["2022-06-01":].all()

    def test_symbol_that_fails_only_at_the_end_is_eligible_early(self):
        dates = pd.bdate_range("2018-01-01", "2023-12-29")
        prices = pd.DataFrame({
            "date": dates, "symbol": "FADE", "volume": 1_000_000,
            "close": self._price_path(dates, [("2018-01-01", 50.0), ("2023-06-01", 1.0)]),
        })

        point_in_time = self._screen(prices).set_index("date")["in_universe"]
        latest_window = self._screen(prices, point_in_time=False)

        # The end-of-sample screen drops FADE entirely (survivorship bias) ...
        assert latest_window.empty
        # ... point in time it is tradable until it actually collapses
        assert point_in_time["2018-03-01":"2023-05-31"].all()
        assert not point_in_time["2023-06-01":].any()

    def test_needs_min_history_before_eligible(self):
        dates = pd.bdate_range("2020-01-01", periods=60)
        prices = pd.DataFrame({"date": dates, "symbol": "NEW", "close": 20.0, "volume": 1_000_000})

        flags = self._screen(prices, min_history_days=21)["in_universe"].to_numpy()

        assert not flags[:20].any() and flags[20:].all()

    def test_volatility_uses_trailing_returns(self):
        dates = pd.bdate_range("2018-01-01", "2023-12-29")
        rets = np.zeros(len(dates))
        calm_until = dates.get_loc(pd.Timestamp("2022-01-03"))
        rets[calm_until:] = np.where(np.arange(len(dates) - calm_until) % 2, 0.05, -0.05)
        prices = _prices_from_returns({"WILD": rets}, dates)
        prices["volume"] = 1_000_000

        flags = self._screen(prices, volatility_filter=True, max_volatility=0.25).set_index("date")["in_universe"]

        assert flags["2018-03-01":"2021-12-31"].all()
        assert not flags["2022-06-01":].any()

    def test_volume_top_n_ranks_each_date_on_trailing_volume(self):
        dates = pd.bdate_range("2020-01-01", "2021-12-31")
        switch = pd.Timestamp("2021-01-04")
        frames = []
        for symbol, early, late in [("A", 3e6, 0.5e6), ("B", 2e6, 2e6), ("C", 1e6, 9e6)]:
            volume = np.where(dates < switch, early, late)
            frames.append(pd.DataFrame({"date": dates, "symbol": symbol, "close": 20.0, "volume": volume}))
        prices = pd.concat(frames, ignore_index=True)

        screened = self._screen(prices, volume_top_n=2)
        members = screened[screened["in_universe"]].groupby("date")["symbol"].apply(frozenset)

        assert members[pd.Timestamp("2020-06-01")] == {"A", "B"}
        # Trailing averages at year two: A ~1.75M, B 2M, C ~5M
        assert members[pd.Timestamp("2021-12-31")] == {"B", "C"}

    def test_signal_only_selects_symbols_in_universe(self):
        dates = pd.bdate_range("2024-01-01", periods=2)
        df = pd.DataFrame({
            "date": np.repeat(dates, 3),
            "symbol": ["A", "B", "C"] * 2,
            "factor": [3.0, 2.0, 1.0] * 2,
            "in_universe": [False, True, True, True, True, True],
        })

        signals = use_factor_as_signal(df, factor_column="factor", top_n=2)

        assert set(signals.loc[signals["date"] == dates[0], "symbol"]) == {"B", "C"}
        assert set(signals.loc[signals["date"] == dates[1], "symbol"]) == {"A", "B"}


class TestPointInTimePricePreprocessor:
    """price_preprocessor rules may only use data up to each row's date."""

    @staticmethod
    def _prices(dates, symbol="A", volume=1_000_000):
        return pd.DataFrame({
            "date": dates, "symbol": symbol, "open": 20.0, "high": 20.0, "low": 20.0,
            "close": 20.0, "volume": volume,
        })

    def test_history_rule_flags_rows_instead_of_trusting_total_length(self):
        from qsresearch.preprocessors import preprocess_price_data

        dates = pd.bdate_range("2020-01-01", periods=600)
        prices = self._prices(dates)

        point_in_time = preprocess_price_data(prices, min_trading_days=504)
        whole_sample = preprocess_price_data(prices, min_trading_days=504, point_in_time=False)

        # 600 rows in total pass the whole-sample rule from day one; point in
        # time the symbol only has 504 days of history from its 504th bar
        assert len(whole_sample) == 600 and "in_universe" not in whole_sample
        flags = point_in_time["in_universe"].to_numpy()
        assert not flags[:503].any() and flags[503:].all()

    def test_symbol_that_never_has_enough_history_is_dropped(self):
        from qsresearch.preprocessors import preprocess_price_data

        prices = pd.concat([
            self._prices(pd.bdate_range("2020-01-01", periods=600), "OLD"),
            self._prices(pd.bdate_range("2021-06-01", periods=100), "NEW"),
        ], ignore_index=True)

        result = preprocess_price_data(prices, min_trading_days=504)

        assert set(result["symbol"]) == {"OLD"}

    def test_low_volume_is_judged_against_trailing_volume(self):
        from qsresearch.preprocessors import preprocess_price_data

        dates = pd.bdate_range("2020-01-01", periods=600)
        volume = np.full(len(dates), 100_000.0)
        volume[300:] = 10_000_000.0  # volume grows 100x later on
        volume[450] = 50_000.0       # a genuinely thin day relative to its past
        prices = self._prices(dates, volume=volume)
        kwargs = dict(remove_low_trading_days=False, remove_large_gaps=False)

        point_in_time = preprocess_price_data(prices, **kwargs)
        whole_sample = preprocess_price_data(prices, point_in_time=False, **kwargs)

        # The whole-sample average (~5M) makes every early day look thin
        assert whole_sample["date"].min() == dates[300]
        # Point in time the early days are normal; only day 450 is removed
        assert set(dates) - set(point_in_time["date"]) == {dates[450]}

    def test_screener_respects_existing_flag(self):
        from qsresearch.preprocessors import preprocess_price_data, universe_screener

        dates = pd.bdate_range("2020-01-01", periods=600)
        prices = self._prices(dates)

        flagged = preprocess_price_data(prices, min_trading_days=504, remove_low_volume=False)
        screened = universe_screener(flagged, volume_top_n=None, volatility_filter=False)

        flags = screened["in_universe"].to_numpy()
        assert not flags[:503].any() and flags[503:].all()

    def test_warmup_covers_history_rule(self):
        from qsresearch.backtest.run_backtest import _infer_warmup_days
        from qsresearch.strategies.factor.config import MOMENTUM_FACTOR_CONFIG

        config = MOMENTUM_FACTOR_CONFIG
        assert _infer_warmup_days(config["factors"], config["preprocessing"]) == 504
        whole_sample = [{"name": "price_preprocessor", "params": {"min_trading_days": 504, "point_in_time": False}}]
        assert _infer_warmup_days(config["factors"], whole_sample) == 252

    def test_run_backtest_with_preprocessing_trades_from_start_date(self, monkeypatch, tmp_path):
        module = importlib.import_module("qsresearch.backtest.run_backtest")

        all_dates = pd.bdate_range("2019-01-01", "2023-06-30")
        rng = np.random.default_rng(1)
        prices = _prices_from_returns(
            {f"S{i}": rng.normal(0.0003 * (i - 3), 0.01, len(all_dates)) for i in range(8)},
            all_dates,
        )
        prices["volume"] = 1_000_000
        monkeypatch.setattr(
            module, "_load_price_data",
            lambda b, s, e: prices[(prices.date >= s) & (prices.date <= e)].reset_index(drop=True),
        )
        config = {
            "start_date": "2022-01-03",
            "end_date": "2023-06-30",
            "preprocessing": [{"name": "price_preprocessor", "params": {"min_trading_days": 504}}],
            "factors": [{"name": "momentum_factor", "func": "qsresearch.features.momentum:add_qsmom_features",
                         "params": {"slow_period": 252}}],
            "algorithm": {"params": {"factor_column": "close_qsmom_21_252_126", "top_n": 3}},
        }

        results = module.run_backtest(config, output_dir=tmp_path, log_to_mlflow=False)
        perf = results["performance"]

        assert results["warmup_days"] == 504
        assert perf["date"].iloc[0] == pd.Timestamp("2022-01-03")
        assert perf["turnover"].iloc[0] == pytest.approx(1.0)


class TestThinUniverse:
    """A universe smaller than top_n must not concentrate the book."""

    @staticmethod
    def _thin_day():
        return pd.DataFrame({
            "date": pd.Timestamp("2024-01-02"),
            "symbol": list("ABCDEFG"),
            "factor": [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            "in_universe": [True, True, True, False, False, False, False],
        })

    def test_slot_weighting_leaves_unfilled_slots_in_cash(self):
        signals = use_factor_as_signal(self._thin_day(), factor_column="factor", top_n=5)

        assert set(signals["symbol"]) == {"A", "B", "C"}
        assert signals["weight"].tolist() == pytest.approx([0.2, 0.2, 0.2])

    def test_equal_weighting_spreads_the_book_over_the_picks(self):
        signals = use_factor_as_signal(
            self._thin_day(), factor_column="factor", top_n=5, weighting="equal",
        )

        assert signals["weight"].tolist() == pytest.approx([1 / 3] * 3)

    def test_unknown_weighting_raises(self):
        with pytest.raises(ValueError):
            use_factor_as_signal(self._thin_day(), factor_column="factor", weighting="cap")

    def test_simulated_book_is_partly_cash_with_slot_weighting(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        prices = _prices_from_returns({s: 0.01 for s in "ABCDEFG"}, dates)
        prices["factor"] = prices["symbol"].map({s: 7.0 - i for i, s in enumerate("ABCDEFG")})
        prices["in_universe"] = prices["symbol"].isin(["A", "B", "C"])

        signals = use_factor_as_signal(prices, factor_column="factor", top_n=5)
        perf = _simulate_portfolio(
            signals, prices, 100.0, "2024-01-01", "2024-01-12", transaction_cost_bps=0.0,
        )

        # 60% invested in names returning 1% a day, 40% in cash
        assert perf["returns"].iloc[1] == pytest.approx(0.006)

    def test_relative_volatility_cap_keeps_universe_size_when_volatility_rises(self):
        from qsresearch.preprocessors import universe_screener

        dates = pd.bdate_range("2018-01-01", "2023-12-29")
        regime = np.where(dates >= pd.Timestamp("2021-01-04"), 3.0, 1.0)  # vol triples market-wide
        rng = np.random.default_rng(2)
        frames = []
        for i in range(10):
            rets = rng.normal(0, 0.006 * (1 + 0.15 * i), len(dates)) * regime
            frame = _prices_from_returns({f"S{i}": rets}, dates)
            frame["volume"] = 1_000_000
            frames.append(frame)
        prices = pd.concat(frames, ignore_index=True)
        common = dict(lookback_days=730, volume_top_n=None, min_history_days=21)

        absolute = universe_screener(prices, volatility_filter=True, max_volatility=0.25, **common)
        relative = universe_screener(
            prices, volatility_filter=False, max_volatility_percentile=0.8, **common,
        )
        count = lambda df: df[df["in_universe"]].groupby("date")["symbol"].nunique()

        # The absolute cap empties out once volatility rises; the relative cap
        # always keeps the calmest 80% (8 of 10)
        assert count(absolute).reindex(dates[dates >= "2023-01-02"], fill_value=0).max() < 3
        late = count(relative)[dates[dates >= "2019-01-02"]]
        assert (late == 8).all()
        # And it drops the most volatile names, not arbitrary ones
        last_day = relative[(relative["date"] == dates[-1]) & relative["in_universe"]]
        assert set(last_day["symbol"]) == {f"S{i}" for i in range(8)}
