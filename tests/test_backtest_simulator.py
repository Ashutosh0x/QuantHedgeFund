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
        assert list(day1["symbol"]) == ["C"] and day1["weight"].iloc[0] == pytest.approx(1.0)
        assert set(signals.loc[signals["date"] == dates[2], "symbol"]) == {"A", "B"}
        assert signals["factor_value"].notna().all()


class TestDelisting:
    """A held position must not ride a stale forward-filled price through a delisting."""
    
    def _prices(self, dates, delist_after=4):
        # A gains 1% a day and stops trading after day `delist_after`; B is flat to the end
        prices = _prices_from_returns({"A": 0.01, "B": 0.0}, dates)
        last_a = dates[delist_after]
        return prices[(prices["symbol"] == "B") | (prices["date"] <= last_a)].reset_index(drop=True)
    
    def _run(self, prices, dates, **kwargs):
        signals = pd.DataFrame({"date": [dates[0]], "symbol": ["A"], "weight": [1.0]})
        return _simulate_portfolio(
            signals, prices, 100.0, str(dates[0].date()), str(dates[-1].date()),
            rebalance_frequency="D", transaction_cost_bps=0.0, **kwargs,
        )
    
    def test_held_position_takes_delisting_return_then_cash(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        
        perf = self._run(self._prices(dates), dates).set_index("date")
        
        # Days 1-4 earn A's 1%; day 5 is the delisting day; then the book is cash
        assert perf.loc[dates[4], "returns"] == pytest.approx(0.01)
        assert perf.loc[dates[5], "returns"] == pytest.approx(-0.30)
        assert perf.loc[dates[5], "delisted_weight"] == pytest.approx(1.0)
        assert (perf.loc[dates[6]:, "returns"] == 0.0).all()
        assert perf["portfolio_value"].iloc[-1] == pytest.approx(100.0 * 1.01 ** 4 * 0.70)
        # Leaving the delisted position is a payout, not a trade
        assert perf.loc[dates[5], "turnover"] == pytest.approx(0.0)
    
    def test_delisting_return_is_configurable(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        
        perf = self._run(self._prices(dates), dates, delisting_return=0.0)
        
        assert perf["portfolio_value"].iloc[-1] == pytest.approx(100.0 * 1.01 ** 4)
        with pytest.raises(ValueError):
            self._run(self._prices(dates), dates, delisting_return=-1.5)
    
    def test_symbol_trading_to_the_end_is_not_delisted(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        prices = _prices_from_returns({"A": 0.01, "B": 0.0}, dates)
        
        perf = self._run(prices, dates)
        
        assert perf["delisted_weight"].sum() == 0.0
        assert perf["portfolio_value"].iloc[-1] == pytest.approx(100.0 * 1.01 ** 9)
    
    def test_rows_dropped_by_preprocessing_are_not_a_delisting(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        raw = _prices_from_returns({"A": 0.01, "B": 0.0}, dates)
        # Preprocessing dropped A's last two rows, but the raw data shows it still trading
        processed = raw[~((raw["symbol"] == "A") & (raw["date"] > dates[7]))]
        last_trades = raw.groupby("symbol")["date"].max()
        
        perf = self._run(processed, dates, last_trade_dates=last_trades)
        
        assert perf["delisted_weight"].sum() == 0.0
        assert perf["returns"].min() >= 0.0
    
    def test_delisted_symbol_is_never_bought_back(self):
        dates = pd.bdate_range("2024-01-01", periods=10)
        prices = self._prices(dates)
        # A stale signal for A after it delisted must not reopen the position
        signals = pd.DataFrame({
            "date": [dates[0], dates[7]],
            "symbol": ["A", "A"],
            "weight": [1.0, 1.0],
        })
        
        perf = _simulate_portfolio(
            signals, prices, 100.0, str(dates[0].date()), str(dates[-1].date()),
            rebalance_frequency="D", transaction_cost_bps=0.0,
        ).set_index("date")
        
        assert perf.loc[dates[7], "turnover"] == pytest.approx(0.0)
        assert (perf.loc[dates[6]:, "returns"] == 0.0).all()
    
    def test_run_backtest_uses_raw_last_trade_dates(self, monkeypatch, tmp_path):
        """A thin last day removed by remove_low_volume must not count as a delisting."""
        module = importlib.import_module("qsresearch.backtest.run_backtest")
        
        all_dates = pd.bdate_range("2021-01-01", "2023-06-30")
        prices = _prices_from_returns({f"S{i}": 0.0005 * i for i in range(5)}, all_dates)
        prices["volume"] = 1_000_000.0
        # The top-momentum symbol trades almost nothing on the final day
        prices.loc[(prices["symbol"] == "S4") & (prices["date"] == all_dates[-1]), "volume"] = 10.0
        monkeypatch.setattr(
            module, "_load_price_data",
            lambda b, s, e: prices[(prices.date >= s) & (prices.date <= e)].reset_index(drop=True),
        )
        config = {
            "start_date": "2022-07-01",
            "end_date": "2023-06-30",
            "preprocessing": [{"name": "price_preprocessor", "params": {
                "min_trading_days": 0, "remove_low_trading_days": False,
                "remove_large_gaps": False, "remove_low_volume": True,
            }}],
            "factors": [{"name": "momentum_factor", "func": "qsresearch.features.momentum:add_qsmom_features",
                         "params": {"slow_period": 252}}],
            "algorithm": {"params": {"factor_column": "close_qsmom_21_252_126", "top_n": 2}},
        }
        
        perf = module.run_backtest(config, output_dir=tmp_path, log_to_mlflow=False)["performance"]
        
        assert perf["delisted_weight"].sum() == 0.0
        assert perf["returns"].iloc[-1] > -0.05
