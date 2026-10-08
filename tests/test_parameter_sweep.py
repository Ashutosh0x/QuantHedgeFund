"""
Tests for walk-forward selection in qsresearch.backtest.parameter_sweep.

Choosing the combination with the best full-sample Sharpe, and reporting that
Sharpe, is in-sample selection: with many combinations one wins by luck. The
sweep picks walk-forward, so each choice is made only on data before the
period it is held for.
"""

import copy
import importlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from qsresearch.backtest.parameter_sweep import (
    build_sweep_config,
    generate_sweep_report,
    walk_forward_selection,
)
from qsresearch.strategies.factor.config import MOMENTUM_FACTOR_CONFIG


DATES = pd.bdate_range("2018-01-01", periods=1000)


def _regime_returns(seed: int = 0) -> pd.DataFrame:
    """EARLY earns in the first half and loses in the second; LATE is the mirror image."""
    rng = np.random.default_rng(seed)
    first_half = np.arange(len(DATES)) < len(DATES) // 2
    drift = np.where(first_half, 0.002, -0.002)
    return pd.DataFrame({
        "EARLY": drift + rng.normal(0, 0.01, len(DATES)),
        "LATE": -drift + rng.normal(0, 0.01, len(DATES)),
    }, index=DATES)


class TestWalkForwardSelection:
    def test_each_choice_uses_only_its_training_window(self):
        returns = _regime_returns()
        
        wf = walk_forward_selection(returns, train_days=200, test_days=100, switch_cost_bps=0.0)
        
        assert [f["test_start"] for f in wf["folds"]] == list(DATES[200::100])
        for fold in wf["folds"]:
            train = returns.loc[fold["train_start"]:fold["train_end"]]
            assert len(train) == 200
            assert fold["train_end"] < fold["test_start"]
            expected = max(train.columns, key=lambda c: train[c].mean() / train[c].std(ddof=0))
            assert fold["selected"] == expected
            test = returns.loc[fold["test_start"]:fold["test_end"], fold["selected"]]
            pd.testing.assert_series_equal(
                wf["oos_returns"].loc[fold["test_start"]:fold["test_end"]], test, check_names=False,
            )
        # EARLY is chosen while its training window is mostly the first half
        assert wf["folds"][0]["selected"] == "EARLY"
        assert wf["folds"][-1]["selected"] == "LATE"
    
    def test_choices_ignore_future_returns(self):
        """Folds that end before a cut are identical whether or not later returns exist."""
        returns = _regime_returns(seed=1)
        full = walk_forward_selection(returns, train_days=150, test_days=50)
        
        for cut in [400, 650, 900]:
            truncated = walk_forward_selection(returns.iloc[:cut], train_days=150, test_days=50)
            complete = [f for f in truncated["folds"] if len(returns.loc[f["test_start"]:f["test_end"]]) == 50]
            assert complete == full["folds"][:len(complete)]
            end = complete[-1]["test_end"]
            pd.testing.assert_series_equal(truncated["oos_returns"].loc[:end], full["oos_returns"].loc[:end])
    
    def test_best_in_sample_sharpe_overstates_noise(self):
        # 40 candidates with no edge at all: the best full-sample Sharpe looks good by luck
        rng = np.random.default_rng(42)
        noise = pd.DataFrame(rng.normal(0, 0.01, (len(DATES), 40)), index=DATES)
        
        wf = walk_forward_selection(noise, train_days=252, test_days=63)
        
        assert wf["in_sample_best_sharpe"] > 0.8
        assert wf["oos_sharpe"] < wf["in_sample_best_sharpe"] - 0.5
    
    def test_switch_cost_charged_on_change(self):
        returns = _regime_returns()
        
        free = walk_forward_selection(returns, train_days=200, test_days=100, switch_cost_bps=0.0)
        costly = walk_forward_selection(returns, train_days=200, test_days=100, switch_cost_bps=10.0)
        
        switches = [f for f in costly["folds"] if f["switched"]]
        assert switches
        for fold in costly["folds"]:
            day = fold["test_start"]
            expected = free["oos_returns"][day]
            if fold["switched"]:
                expected = (1 + expected) * (1 - 0.001) - 1
            assert costly["oos_returns"][day] == pytest.approx(expected)
    
    def test_final_selection_uses_latest_training_window(self):
        returns = _regime_returns()
        
        wf = walk_forward_selection(returns, train_days=200, test_days=100)
        
        assert wf["selected"] == "LATE"
        # The biased full-sample pick is reported for comparison only
        assert wf["in_sample_best"] in {"EARLY", "LATE"}
    
    def test_needs_a_full_training_window(self):
        with pytest.raises(ValueError, match="train_days"):
            walk_forward_selection(_regime_returns().iloc[:200], train_days=200, test_days=50)


class TestSweepConfig:
    def test_build_config_is_deep_and_renames_factor_column(self):
        base = copy.deepcopy(MOMENTUM_FACTOR_CONFIG)
        before = copy.deepcopy(base)
        
        config = build_sweep_config(base, {"fast_period": 63, "slow_period": 504, "top_n": 10})
        
        assert base == before
        momentum = next(f for f in config["factors"] if f["name"] == "momentum_factor")
        assert momentum["params"]["fast_period"] == 63
        assert momentum["params"]["slow_period"] == 504
        # Without the rename every non-default combination failed: column not found
        assert config["algorithm"]["params"]["factor_column"] == "close_qsmom_63_504_126"
        assert config["algorithm"]["params"]["top_n"] == 10
    
    def test_run_iterative_sweep(self, monkeypatch, tmp_path):
        module = importlib.import_module("qsresearch.backtest.parameter_sweep")
        before = copy.deepcopy(MOMENTUM_FACTOR_CONFIG)
        regimes = _regime_returns()
        seen = []
        
        def fake_run_backtest(config, log_to_mlflow=True):
            params = config["algorithm"]["params"]
            seen.append(params["factor_column"])
            # top_n=10 is the EARLY regime, top_n=30 the LATE one
            column = "EARLY" if params["top_n"] == 10 else "LATE"
            returns = regimes[column]
            return {
                "performance": pd.DataFrame({"date": DATES, "returns": returns.to_numpy()}),
                "metrics": {"portfolio_daily_sharpe": float(returns.mean() / returns.std() * np.sqrt(252))},
            }
        
        monkeypatch.setattr(module, "run_backtest", fake_run_backtest)
        monkeypatch.setattr(
            module, "get_settings",
            lambda: SimpleNamespace(dashboard_data_dir=tmp_path, mlflow_tracking_uri=""),
        )
        sweep_config = {
            "combinations": [{"fast_period": 21, "top_n": 10}, {"fast_period": 63, "top_n": 30}],
            "walk_forward": {"train_days": 200, "test_days": 100},
        }
        
        result = module.run_iterative_sweep(sweep_config, run_date=pd.Timestamp("2021-10-29").date(), log_to_mlflow=False)
        
        assert seen == ["close_qsmom_21_252_126", "close_qsmom_63_252_126"]
        assert MOMENTUM_FACTOR_CONFIG == before
        assert len(result["combinations"]) == 2
        assert result["selected_params"] == {"fast_period": 63, "top_n": 30}
        walk_forward = result["walk_forward"]
        assert walk_forward["folds"][0]["selected_params"] == {"fast_period": 21, "top_n": 10}
        assert walk_forward["oos_metrics"]["portfolio_daily_sharpe"] == pytest.approx(walk_forward["oos_sharpe"])
        saved = json.loads((tmp_path / "sweeps" / "sweep_2021-10-29.json").read_text())
        assert saved["walk_forward"]["selected_params"] == {"fast_period": 63, "top_n": 30}
        assert "oos_returns" not in saved["walk_forward"]
        assert not generate_sweep_report(result).empty
    
    def test_unknown_sweep_parameter_is_rejected(self, monkeypatch, tmp_path):
        module = importlib.import_module("qsresearch.backtest.parameter_sweep")
        monkeypatch.setattr(
            module, "get_settings",
            lambda: SimpleNamespace(dashboard_data_dir=tmp_path, mlflow_tracking_uri=""),
        )
        
        with pytest.raises(ValueError, match="lookback"):
            module.run_iterative_sweep({"combinations": [{"lookback": 5}]}, log_to_mlflow=False)
