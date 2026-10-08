"""
QS Research - Parameter Sweep Engine

Run iterative parameter sweeps to find optimal strategy configurations.
Based on the Quant Science production sweep methodology.

Picking the combination with the best full-sample Sharpe and reporting that
Sharpe is in-sample selection: with enough combinations, one looks good by
luck. The sweep therefore selects walk-forward. For each test window, the
combination with the best Sharpe over the preceding training window is held,
and the stitched test windows give an out-of-sample equity curve. Its Sharpe,
not the best full-sample one, is the figure to judge the strategy by.
"""

from typing import Dict, List, Any, Optional
from datetime import date
from pathlib import Path
import copy
import itertools
import json

import numpy as np
import pandas as pd
from loguru import logger

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from config.settings import get_settings
from qsresearch.backtest.run_backtest import DEFAULT_TRANSACTION_COST_BPS, run_backtest
from qsresearch.portfolio_analysis.performance_metrics import calculate_all_metrics


TRADING_DAYS_PER_YEAR = 252
DEFAULT_PARAM_GRID = {
    "fast_period": [21, 42, 63],
    "slow_period": [126, 252, 504],
    "top_n": [10, 20, 30],
}
SWEEPABLE_PARAMS = {"fast_period", "slow_period", "signal_period", "top_n"}
DEFAULT_WALK_FORWARD = {
    "train_days": 504,   # 2 years to choose on
    "test_days": 126,    # then hold the choice for 6 months
    "anchored": False,   # rolling training window; True grows it from the start
    "switch_cost_bps": None,  # default: 2x the base config's transaction cost
}


def run_iterative_sweep(
    sweep_config: Dict[str, Any],
    run_date: Optional[date] = None,
    experiment_name: str = "Momentum_Factor_Iterative_Sweep",
    max_combinations: int = 100,
    log_to_mlflow: bool = True,
) -> Dict[str, Any]:
    """
    Run an iterative parameter sweep with walk-forward selection.
    
    Every combination is backtested once over the full period. Because the
    backtest is point-in-time, a combination's return on a day uses only data
    up to that day, so walk_forward_selection can choose between the return
    streams without look-ahead.
    
    Args:
        sweep_config: Configuration for the sweep containing:
            - combinations: Optional explicit list of parameter dicts
            - param_grid: Dict of parameter names to lists of values, used
              when no combinations are given (fast_period, slow_period,
              signal_period, top_n)
            - walk_forward: Optional overrides for DEFAULT_WALK_FORWARD
        run_date: Date for the sweep (the backtests' end_date)
        experiment_name: MLflow experiment name
        max_combinations: Maximum number of combinations to test
        log_to_mlflow: Whether to log to MLflow
    
    Returns:
        Dictionary with:
            - combinations: one result per combination; its metrics are
              full-sample (in-sample) and must not be used to pick a winner
            - walk_forward: walk-forward summary (None if it could not run)
            - selected_params: parameters chosen on the most recent training
              window, to use from run_date on (None if walk-forward did not run)
    """
    if run_date is None:
        run_date = date.today()
    
    settings = get_settings()
    
    logger.info(f"Starting iterative parameter sweep for {run_date}")
    
    # Get base configuration
    from qsresearch.strategies.factor.config import MOMENTUM_FACTOR_CONFIG
    base_config = copy.deepcopy(MOMENTUM_FACTOR_CONFIG)
    
    combinations = _sweep_combinations(sweep_config)
    
    # Limit combinations
    if len(combinations) > max_combinations:
        logger.warning(f"Limiting from {len(combinations)} to {max_combinations} combinations")
        combinations = combinations[:max_combinations]
    
    logger.info(f"Testing {len(combinations)} parameter combinations")
    
    # Setup MLflow
    if log_to_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    results = []
    returns_by_run = {}
    
    for i, params in enumerate(combinations):
        # Build run name
        param_str = "_".join([f"{k[:3]}{v}" for k, v in params.items()])
        run_name = f"IterSweep/qsbacktest_{param_str}"
        
        logger.info(f"[{i+1}/{len(combinations)}] Testing: {params}")
        
        test_config = build_sweep_config(base_config, params)
        test_config["run_name"] = run_name
        test_config["end_date"] = run_date.isoformat()
        
        try:
            # Run backtest
            backtest_results = run_backtest(
                test_config,
                log_to_mlflow=log_to_mlflow,
            )
            
            metrics = backtest_results.get("metrics", {})
            performance = backtest_results["performance"]
            returns_by_run[run_name] = performance.set_index("date")["returns"]
            
            result = {
                "params": params,
                "run_name": run_name,
                "sample": "full (in-sample)",
                "total_return": metrics.get("portfolio_total_return", 0),
                "sharpe_ratio": metrics.get("portfolio_daily_sharpe", 0),
                "max_drawdown": metrics.get("portfolio_max_drawdown", 0),
                "calmar_ratio": metrics.get("portfolio_calmar", 0),
                "win_rate": metrics.get("portfolio_win_rate", 0),
                "success": True,
            }
            
            logger.info(f"  → Sharpe: {result['sharpe_ratio']:.4f}, Return: {result['total_return']:.2%}")
        
        except Exception as e:
            logger.error(f"  → Failed: {e}")
            result = {
                "params": params,
                "run_name": run_name,
                "success": False,
                "error": str(e),
            }
        
        results.append(result)
    
    # Walk-forward selection over the successful runs' daily returns
    walk_forward = None
    selected_params = None
    if returns_by_run:
        wf_settings = {**DEFAULT_WALK_FORWARD, **sweep_config.get("walk_forward", {})}
        if wf_settings["switch_cost_bps"] is None:
            sim_cost = base_config.get("simulation", {}).get("transaction_cost_bps", DEFAULT_TRANSACTION_COST_BPS)
            wf_settings["switch_cost_bps"] = 2 * sim_cost
        params_by_run = {r["run_name"]: r["params"] for r in results if r["success"]}
        try:
            walk_forward = walk_forward_selection(pd.DataFrame(returns_by_run), **wf_settings)
        except ValueError as e:
            logger.error(f"Walk-forward selection skipped: {e}")
        else:
            for fold in walk_forward["folds"]:
                fold["selected_params"] = params_by_run[fold["selected"]]
            selected_params = params_by_run[walk_forward["selected"]]
            walk_forward["selected_params"] = selected_params
            walk_forward["in_sample_best_params"] = params_by_run[walk_forward["in_sample_best"]]
            walk_forward["oos_metrics"] = _oos_metrics(walk_forward["oos_returns"])
            logger.info(
                f"Walk-forward out-of-sample Sharpe {walk_forward['oos_sharpe']:.4f} over "
                f"{len(walk_forward['folds'])} folds (best in-sample Sharpe "
                f"{walk_forward['in_sample_best_sharpe']:.4f} is optimistic); "
                f"selected for live: {selected_params}"
            )
            if log_to_mlflow and MLFLOW_AVAILABLE:
                _log_walk_forward_to_mlflow(walk_forward, wf_settings)
    
    # Save sweep results
    output_dir = settings.dashboard_data_dir / "sweeps"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"sweep_{run_date.isoformat()}.json"
    with open(output_path, "w") as f:
        json.dump(
            {"combinations": results, "walk_forward": _walk_forward_summary(walk_forward)},
            f, indent=2, default=str,
        )
    
    logger.info(f"Sweep complete. Results saved to {output_path}")
    
    return {
        "combinations": results,
        "walk_forward": walk_forward,
        "selected_params": selected_params,
    }


def _sweep_combinations(sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Explicit combinations if given, else the cartesian product of param_grid."""
    combinations = sweep_config.get("combinations") or []
    if not combinations:
        param_grid = sweep_config.get("param_grid", DEFAULT_PARAM_GRID)
        names = list(param_grid.keys())
        combinations = [dict(zip(names, values)) for values in itertools.product(*param_grid.values())]
    for params in combinations:
        unknown = set(params) - SWEEPABLE_PARAMS
        if unknown:
            raise ValueError(f"Unknown sweep parameters {sorted(unknown)}; sweepable: {sorted(SWEEPABLE_PARAMS)}")
    return [dict(params) for params in combinations]


def build_sweep_config(base_config: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copy base_config with the sweep parameters applied.
    
    The copy is deep, so the shared base config is never modified. Changing a
    momentum period renames the factor column (close_qsmom_{fast}_{slow}_{signal}),
    so the algorithm's factor_column follows it.
    """
    config = copy.deepcopy(base_config)
    
    period_keys = ("fast_period", "slow_period", "signal_period")
    for factor in config.get("factors", []):
        if factor.get("name") != "momentum_factor":
            continue
        factor_params = factor.setdefault("params", {})
        for key in period_keys:
            if key in params:
                factor_params[key] = params[key]
        factor_column = (
            f"close_qsmom_{factor_params.get('fast_period', 21)}"
            f"_{factor_params.get('slow_period', 252)}"
            f"_{factor_params.get('signal_period', 126)}"
        )
        algorithm_params = config.setdefault("algorithm", {}).setdefault("params", {})
        if str(algorithm_params.get("factor_column", "close_qsmom_")).startswith("close_qsmom_"):
            algorithm_params["factor_column"] = factor_column
    
    if "top_n" in params:
        config.setdefault("algorithm", {}).setdefault("params", {})["top_n"] = params["top_n"]
    
    return config


def _sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe, computed as in performance_metrics (population std)."""
    returns = returns.dropna()
    std = returns.std(ddof=0)
    if len(returns) == 0 or not std > 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def walk_forward_selection(
    returns: pd.DataFrame,
    train_days: int = DEFAULT_WALK_FORWARD["train_days"],
    test_days: int = DEFAULT_WALK_FORWARD["test_days"],
    anchored: bool = DEFAULT_WALK_FORWARD["anchored"],
    switch_cost_bps: float = 2 * DEFAULT_TRANSACTION_COST_BPS,
) -> Dict[str, Any]:
    """
    Choose between strategy return streams walk-forward.
    
    For each fold, the column with the best Sharpe over the training window
    (the train_days trading days before the test window, or all earlier days
    if anchored) is held over the next test_days. Ties go to the earlier
    column. The choice for a test window never sees that window's returns.
    Switching to a different column at a fold boundary is charged
    switch_cost_bps once (2x the trading cost covers selling one book and
    buying another).
    
    Args:
        returns: Daily returns, dates x candidates (e.g. run names)
        train_days: Trading days of history to choose on
        test_days: Trading days to hold each choice
        anchored: Grow the training window from the first date
        switch_cost_bps: Cost of switching candidates, in basis points
    
    Returns:
        Dictionary with folds, oos_returns (Series), oos_sharpe,
        oos_total_return, in_sample_best / in_sample_best_sharpe (the biased
        full-sample pick, for comparison) and selected / selected_train_sharpe
        (the choice on the most recent training window, to use next)
    
    Raises:
        ValueError: If there is no full training window plus test data
    """
    if train_days < 1 or test_days < 1:
        raise ValueError(f"train_days and test_days must be >= 1, got {train_days}, {test_days}")
    if switch_cost_bps < 0:
        raise ValueError(f"switch_cost_bps must be >= 0, got {switch_cost_bps}")
    returns = returns.sort_index().fillna(0.0)
    n = len(returns)
    if returns.shape[1] == 0 or n <= train_days:
        raise ValueError(
            f"Walk-forward needs more than train_days={train_days} trading days of returns "
            f"for at least one candidate; got {n} days, {returns.shape[1]} candidates"
        )
    
    def best(window: pd.DataFrame):
        scores = window.apply(_sharpe)
        return scores.idxmax(), float(scores.max())
    
    switch_cost = switch_cost_bps / 10_000
    folds = []
    oos_parts = []
    previous = None
    for i in range(train_days, n, test_days):
        train = returns.iloc[0 if anchored else i - train_days:i]
        test = returns.iloc[i:i + test_days]
        chosen, train_sharpe = best(train)
        fold_returns = test[chosen].copy()
        switched = previous is not None and chosen != previous
        if switched:
            fold_returns.iloc[0] = (1.0 + fold_returns.iloc[0]) * (1.0 - switch_cost) - 1.0
        folds.append({
            "train_start": train.index[0],
            "train_end": train.index[-1],
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "selected": chosen,
            "switched": switched,
            "train_sharpe": train_sharpe,
            "test_sharpe": _sharpe(test[chosen]),
        })
        oos_parts.append(fold_returns)
        previous = chosen
    
    oos_returns = pd.concat(oos_parts).rename("oos_returns")
    in_sample_best, in_sample_best_sharpe = best(returns)
    selected, selected_train_sharpe = best(returns if anchored else returns.iloc[-train_days:])
    
    return {
        "folds": folds,
        "oos_returns": oos_returns,
        "oos_sharpe": _sharpe(oos_returns),
        "oos_total_return": float((1.0 + oos_returns).prod() - 1.0),
        "oos_start": oos_returns.index[0],
        "oos_end": oos_returns.index[-1],
        "in_sample_best": in_sample_best,
        "in_sample_best_sharpe": in_sample_best_sharpe,
        "selected": selected,
        "selected_train_sharpe": selected_train_sharpe,
    }


def _oos_metrics(oos_returns: pd.Series) -> Dict[str, float]:
    """Full metric set for the stitched out-of-sample curve."""
    performance = pd.DataFrame({
        "date": oos_returns.index,
        "returns": oos_returns.to_numpy(),
        "portfolio_value": np.cumprod(1.0 + oos_returns.to_numpy()),
    })
    return calculate_all_metrics(performance)


def _walk_forward_summary(walk_forward: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """JSON-friendly walk-forward result (the daily OOS returns are left out)."""
    if walk_forward is None:
        return None
    return {k: v for k, v in walk_forward.items() if k != "oos_returns"}


def _log_walk_forward_to_mlflow(walk_forward: Dict[str, Any], wf_settings: Dict[str, Any]) -> None:
    """Log the out-of-sample result as its own run."""
    with mlflow.start_run(run_name="IterSweep/walk_forward", nested=True):
        mlflow.log_params({f"wf.{k}": v for k, v in wf_settings.items()})
        mlflow.log_params({f"selected.{k}": v for k, v in walk_forward["selected_params"].items()})
        mlflow.log_metric("oos_sharpe", walk_forward["oos_sharpe"])
        mlflow.log_metric("oos_total_return", walk_forward["oos_total_return"])
        mlflow.log_metric("in_sample_best_sharpe", walk_forward["in_sample_best_sharpe"])
        mlflow.log_metric("folds", len(walk_forward["folds"]))


def generate_sweep_report(results: Any) -> pd.DataFrame:
    """
    Generate a summary report from sweep results.
    
    The Sharpe ratios here are full-sample (in-sample); sorting by them shows
    which combinations did well in hindsight, not which to trade. Use the
    walk-forward result for that.
    
    Args:
        results: The dict returned by run_iterative_sweep, or its
            "combinations" list
    
    Returns:
        DataFrame with sweep summary
    """
    if isinstance(results, dict):
        results = results.get("combinations", [])
    
    rows = []
    
    for r in results:
        if r.get("success", False):
            row = {
                "run_name": r.get("run_name", ""),
                **r.get("params", {}),
                "total_return": r.get("total_return", 0),
                "sharpe_ratio": r.get("sharpe_ratio", 0),
                "max_drawdown": r.get("max_drawdown", 0),
                "calmar_ratio": r.get("calmar_ratio", 0),
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by Sharpe ratio
    if not df.empty:
        df = df.sort_values("sharpe_ratio", ascending=False)
    
    return df
