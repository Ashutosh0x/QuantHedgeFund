"""
Automation - Prefect Flows

Production-level Prefect flows for nightly automation.
Based on the Quant Science production deployment architecture.
"""

from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import json

from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

from config.settings import get_settings


# =============================================================================
# Task Definitions
# =============================================================================

@task(
    name="refresh_database",
    retries=3,
    retry_delay_seconds=60,
    cache_key_fn=task_input_hash,
    cache_expiration=timedelta(hours=12),
)
def refresh_database(run_date: date) -> Dict[str, Any]:
    """
    Refresh the DuckDB database with latest market data.
    
    This task downloads new price and fundamental data from FMP.
    """
    logger = get_run_logger()
    logger.info(f"Refreshing database for {run_date}")
    
    from qsconnect import Client
    
    client = Client()
    
    # Download latest prices
    prices = client.bulk_historical_prices(
        start_date=run_date - timedelta(days=5),
        end_date=run_date,
    )
    
    logger.info(f"Downloaded {len(prices)} price records")
    
    client.close()
    
    return {
        "run_date": run_date.isoformat(),
        "records_updated": len(prices),
    }


@task(
    name="run_nightly_backtest",
    retries=1,
    retry_delay_seconds=300,
)
def run_nightly_backtest(
    run_date: date,
    config_name: str = "momentum_factor",
) -> Dict[str, Any]:
    """
    Run the nightly production backtest.
    
    Executes the configured strategy and logs to MLflow.
    """
    logger = get_run_logger()
    logger.info(f"Running nightly backtest for {run_date}")
    
    from qsresearch.backtest.run_backtest import run_backtest
    from qsresearch.strategies.factor.config import MOMENTUM_FACTOR_CONFIG
    
    # Override config with run date
    config = MOMENTUM_FACTOR_CONFIG.copy()
    config["end_date"] = run_date.isoformat()
    config["run_name"] = f"Nightly Backtest/{run_date.isoformat()}"
    
    results = run_backtest(config, log_to_mlflow=True)
    
    metrics = results.get("metrics", {})
    
    logger.info(f"Backtest completed. Sharpe: {metrics.get('portfolio_daily_sharpe', 0):.4f}")
    
    return {
        "run_date": run_date.isoformat(),
        "total_return": metrics.get("portfolio_total_return", 0),
        "sharpe_ratio": metrics.get("portfolio_daily_sharpe", 0),
        "max_drawdown": metrics.get("portfolio_max_drawdown", 0),
    }


@task(
    name="run_parameter_sweep",
    retries=1,
    retry_delay_seconds=300,
)
def run_parameter_sweep(
    run_date: date,
    sweep_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run parameter sweep for strategy optimization.
    
    Tests multiple combinations of strategy parameters and selects one
    walk-forward; the reported Sharpe is the out-of-sample one.
    """
    logger = get_run_logger()
    logger.info(f"Running parameter sweep with {len(sweep_config.get('combinations', []))} combinations")
    
    from qsresearch.backtest.parameter_sweep import run_iterative_sweep
    
    sweep = run_iterative_sweep(
        sweep_config=sweep_config,
        run_date=run_date,
    )
    walk_forward = sweep["walk_forward"]
    
    # The best full-sample Sharpe is in-sample and optimistic; report the
    # walk-forward (out-of-sample) result instead
    if walk_forward is None:
        logger.warning("Walk-forward selection did not run; no parameters selected")
    else:
        logger.info(
            f"Selected {sweep['selected_params']}: out-of-sample Sharpe "
            f"{walk_forward['oos_sharpe']:.4f} (best in-sample {walk_forward['in_sample_best_sharpe']:.4f})"
        )
    
    return {
        "run_date": run_date.isoformat(),
        "combinations_tested": len(sweep["combinations"]),
        "oos_sharpe": walk_forward["oos_sharpe"] if walk_forward else None,
        "in_sample_best_sharpe": walk_forward["in_sample_best_sharpe"] if walk_forward else None,
        "selected_params": sweep["selected_params"],
    }


@task(
    name="generate_dashboard_snapshot",
    retries=2,
    retry_delay_seconds=60,
)
def generate_dashboard_snapshot(run_date: date) -> Dict[str, Any]:
    """
    Generate dashboard data snapshot.
    
    Creates the data files needed by the Streamlit dashboard.
    """
    logger = get_run_logger()
    logger.info(f"Generating dashboard snapshot for {run_date}")
    
    settings = get_settings()
    
    # Load latest backtest results
    import pickle
    from pathlib import Path
    
    backtest_dir = settings.dashboard_data_dir / "backtests"
    latest_backtest = sorted(backtest_dir.glob("*.pkl"))[-1] if backtest_dir.exists() else None
    
    if latest_backtest:
        with open(latest_backtest, "rb") as f:
            backtest_data = pickle.load(f)
        
        # Generate snapshot
        snapshot = {
            "run_date": run_date.isoformat(),
            "backtest_file": str(latest_backtest),
            "metrics": backtest_data.get("metrics", {}),
        }
        
        # Save snapshot
        snapshot_path = settings.dashboard_data_dir / "latest_snapshot.json"
        with open(snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        logger.info(f"Dashboard snapshot saved to {snapshot_path}")
        
        return snapshot
    
    return {"error": "No backtest data found"}


@task(
    name="deploy_model",
    retries=1,
)
def deploy_model(
    model_run_id: str,
    alias: str = "live-trading",
) -> Dict[str, Any]:
    """
    Deploy a model to production.
    
    Promotes a specific MLflow run to the live trading alias.
    """
    logger = get_run_logger()
    logger.info(f"Deploying model {model_run_id} with alias {alias}")
    
    import mlflow
    
    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    
    # Get run info
    run = mlflow.get_run(model_run_id)
    
    # Create deployment state
    deployment_state = {
        "run_id": model_run_id,
        "alias": alias,
        "deployed_at": datetime.now().isoformat(),
        "artifact_uri": run.info.artifact_uri,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
    }
    
    # Save deployment state
    deployment_path = settings.dashboard_data_dir / "deployment_state.json"
    with open(deployment_path, "w") as f:
        json.dump(deployment_state, f, indent=2)
    
    logger.info(f"Model deployed: {deployment_state}")
    
    return deployment_state


# =============================================================================
# Flow Definitions
# =============================================================================

@flow(
    name="nightly-qsconnect-refresh",
    description="Nightly database refresh with latest market data",
)
def nightly_database_refresh_flow():
    """Nightly flow to refresh the database."""
    run_date = date.today()
    return refresh_database(run_date)


@flow(
    name="nightly-backtest",
    description="Nightly production backtest",
)
def nightly_backtest_flow():
    """Nightly flow to run the production backtest."""
    run_date = date.today()
    
    # Refresh data first
    refresh_result = refresh_database(run_date)
    
    # Run backtest
    backtest_result = run_nightly_backtest(run_date)
    
    return {
        "refresh": refresh_result,
        "backtest": backtest_result,
    }


@flow(
    name="nightly-iterative-sweep",
    description="Nightly parameter sweep for strategy optimization",
)
def nightly_sweep_flow():
    """Nightly flow to run parameter sweeps."""
    run_date = date.today()
    
    # Default sweep configuration
    sweep_config = {
        "base_config": "momentum_factor",
        "combinations": [
            {"fast_period": 21, "slow_period": 252, "top_n": 20},
            {"fast_period": 21, "slow_period": 252, "top_n": 30},
            {"fast_period": 63, "slow_period": 252, "top_n": 20},
            {"fast_period": 21, "slow_period": 504, "top_n": 20},
        ],
        "rounds": 6,
    }
    
    return run_parameter_sweep(run_date, sweep_config)


@flow(
    name="nightly-dashboard-snapshot",
    description="Generate dashboard data snapshot",
)
def nightly_dashboard_flow():
    """Nightly flow to update dashboard data."""
    run_date = date.today()
    return generate_dashboard_snapshot(run_date)


@flow(
    name="nightly-pipeline-orchestrator",
    description="Master orchestrator for all nightly jobs",
)
def nightly_orchestrator_flow():
    """
    Master orchestrator that runs all nightly jobs in sequence.
    
    Schedule: At 01:00 AM, Tuesday through Saturday
    """
    run_date = date.today()
    
    # Step 1: Refresh database
    refresh_result = refresh_database(run_date)
    
    # Step 2: Run parameter sweep (for research)
    sweep_config = {"combinations": [], "rounds": 6}
    sweep_result = run_parameter_sweep(run_date, sweep_config)
    
    # Step 3: Run production backtest
    backtest_result = run_nightly_backtest(run_date)
    
    # Step 4: Generate dashboard snapshot
    dashboard_result = generate_dashboard_snapshot(run_date)
    
    return {
        "run_date": run_date.isoformat(),
        "refresh": refresh_result,
        "sweep": sweep_result,
        "backtest": backtest_result,
        "dashboard": dashboard_result,
    }


@flow(
    name="deploy-live-trading-model",
    description="Deploy a model to live trading",
)
def deploy_model_flow(model_run_id: str, alias: str = "live-trading"):
    """Flow to deploy a model to production."""
    return deploy_model(model_run_id, alias)


# =============================================================================
# Deployment Setup
# =============================================================================

def create_deployments():
    """Create Prefect deployments for all flows."""
    
    # Nightly orchestrator - runs at 1 AM Tuesday-Saturday
    Deployment.build_from_flow(
        flow=nightly_orchestrator_flow,
        name="nightly-pipeline-orchestrator",
        schedule=CronSchedule(cron="0 1 * * 2-6"),  # 1 AM Tue-Sat
        work_pool_name="default",
    ).apply()
    
    # Database refresh - runs at 10 PM daily
    Deployment.build_from_flow(
        flow=nightly_database_refresh_flow,
        name="dgx-nightly-refresh",
        schedule=CronSchedule(cron="0 22 * * *"),  # 10 PM daily
        work_pool_name="default",
    ).apply()
    
    # Backtest - runs at 2 AM daily
    Deployment.build_from_flow(
        flow=nightly_backtest_flow,
        name="nightly-backtest",
        schedule=CronSchedule(cron="0 2 * * *"),  # 2 AM daily
        work_pool_name="default",
    ).apply()
    
    # Dashboard snapshot - runs at 3 AM daily
    Deployment.build_from_flow(
        flow=nightly_dashboard_flow,
        name="nightly-dashboard-snapshot",
        schedule=CronSchedule(cron="0 3 * * *"),  # 3 AM daily
        work_pool_name="default",
    ).apply()
    
    print("✅ All deployments created successfully!")


if __name__ == "__main__":
    create_deployments()
