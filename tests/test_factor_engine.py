"""
Tests for FactorEngine.create_composite_factor.

Factors are z-scored across symbols within each date. The old version z-scored
over the whole frame, so every past composite was scaled by the mean and spread
of factor values from later dates.
"""

import numpy as np
import pandas as pd
import pytest

from qsresearch.features.factor_engine import FactorEngine


def _frame(values_by_date: dict) -> pd.DataFrame:
    rows = [
        {"date": pd.Timestamp(date), "symbol": symbol, "f1": f1, "f2": f2}
        for date, by_symbol in values_by_date.items()
        for symbol, (f1, f2) in by_symbol.items()
    ]
    return pd.DataFrame(rows)


class TestCompositeFactor:
    def test_zscores_within_each_date(self):
        df = _frame({
            "2024-01-02": {"A": (1.0, 10.0), "B": (2.0, 20.0), "C": (3.0, 30.0)},
            # Every value is 100 higher on day two; a whole-frame z-score made all of
            # day one's composites strongly negative because of it
            "2024-01-03": {"A": (101.0, 110.0), "B": (103.0, 120.0), "C": (102.0, 130.0)},
        })
        
        result = FactorEngine().create_composite_factor(df, ["f1", "f2"])
        composite = result.set_index(["date", "symbol"])["composite_factor"]
        
        day_one = composite.loc[pd.Timestamp("2024-01-02")]
        assert day_one.to_dict() == pytest.approx({"A": -1.0, "B": 0.0, "C": 1.0}, abs=1e-5)
        assert composite.groupby(level="date").mean().to_numpy() == pytest.approx([0.0, 0.0], abs=1e-9)
    
    def test_composite_ignores_future_rows(self):
        rng = np.random.default_rng(2)
        dates = pd.bdate_range("2024-01-01", periods=40)
        df = pd.DataFrame({
            "date": np.repeat(dates, 5),
            "symbol": ["A", "B", "C", "D", "E"] * len(dates),
            # Factor levels drift over time, as raw factor values often do
            "f1": rng.normal(0, 1, 5 * len(dates)) + np.repeat(np.arange(len(dates)), 5),
            "f2": rng.normal(0, 1, 5 * len(dates)),
        })
        engine = FactorEngine()
        
        full = engine.create_composite_factor(df, ["f1", "f2"], weights=[0.7, 0.3])
        cut = df["date"] <= dates[19]
        truncated = engine.create_composite_factor(df[cut], ["f1", "f2"], weights=[0.7, 0.3])
        
        pd.testing.assert_series_equal(truncated["composite_factor"], full.loc[cut, "composite_factor"])
    
    def test_single_symbol_date_and_no_date_column(self):
        df = _frame({
            "2024-01-02": {"A": (5.0, 1.0)},
            "2024-01-03": {"A": (1.0, 1.0), "B": (3.0, 3.0)},
        })
        engine = FactorEngine()
        
        result = engine.create_composite_factor(df, ["f1", "f2"])
        # One symbol has no cross-sectional spread, so it sits at 0
        assert result["composite_factor"].iloc[0] == pytest.approx(0.0)
        
        # Without a date column the frame is one cross-section
        single = df[df["date"] == pd.Timestamp("2024-01-03")].drop(columns="date")
        no_date = engine.create_composite_factor(single, ["f1", "f2"])
        assert no_date["composite_factor"].to_numpy() == pytest.approx(
            result["composite_factor"].iloc[1:].to_numpy()
        )
