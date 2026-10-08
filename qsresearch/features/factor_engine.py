"""
QS Research - Factor Engine

Central engine for calculating and managing quantitative factors.
"""

from typing import Dict, List, Callable, Any, Optional
import numpy as np
import pandas as pd
import polars as pl
from loguru import logger

from qsresearch.features.momentum import add_qsmom_features


class FactorEngine:
    """
    Engine for calculating and managing quantitative factors.
    
    The FactorEngine provides a unified interface for:
    - Registering factor calculation functions
    - Computing multiple factors on price data
    - Combining factors into composite signals
    """
    
    # Registry of built-in factors
    BUILTIN_FACTORS = {
        "qsmom": add_qsmom_features,
    }
    
    def __init__(self):
        """Initialize the factor engine."""
        self._custom_factors: Dict[str, Callable] = {}
        logger.info("Factor engine initialized")
    
    def register_factor(
        self,
        name: str,
        func: Callable[[pd.DataFrame, Any], pd.DataFrame],
    ) -> None:
        """
        Register a custom factor calculation function.
        
        Args:
            name: Name of the factor
            func: Function that takes a DataFrame and returns DataFrame with factor
        """
        self._custom_factors[name] = func
        logger.info(f"Registered custom factor: {name}")
    
    def calculate_factor(
        self,
        df: pd.DataFrame,
        factor_name: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Calculate a single factor.
        
        Args:
            df: Price DataFrame
            factor_name: Name of factor to calculate
            **kwargs: Additional arguments for the factor function
            
        Returns:
            DataFrame with factor added
        """
        # Check custom factors first
        if factor_name in self._custom_factors:
            func = self._custom_factors[factor_name]
        elif factor_name in self.BUILTIN_FACTORS:
            func = self.BUILTIN_FACTORS[factor_name]
        else:
            raise ValueError(f"Unknown factor: {factor_name}")
        
        logger.info(f"Calculating factor: {factor_name}")
        return func(df, **kwargs)
    
    def calculate_factors(
        self,
        df: pd.DataFrame,
        factors: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Calculate multiple factors.
        
        Args:
            df: Price DataFrame
            factors: List of factor specifications
                     Each dict should have 'name' and optional 'params'
                     
        Returns:
            DataFrame with all factors added
        """
        result = df.copy()
        
        for factor_spec in factors:
            name = factor_spec.get("name")
            params = factor_spec.get("params", {})
            
            if name:
                result = self.calculate_factor(result, name, **params)
        
        return result
    
    def create_composite_factor(
        self,
        df: pd.DataFrame,
        factor_columns: List[str],
        weights: Optional[List[float]] = None,
        output_column: str = "composite_factor",
        date_column: str = "date",
    ) -> pd.DataFrame:
        """
        Create a composite factor from multiple factor columns.
        
        Each factor is z-scored across symbols within each date, so a row's
        composite uses only values from its own date.
        
        Args:
            df: DataFrame with factor columns
            factor_columns: List of column names to combine
            weights: Optional weights for each factor (equal weight if None)
            output_column: Name of the output column
            date_column: Date column to z-score within (a frame without it
                is treated as a single cross-section)
            
        Returns:
            DataFrame with composite factor added
        """
        df = df.copy()
        
        if weights is None:
            weights = [1.0 / len(factor_columns)] * len(factor_columns)
        
        if len(weights) != len(factor_columns):
            raise ValueError("Number of weights must match number of factors")
        
        # Normalize each factor to cross-sectional z-scores per date; a z-score
        # over the whole frame would scale past values with future data
        if date_column in df.columns:
            by_date = df.groupby(date_column)
        else:
            by_date = df.groupby(np.zeros(len(df)))
        normalized = pd.DataFrame(index=df.index)
        for col in factor_columns:
            mean = by_date[col].transform("mean")
            # A date with one symbol has no spread; its z-score is 0
            std = by_date[col].transform("std").fillna(0.0)
            normalized[col] = (df[col] - mean) / (std + 1e-6)
        
        # Calculate weighted average
        df[output_column] = sum(
            normalized[col] * weight 
            for col, weight in zip(factor_columns, weights)
        )
        
        return df
    
    @property
    def available_factors(self) -> List[str]:
        """Get list of all available factors."""
        return list(self.BUILTIN_FACTORS.keys()) + list(self._custom_factors.keys())
