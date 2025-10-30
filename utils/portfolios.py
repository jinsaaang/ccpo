import numpy as np
import pandas as pd
from typing import List, Dict, Optional

class Portfolio:
    """
    Simple and intuitive Portfolio class for tracking performance over time.
    Stores dates, weights, realized returns, solve times, and CPP thresholds.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.dates = []              # List of dates
        self.weights = []            # List of weight vectors (one per rebalancing period)
        self.returns = []            # List of realized returns (one per day)
        self.solve_times = []        # List of solve times (one per rebalancing, 0.0 for other days)
        self.thresholds_post = []    # List of post-calibration thresholds (CPP-specific)
        
    def add_period(self, 
                   date, 
                   weight: np.ndarray, 
                   realized_return: float, 
                   solve_time: float = 0.0, 
                   threshold_post: float = None):
        """
        Add data for one time period
        
        Args:
            date: Date of this period
            weight: Portfolio weights for this period
            realized_return: Realized portfolio return (w^T * Y)
            solve_time: Solver time (0.0 if no rebalancing)
            threshold_post: Post-calibration threshold (CPP-specific)
        """
        self.dates.append(date)
        self.weights.append(weight)
        self.returns.append(realized_return)
        self.solve_times.append(solve_time)
        if threshold_post is not None:
            self.thresholds_post.append(threshold_post)
    
    def get_returns_array(self) -> np.ndarray:
        """Get returns as numpy array"""
        return np.array(self.returns)
    
    def get_weights_array(self) -> np.ndarray:
        """Get weights as numpy array (shape: n_periods x n_assets)"""
        return np.array(self.weights)
    
    def get_returns_series(self) -> pd.Series:
        """Get returns as pandas Series with dates as index"""
        return pd.Series(self.returns, index=self.dates, name=self.name)
    
    def get_weights_df(self) -> pd.DataFrame:
        """Get weights as DataFrame with dates as index"""
        if not self.weights:
            return pd.DataFrame()
        
        n_assets = len(self.weights[0])
        return pd.DataFrame(
            self.weights,
            index=self.dates,
            columns=[f"Asset_{i}" for i in range(n_assets)]
        )
    
    def __len__(self):
        """Return number of periods"""
        return len(self.returns)
    
    def __repr__(self):
        return f"Portfolio(name='{self.name}', n_periods={len(self)})"


class EqualWeightPortfolio:
    """
    Equal Weight (1/N) Portfolio - Simple baseline strategy
    Allocates equal weight to all assets
    """
    
    def __init__(self, n_assets: int, name: str = "Equal-Weight"):
        """
        Args:
            n_assets: Number of assets in the portfolio
            name: Name of the portfolio
        """
        self.n_assets = n_assets
        self.name = name
        self.weights = np.ones(n_assets) / n_assets  # Equal weights
        
    def get_weights(self) -> np.ndarray:
        """Return equal weights"""
        return self.weights
    
    def rebalance(self, *args, **kwargs) -> np.ndarray:
        """
        Rebalancing is trivial for equal weight - always return same weights
        This method exists for compatibility with other portfolio strategies
        """
        return self.weights
    
    def __repr__(self):
        return f"EqualWeightPortfolio(n_assets={self.n_assets}, name='{self.name}')"


def create_equal_weight_portfolio(
    returns_array: np.ndarray,
    dates: pd.DatetimeIndex,
    name: str = "Equal-Weight"
) -> Portfolio:
    """
    Create a Portfolio object with equal weight strategy
    
    Args:
        returns_array: Asset returns (n_periods x n_assets)
        dates: Dates for each period
        name: Portfolio name
        
    Returns:
        Portfolio object with equal weight allocations
    """
    n_periods, n_assets = returns_array.shape
    equal_weights = np.ones(n_assets) / n_assets
    
    portfolio = Portfolio(name=name)
    
    for i, date in enumerate(dates):
        realized_return = equal_weights @ returns_array[i]
        portfolio.add_period(
            date=date,
            weight=equal_weights,
            realized_return=realized_return,
            solve_time=0.0,
            threshold_post=None
        )
    
    return portfolio
