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
