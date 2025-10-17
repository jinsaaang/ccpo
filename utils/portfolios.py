import numpy as np
import pandas as pd
from typing import List, Dict, Optional

class Portfolio:
    """Portfolio class to store weights and calculate returns"""
    
    def __init__(self, name: str):
        self.name = name
        self.weights_history = []  # List of weight vectors over time
        self.dates = []  # Corresponding dates
        self.returns_history = []  # Portfolio returns
        self.solve_times = []  # Solve time for each rebalancing
        
    def add_weights(self, weights: np.ndarray, date, solve_time: float = 0.0):
        """Add portfolio weights for a specific date"""
        self.weights_history.append(weights)
        self.dates.append(date)
        self.solve_times.append(solve_time)
        
    def calculate_returns(self, asset_returns: np.ndarray):
        """
        Calculate portfolio returns given asset returns
        
        Args:
            asset_returns: Array of shape (T, n_assets) - realized returns
        """
        self.returns_history = []
        
        for i, weights in enumerate(self.weights_history):
            if i < len(asset_returns):
                port_return = np.dot(weights, asset_returns[i])
                self.returns_history.append(port_return)
        
        self.returns_history = np.array(self.returns_history)
        
    def get_weights_df(self) -> pd.DataFrame:
        """Get weights as DataFrame"""
        if not self.weights_history:
            return pd.DataFrame()
        
        return pd.DataFrame(
            self.weights_history,
            index=self.dates,
            columns=[f"Asset_{i}" for i in range(len(self.weights_history[0]))]
        )
    
    def get_returns_series(self) -> pd.Series:
        """Get returns as Series"""
        if not self.returns_history:
            return pd.Series()
        
        return pd.Series(self.returns_history, index=self.dates[:len(self.returns_history)])
    
    def get_average_solve_time(self) -> float:
        """Get average solve time"""
        if not self.solve_times:
            return 0.0
        return np.mean(self.solve_times)


class PortfolioCollection:
    """Collection of portfolios for comparison"""
    
    def __init__(self):
        self.portfolios: Dict[str, Portfolio] = {}
        
    def add_portfolio(self, name: str) -> Portfolio:
        """Add a new portfolio to the collection"""
        portfolio = Portfolio(name)
        self.portfolios[name] = portfolio
        return portfolio
    
    def get_portfolio(self, name: str) -> Optional[Portfolio]:
        """Get portfolio by name"""
        return self.portfolios.get(name)
    
    def get_all_names(self) -> List[str]:
        """Get all portfolio names"""
        return list(self.portfolios.keys())
    
    def calculate_all_returns(self, asset_returns: np.ndarray):
        """Calculate returns for all portfolios"""
        for portfolio in self.portfolios.values():
            portfolio.calculate_returns(asset_returns)
