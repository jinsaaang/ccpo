"""
Run CPP methods (MIP, KKT, SAA) for portfolio optimization
Uses the existing cpp framework (solver.py, chance_constraint_encoders.py)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Tuple, List
import time

from data.data_loader import TimeSeriesDataLoader
from utils.portfolios import Portfolio, PortfolioCollection
from utils.metrics import calculate_all_metrics, print_metrics, compare_portfolios
from utils.evaluate import generate_rolling_splits, print_rolling_splits, aggregate_metrics_across_splits, print_aggregated_metrics

# Import cpp solver framework
sys.path.append(os.path.join(os.path.dirname(__file__)))
from solver import solve


class CPPPortfolioOptimizer:
    """
    CPP-based portfolio optimizer using existing cpp framework
    Wraps solver.py to solve portfolio optimization problems
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
        """
        self.alpha = alpha
        
    def solve_cpp_method(self,
                        calibration_returns: np.ndarray,
                        n_assets: int,
                        method: str,
                        omega: float = 0.9,
                        time_limit: float = 300.0) -> Tuple[np.ndarray, float, float]:
        """
        Solve portfolio optimization using cpp framework
        
        Problem formulation:
        max_{w,s} s
        s.t. P(w^T r >= s) >= 1 - alpha
             sum(w) = 1
             w >= 0
        
        Reformulated as:
        min_{w} -s  (objective)
        s.t. P(s - w^T r <= 0) >= 1 - alpha  (chance constraint)
             sum(w) = 1  (budget)
             w >= 0  (long-only, via variable bounds)
        
        Args:
            calibration_returns: K x n_assets array of returns
            n_assets: Number of assets
            method: 'SAA', 'CPP-KKT', 'CPP-MIP'
            omega: SAA parameter
            time_limit: Time limit in seconds
            
        Returns:
            weights, threshold, solve_time
        """
        K = len(calibration_returns)
        
        # Convert calibration returns to list of arrays (format expected by solver)
        training_Ys = [calibration_returns[i, :] for i in range(K)]
        
        # Decision variable dimension: n_assets + 1 (weights + threshold s)
        # x[0:n_assets] = weights, x[n_assets] = threshold s
        x_dim = n_assets + 1
        
        # Chance constraint: P(s - w^T r <= 0) >= 1 - alpha
        # f(x, Y) = x[n_assets] - sum(x[i] * Y[i] for i in range(n_assets))
        def f(x, Y):
            portfolio_return = sum(x[i] * Y[i] for i in range(n_assets))
            threshold = x[n_assets]
            return threshold - portfolio_return  # Should be <= 0
        
        # Objective: minimize -s (i.e., maximize s)
        def J(x):
            return -x[n_assets]
        
        # Constraint: sum(w) = 1
        def g_budget(x):
            return sum(x[i] for i in range(n_assets)) - 1
        
        # No inequality constraints (bounds handled by variable definition)
        hs = []
        gs = [g_budget]
        
        try:
            # Call cpp solver
            time_start = time.time()
            
            solution, solve_time_internal = solve(
                x_dim=x_dim,
                delta=self.alpha,
                training_Ys=training_Ys,
                hs=hs,
                gs=gs,
                f=f,
                J=J,
                method=method,
                omega=omega if method == "SAA" else None,
                robust=False,  # Can enable if needed
                epsilon=None
            )
            
            solve_time = time.time() - time_start
            
            # Extract weights and threshold
            if isinstance(solution, str):
                # Solver failed
                print(f"Warning: Solver returned status: {solution}")
                weights = np.ones(n_assets) / n_assets
                threshold = 0.0
            else:
                weights = np.array(solution[:n_assets])
                threshold = solution[n_assets]
                
                # Ensure weights are valid
                weights = np.maximum(weights, 0)
                weights = weights / np.sum(weights)
            
            return weights, threshold, solve_time
            
        except Exception as e:
            print(f"Error in CPP solver: {e}")
            # Return equal weights as fallback
            return np.ones(n_assets) / n_assets, 0.0, 0.0


def run_cpp_backtest(
    data_path: str = "snp50.csv",
    frequency: str = 'weekly',
    lookback: int = 52,
    train_end_date: str = '2018-12-31',
    val_end_date: str = '2020-12-31',
    test_end_date: str = '2023-12-31',
    alpha: float = 0.1,
    methods: List[str] = ['MIP', 'KKT', 'SAA'],
    time_limit: float = 300.0,
    big_M: float = 100.0,
    verbose: bool = True
):
    """
    Run CPP backtest for a single train/val/test split
    
    Args:
        data_path: Path to data file
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        lookback: Lookback window for calibration
        train_end_date: End date for training
        val_end_date: End date for validation
        test_end_date: End date for test
        alpha: Miscoverage rate
        methods: List of methods to run ['MIP', 'KKT', 'SAA']
        time_limit: Solver time limit (seconds)
        big_M: Big-M constant for MIP
        verbose: Print detailed logs
        
    Returns:
        portfolio_collection, all_metrics, comparison_df
    """
    
    if verbose:
        print("="*80)
        print("CPP Portfolio Optimization Backtest")
        print("="*80)
        print(f"Data: {data_path}")
        print(f"Frequency: {frequency}")
        print(f"Lookback: {lookback}")
        print(f"Alpha (miscoverage): {alpha:.1%}")
        print(f"Methods: {methods}")
        print(f"Test period: {val_end_date} to {test_end_date}")
        print("="*80)
    
    # Load data
    if verbose:
        print("\nLoading data...")
    loader = TimeSeriesDataLoader(data_path=data_path)
    loader.preprocess_data()
    data_resampled = loader.resample_frequency(loader.processed_data, frequency)
    
    # Convert to returns
    returns = data_resampled.pct_change().dropna()
    dates = returns.index
    returns_array = returns.values
    n_assets = returns_array.shape[1]
    
    if verbose:
        print(f"Data loaded: {returns.shape[0]} periods, {n_assets} assets")
        print(f"Date range: {returns.index.min()} to {returns.index.max()}")
    
    # Split data by dates
    train_mask = dates <= pd.to_datetime(train_end_date)
    val_mask = (dates > pd.to_datetime(train_end_date)) & (dates <= pd.to_datetime(val_end_date))
    test_mask = (dates > pd.to_datetime(val_end_date)) & (dates <= pd.to_datetime(test_end_date))
    
    train_returns = returns_array[train_mask]
    val_returns = returns_array[val_mask]
    test_returns = returns_array[test_mask]
    test_dates = dates[test_mask]
    
    if verbose:
        print(f"\nData split:")
        print(f"  Train: {len(train_returns)} periods")
        print(f"  Val: {len(val_returns)} periods")
        print(f"  Test: {len(test_returns)} periods")
    
    # Initialize portfolios
    portfolio_collection = PortfolioCollection()
    portfolios = {}
    for method in methods:
        portfolios[method] = portfolio_collection.add_portfolio(f"CPP-{method}")
    
    # Initialize optimizer
    optimizer = CPPPortfolioOptimizer(alpha=alpha)
    
    # Rolling window backtest on test set
    if verbose:
        print(f"\nRunning backtest on test set...")
        print(f"Using rolling window of {lookback} periods for calibration")
    
    # Combine train + val for calibration
    calibration_data = np.vstack([train_returns, val_returns])
    
    for t in range(len(test_returns)):
        if verbose and t % 10 == 0:
            print(f"  Processing period {t+1}/{len(test_returns)}...")
        
        # Use last 'lookback' periods for calibration
        if t < lookback:
            # Use end of calibration data + test data so far
            calib_start = len(calibration_data) - (lookback - t)
            calib_returns = np.vstack([
                calibration_data[calib_start:],
                test_returns[:t] if t > 0 else []
            ])
        else:
            # Use only test data
            calib_returns = test_returns[t-lookback:t]
        
        current_date = test_dates[t]
        
        # Solve for each method
        for method in methods:
            cpp_method = {
                'MIP': 'CPP-MIP',
                'KKT': 'CPP-KKT',
                'SAA': 'SAA'
            }[method]
            
            omega = 0.9 if method == 'SAA' else None
            
            weights, threshold, solve_time = optimizer.solve_cpp_method(
                calib_returns, n_assets, 
                method=cpp_method,
                omega=omega,
                time_limit=time_limit
            )
            
            # Store weights
            portfolios[method].add_weights(weights, current_date, solve_time)
    
    # Calculate returns for all portfolios
    if verbose:
        print("\nCalculating portfolio returns...")
    for method in methods:
        portfolios[method].calculate_returns(test_returns)
    
    # Calculate metrics
    if verbose:
        print("\nCalculating metrics...")
    all_metrics = {}
    
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12}[frequency]
    
    for method in methods:
        portfolio = portfolios[method]
        metrics = calculate_all_metrics(
            portfolio_returns=portfolio.returns_history,
            weights_history=portfolio.weights_history,
            threshold=0.0,  # Threshold for violation
            solve_times=portfolio.solve_times,
            periods_per_year=periods_per_year
        )
        all_metrics[f"CPP-{method}"] = metrics
        if verbose:
            print_metrics(metrics, f"CPP-{method}")
    
    # Compare portfolios
    if verbose:
        print("\n" + "="*80)
        print("Portfolio Comparison")
        print("="*80)
    comparison_df = compare_portfolios(all_metrics)
    if verbose:
        print(comparison_df.to_string())
    
    return portfolio_collection, all_metrics, comparison_df


def run_cpp_rolling_backtest(
    data_path: str = "snp50.csv",
    frequency: str = 'weekly',
    lookback: int = 52,
    start_date: str = '2005-01-01',
    end_date: str = '2024-12-31',
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_period_years: float = 2.0,
    min_train_years: float = 6.0,
    alpha: float = 0.1,
    methods: List[str] = ['MIP', 'KKT', 'SAA'],
    time_limit: float = 300.0,
    big_M: float = 100.0
):
    """
    Run CPP backtest with rolling windows (non-overlapping test periods)
    
    Args:
        data_path: Path to data file
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        lookback: Lookback window for calibration
        start_date: Start date of available data
        end_date: End date of available data
        train_ratio: Ratio of training data (default: 0.6)
        val_ratio: Ratio of validation data (default: 0.2)
        test_period_years: Length of each test period in years (default: 2.0)
        min_train_years: Minimum training years for first split (default: 6.0)
        alpha: Miscoverage rate
        methods: List of methods to run ['MIP', 'KKT', 'SAA']
        time_limit: Solver time limit (seconds)
        big_M: Big-M constant for MIP
        
    Returns:
        all_split_results: List of (portfolio_collection, metrics, comparison_df) for each split
        aggregated_metrics: DataFrame with aggregated metrics across splits
    """
    
    print("\n" + "🎯"*40)
    print("CPP Rolling Window Backtest")
    print("🎯"*40)
    
    # Generate rolling splits
    splits = generate_rolling_splits(
        start_date=start_date,
        end_date=end_date,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=1.0 - train_ratio - val_ratio,
        test_period_years=test_period_years,
        min_train_years=min_train_years
    )
    
    print_rolling_splits(splits)
    
    # Run backtest for each split
    all_split_results = []
    all_split_metrics = []
    
    for i, split in enumerate(splits, 1):
        print(f"\n{'='*80}")
        print(f"Running Split {i}/{len(splits)}")
        print(f"{'='*80}")
        
        portfolio_collection, metrics, comparison_df = run_cpp_backtest(
            data_path=data_path,
            frequency=frequency,
            lookback=lookback,
            train_end_date=split['train_end'],
            val_end_date=split['val_end'],
            test_end_date=split['test_end'],
            alpha=alpha,
            methods=methods,
            time_limit=time_limit,
            big_M=big_M,
            verbose=True
        )
        
        all_split_results.append((portfolio_collection, metrics, comparison_df))
        all_split_metrics.append(metrics)
    
    # Aggregate metrics across splits
    print("\n" + "📊"*40)
    print("Aggregating Results Across All Splits")
    print("📊"*40)
    
    method_names = [f"CPP-{m}" for m in methods]
    aggregated_metrics = aggregate_metrics_across_splits(all_split_metrics, method_names)
    
    print_aggregated_metrics(aggregated_metrics)
    
    return all_split_results, aggregated_metrics


if __name__ == "__main__":
    # Run rolling window backtest using existing cpp framework
    all_results, aggregated_metrics = run_cpp_rolling_backtest(
        data_path="snp50.csv",
        frequency='weekly',
        lookback=52,  # 1 year calibration window
        start_date='2005-01-01',
        end_date='2024-12-31',
        train_ratio=0.6,  # 60% train
        val_ratio=0.2,    # 20% val
        test_period_years=2.0,  # 2-year non-overlapping test periods
        min_train_years=6.0,    # Minimum 6 years for first training
        alpha=0.1,  # 90% coverage
        methods=['MIP', 'KKT', 'SAA'],  # Using cpp framework methods
        time_limit=300.0,
        big_M=100.0
    )
    
    print("\n✅ Rolling window backtest completed!")
    print(f"Total splits evaluated: {len(all_results)}")
    print("\n📌 Note: Using existing cpp framework (solver.py + chance_constraint_encoders.py)")
    
    # Save aggregated results
    aggregated_metrics.to_csv('cpp_aggregated_results.csv', index=False)
    print("\n💾 Aggregated results saved to 'cpp_aggregated_results.csv'")
