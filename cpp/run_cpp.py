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
                        method: str,
                        omega: float = 0.9,
                        time_limit: float = 300.0) -> Tuple[np.ndarray, float, float, str]:
        """
        Solve portfolio optimization using cpp framework
        
        Problem formulation:
        max_{w,s} s
        s.t. P(w^T r >= s) >= 1 - alpha
             sum(w) = 1
             w >= 0
        
        Reformulated as CCO:
        min_{w,s} -s  (objective)
        s.t. P(s - w^T r <= 0) >= 1 - alpha  (chance constraint)
             sum(w) = 1  (budget constraint)
             w >= 0  (long-only, enforced by solver with lb=0)
        
        Args:
            calibration_returns: K x n_assets array of returns
            method: 'SAA', 'CPP-KKT', 'CPP-MIP'
            omega: SAA parameter (only used for SAA method)
            time_limit: Solver time limit in seconds
            
        Returns:
            weights: Portfolio weights (n_assets,)
            threshold: VaR threshold s
            solve_time: Time taken to solve
            status: 'optimal' or error status string
        """
        # Get number of assets from data shape (safer than passing as argument)
        K, n_assets = calibration_returns.shape
        
        # Convert calibration returns to list of arrays (format expected by solver)
        training_Ys = [calibration_returns[i, :] for i in range(K)]
        
        # Decision variable dimension: n_assets + 1 (weights + threshold s)
        # x[0:n_assets] = weights (with lb=0 for long-only)
        # x[n_assets] = threshold s
        x_dim = n_assets + 1
        
        # Chance constraint: P(s - w^T r <= 0) >= 1 - alpha
        def f(x, Y):
            portfolio_return = sum(x[i] * Y[i] for i in range(n_assets))
            threshold = x[n_assets]
            return threshold - portfolio_return  # Should be <= 0 with probability >= 1-alpha
        
        # Objective: minimize -s (i.e., maximize s = VaR)
        def J(x):
            return -x[n_assets]
        
        # Equality constraint: sum(w) = 1
        def g_budget(x):
            return sum(x[i] for i in range(n_assets)) - 1.0
        
        # No inequality constraints (long-only enforced by lb=0 in solver)
        hs = []
        # hs = [lambda x, i=i: -float(x[i]) for i in range(n_assets)]
        gs = [g_budget]
        
        try:
            # Pass time_limit to solver
            time_start = time.time()
            
            print(f"    [DEBUG] Calling solver: method={method}, n_assets={n_assets}, K={K}, time_limit={time_limit}")
            
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
                robust=False,
                epsilon=None,
                time_limit=time_limit
            )
            
            print(f"    [DEBUG] Solver returned: type={type(solution)}, solve_time={solve_time_internal:.2f}s")
            
            solve_time = time.time() - time_start
            
            # Check solution status
            if isinstance(solution, str):
                # Solver failed or infeasible
                status = solution
                print(f"⚠️  Solver status: {status}")
                
                # Return None to indicate failure (don't silently use equal weights)
                return None, None, solve_time, status
            else:
                # Solution found
                weights = np.array(solution[:n_assets])
                threshold = solution[n_assets]
                
                # Verify constraints (weights should already be >= 0 and sum to ~1 from solver)
                weight_sum = np.sum(weights)
                min_weight = np.min(weights)
                
                if min_weight < -1e-6 or abs(weight_sum - 1.0) > 1e-4:
                    print(f"⚠️  Warning: Solver returned invalid weights!")
                    print(f"    min(w) = {min_weight:.6f}, sum(w) = {weight_sum:.6f}")
                    return None, None, solve_time, "invalid_solution"
                
                return weights, threshold, solve_time, "optimal"
            
        except Exception as e:
            print(f"❌ Error in CPP solver: {e}")
            import traceback
            traceback.print_exc()
            return None, None, 0.0, f"error: {str(e)}"


def run_cpp_backtest(
    data_path: str = "snp50.csv",
    frequency: str = 'weekly',
    train_end_date: str = '2018-12-31',
    val_end_date: str = '2020-12-31',
    test_end_date: str = '2023-12-31',
    alpha: float = 0.1,
    methods: List[str] = ['MIP', 'KKT', 'SAA'],
    time_limit: float = 300.0,
    verbose: bool = True
):
    """
    Run CPP backtest for a single train/val/test split
    
    Args:
        data_path: Path to data file
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        train_end_date: End date for training
        val_end_date: End date for validation
        test_end_date: End date for test
        alpha: Miscoverage rate
        methods: List of methods to run ['MIP', 'KKT', 'SAA']
        time_limit: Solver time limit (seconds)
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
        print(f"Alpha (miscoverage): {alpha:.1%}")
        print(f"Methods: {methods}")
        print(f"Test period: {val_end_date} to {test_end_date}")
        print("="*80)
    
    # Load data
    if verbose:
        print("\nLoading data...")
    
    # Use loader for preprocessing and resampling only
    loader = TimeSeriesDataLoader(data_path=data_path)
    loader.preprocess_data()
    data_resampled = loader.resample_frequency(loader.processed_data, frequency)
    
    # Convert to returns (simple numpy array - no batching needed for CPP)
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
        print(f"Using all historical data for calibration (expanding window)")
    
    # Combine train + val for initial calibration base
    calibration_base = np.vstack([train_returns, val_returns])
    
    for t in range(len(test_returns)):
        if verbose and t % 10 == 0:
            print(f"  Processing period {t+1}/{len(test_returns)}...")
        
        # Use all historical data up to (but not including) current period t
        # t=0: only calibration_base (train + val)
        # t=1: calibration_base + test_returns[:1] (i.e., test_returns[0])
        # t=k: calibration_base + test_returns[:k]
        if t == 0:
            calib_returns = calibration_base
        else:
            calib_returns = np.vstack([
                calibration_base,
                test_returns[:t]  # Up to t-1 (t is not included)
            ])
        
        current_date = test_dates[t]
        
        # Solve for each method
        for method in methods:
            cpp_method = {
                'MIP': 'CPP-MIP',
                'KKT': 'CPP-KKT',
                'SAA': 'SAA'
            }[method]
            
            omega = 0.9 if method == 'SAA' else None
            
            if verbose and t == 0:
                print(f"    Solving {method} with {len(calib_returns)} calibration samples...")
            
            weights, threshold, solve_time, status = optimizer.solve_cpp_method(
                calib_returns,
                method=cpp_method,
                omega=omega,
                time_limit=time_limit
            )
            
            if verbose and t == 0:
                print(f"    {method} completed: status={status}, time={solve_time:.2f}s")
            
            # Handle solver failure
            if status != "optimal":
                if verbose:
                    print(f"⚠️  Period {t+1}, {method}: Solver failed with status '{status}'")
                    print(f"    Skipping this period (using previous weights if available)")
                
                # Skip this period or use previous weights
                # Don't add weights for failed solves to avoid contaminating results
                continue
            
            # Store weights only if solve was successful
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
    start_date: str = '2005-01-01',
    end_date: str = '2024-12-31',
    train_years: int = 6,
    test_years: int = 2,
    alpha: float = 0.1,
    methods: List[str] = ['MIP', 'KKT', 'SAA'],
    time_limit: float = 300.0
):
    """
    Run CPP backtest with rolling windows (non-overlapping test periods)
    
    Args:
        data_path: Path to data file
        frequency: Data frequency ('daily', 'weekly', 'monthly')
        start_date: Start date of available data
        end_date: End date of available data
        train_years: Years of training data (default: 6)
        test_years: Years of each test period (default: 2)
        alpha: Miscoverage rate
        methods: List of methods to run ['MIP', 'KKT', 'SAA']
        time_limit: Solver time limit (seconds)
        
    Returns:
        all_split_results: List of (portfolio_collection, metrics, comparison_df) for each split
        aggregated_metrics: DataFrame with aggregated metrics across splits
    
    Note:
        - Big-M values are set in config/config_basic.py (M=100, m=-100)
        - Uses expanding window: all historical data up to current period
    """
    
    print("CPP Rolling Window Backtest Starts")
    
    # Generate rolling splits with 6:2:2 ratio (train:val:test)
    splits = generate_rolling_splits(
        start_date=start_date,
        end_date=end_date,
        train_years=train_years,
        test_years=test_years
    )
    
    print_rolling_splits(splits)
    # print(stop)
    
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
            train_end_date=split['train_end'],
            val_end_date=split['val_end'],
            test_end_date=split['test_end'],
            alpha=alpha,
            methods=methods,
            time_limit=time_limit,
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
        start_date='2005-01-01',
        end_date='2015-01-01',
        train_years=6,  # 6 years train
        test_years=2,   # 2 years test (val is automatically 2 years = 6:2:2 ratio)
        alpha=0.1,  # 90% coverage
        methods=['KKT'],  # Start with KKT (much faster than MIP)
        time_limit=60.0  # Reduce to 60 seconds for testing
    )
    
    print("\n✅ Rolling window backtest completed!")
    print(f"Total splits evaluated: {len(all_results)}")
    print("\n📌 Note: Using existing cpp framework (solver.py + chance_constraint_encoders.py)")
    print(f"📌 Big-M values: M={100.0}, m={-100.0} (from config/config_basic.py)")
    print(f"📌 Calibration: Expanding window (all historical data)")
    
    # Save aggregated results
    aggregated_metrics.to_csv('./results/cpp_aggregated_results.csv', index=False)
    print("\n💾 Aggregated results saved to '/results/cpp_aggregated_results.csv'")
