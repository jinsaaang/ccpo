import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime, timedelta


def generate_rolling_splits(
    dates: pd.DatetimeIndex,
    K: int,
    L: int,
    V: int,
    step_size: int = None,
    expanding_window: bool = False,
    K_max: int = None
) -> List[Dict]:
    """
    Generate rolling or expanding window splits by data points (K, L, V)
    
    Args:
        dates: DatetimeIndex of the data
        K: Initial number of data points for optimization (train)
        L: Number of data points for calibration (val)
        V: Number of data points for validation (test)
        step_size: How many points to move forward for each split (default: V)
        expanding_window: If True, K grows over time (K_start stays at 0)
        K_max: Maximum size for K in expanding window mode (None = unlimited)
        
    Returns:
        List of dicts with 'K_start_idx', 'K_end_idx', 'L_end_idx', 'V_end_idx',
        'K_start_date', 'K_end_date', 'L_end_date', 'V_end_date', 'K_size'
        
    Example:
        Rolling window (expanding_window=False):
        If K=100, L=50, V=50, step_size=50:
        - Split 1: K=[0:100], L=[100:150], V=[150:200]
        - Split 2: K=[50:150], L=[150:200], V=[200:250]
        
        Expanding window (expanding_window=True):
        If K=100, L=50, V=50, step_size=50:
        - Split 1: K=[0:100], L=[100:150], V=[150:200]
        - Split 2: K=[0:150], L=[150:200], V=[200:250]
        - Split 3: K=[0:200], L=[200:250], V=[250:300]
    """
    if step_size is None:
        step_size = V  # Default: move forward by V points
    
    n_data = len(dates)
    
    # Check initial window size
    total_window = K + L + V
    if n_data < total_window:
        raise ValueError(f"Not enough data. Need {total_window} points, got {n_data}")
    
    splits = []
    
    if expanding_window:
        # Expanding window: K_start stays at 0, K grows
        K_start_idx = 0
        current_end = K  # Start with initial K size
        
        while True:
            # Determine K size (with optional cap)
            K_current = current_end - K_start_idx
            if K_max is not None:
                K_current = min(K_current, K_max)
                K_end_idx = K_start_idx + K_current
            else:
                K_end_idx = current_end
            
            L_end_idx = K_end_idx + L
            V_end_idx = L_end_idx + V
            
            # Check if we have enough data for full split
            if V_end_idx > n_data:
                break
            
            split = {
                # Indices (end is exclusive)
                'K_start_idx': K_start_idx,
                'K_end_idx': K_end_idx,
                'L_end_idx': L_end_idx,
                'V_end_idx': V_end_idx,
                'K_size': K_current,
                # Dates
                'K_start_date': dates[K_start_idx].strftime('%Y-%m-%d'),
                'K_end_date': dates[K_end_idx - 1].strftime('%Y-%m-%d'),
                'L_end_date': dates[L_end_idx - 1].strftime('%Y-%m-%d'),
                'V_end_date': dates[V_end_idx - 1].strftime('%Y-%m-%d'),
            }
            
            splits.append(split)
            
            # Move forward
            current_end += step_size
    
    else:
        # Rolling window: standard sliding window
        current_start = 0
        
        while True:
            K_start_idx = current_start
            K_end_idx = current_start + K
            L_end_idx = K_end_idx + L
            V_end_idx = L_end_idx + V
            
            # Check if we have enough data for full split
            if V_end_idx > n_data:
                break
            
            split = {
                # Indices (end is exclusive)
                'K_start_idx': K_start_idx,
                'K_end_idx': K_end_idx,
                'L_end_idx': L_end_idx,
                'V_end_idx': V_end_idx,
                'K_size': K,
                # Dates
                'K_start_date': dates[K_start_idx].strftime('%Y-%m-%d'),
                'K_end_date': dates[K_end_idx - 1].strftime('%Y-%m-%d'),
                'L_end_date': dates[L_end_idx - 1].strftime('%Y-%m-%d'),
                'V_end_date': dates[V_end_idx - 1].strftime('%Y-%m-%d'),
            }
            
            splits.append(split)
            
            # Move forward
            current_start += step_size
    
    return splits


def print_rolling_splits(splits: List[Dict], K: int, L: int, V: int):
    """Print rolling splits in a readable format"""
    print("\n" + "="*60)
    print(f"Rolling Window Splits (Total: {len(splits)} periods)")
    print(f"Window Size: K={K}, L={L}, V={V} (Total={K+L+V} points)")
    print("="*60)
    
    # for i, split in enumerate(splits, 1):
    #     K_size = split['K_end_idx'] - split['K_start_idx']
    #     L_size = split['L_end_idx'] - split['K_end_idx']
    #     V_size = split['V_end_idx'] - split['L_end_idx']
        
    #     print(f"\n📅 Split {i}:")
    #     print(f"  K (Optimize):  [{split['K_start_idx']:4d}:{split['K_end_idx']:4d}] = {K_size:3d} points  |  {split['K_start_date']} to {split['K_end_date']}")
    #     print(f"  L (Calibrate): [{split['K_end_idx']:4d}:{split['L_end_idx']:4d}] = {L_size:3d} points  |  {pd.to_datetime(split['K_end_date']) + pd.Timedelta(days=1):%Y-%m-%d} to {split['L_end_date']}")
    #     print(f"  V (Validate):  [{split['L_end_idx']:4d}:{split['V_end_idx']:4d}] = {V_size:3d} points  |  {pd.to_datetime(split['L_end_date']) + pd.Timedelta(days=1):%Y-%m-%d} to {split['V_end_date']}")
    # print("="*80 + "\n")


def aggregate_metrics_across_splits(
    all_split_metrics: List[Dict[str, Dict]],
    method_names: List[str]
) -> pd.DataFrame:
    """
    Aggregate metrics across multiple rolling window splits
    
    Args:
        all_split_metrics: List of metric dicts for each split
        method_names: List of method names
        
    Returns:
        DataFrame with mean and std of metrics
    """
    
    # Collect metrics for each method
    method_metrics = {name: [] for name in method_names}
    
    for split_metrics in all_split_metrics:
        for method_name in method_names:
            if method_name in split_metrics:
                method_metrics[method_name].append(split_metrics[method_name])
    
    # Calculate statistics
    results = []
    
    for method_name in method_names:
        metrics_list = method_metrics[method_name]
        
        if not metrics_list:
            continue
        
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(metrics_list)
        
        # Calculate mean and std
        mean_metrics = df.mean()
        std_metrics = df.std()
        
        # Create result dict
        result = {
            'method': method_name,
            'n_splits': len(metrics_list)
        }
        
        for metric_name in df.columns:
            result[f'{metric_name}_mean'] = mean_metrics[metric_name]
            result[f'{metric_name}_std'] = std_metrics[metric_name]
        
        results.append(result)
    
    return pd.DataFrame(results)


def print_aggregated_metrics(agg_df: pd.DataFrame):
    """Print aggregated metrics in a readable format"""
    
    print("\n" + "="*80)
    print("Aggregated Metrics Across All Rolling Windows")
    print("="*80)
    
    for _, row in agg_df.iterrows():
        method = row['method']
        n_splits = int(row['n_splits'])
        
        print(f"\n🎯 {method} (n={n_splits} splits)")
        print("-" * 60)
        
        # Return metrics
        if 'cumulative_return_mean' in row:
            print(f"  Cumulative Return:    {row['cumulative_return_mean']:>10.2%} ± {row['cumulative_return_std']:>8.2%}")
        if 'annualized_return_mean' in row:
            print(f"  Annualized Return:    {row['annualized_return_mean']:>10.2%} ± {row['annualized_return_std']:>8.2%}")
        
        # Risk metrics
        if 'volatility_mean' in row:
            print(f"  Volatility (Ann.):    {row['volatility_mean']:>10.2%} ± {row['volatility_std']:>8.2%}")
        if 'sharpe_ratio_mean' in row:
            print(f"  Sharpe Ratio:         {row['sharpe_ratio_mean']:>10.4f} ± {row['sharpe_ratio_std']:>8.4f}")
        if 'max_drawdown_mean' in row:
            print(f"  Max Drawdown:         {row['max_drawdown_mean']:>10.2%} ± {row['max_drawdown_std']:>8.2%}")
        
        # Coverage metrics
        if 'coverage_rate_mean' in row:
            print(f"  Coverage Rate:        {row['coverage_rate_mean']:>10.2%} ± {row['coverage_rate_std']:>8.2%}")
        if 'violation_rate_mean' in row:
            print(f"  Violation Rate:       {row['violation_rate_mean']:>10.2%} ± {row['violation_rate_std']:>8.2%}")
        
        # Trading metrics
        if 'turnover_mean' in row:
            print(f"  Avg Turnover:         {row['turnover_mean']:>10.2%} ± {row['turnover_std']:>8.2%}")
        
        # Computational metrics
        if 'avg_solve_time_mean' in row:
            print(f"  Avg Solve Time:       {row['avg_solve_time_mean']:>10.4f}s ± {row['avg_solve_time_std']:>8.4f}s")
    
    print("="*80 + "\n")


# if __name__ == "__main__":
#     # Test rolling splits generation
#     import sys
#     sys.path.append('..')
#     from config import config_basic as config
    
#     # Create sample dates (daily data for 2 years)
#     dates = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
    
#     print(f"Total data points: {len(dates)}")
#     print(f"Date range: {dates[0]} to {dates[-1]}")
    
#     splits = generate_rolling_splits(
#         dates=dates,
#         K=config.K,
#         L=config.L,
#         V=config.V,
#         step_size=config.V  # Move forward by V points
#     )
    
#     print_rolling_splits(splits, K=config.K, L=config.L, V=config.V)
