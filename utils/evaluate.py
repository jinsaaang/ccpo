import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime, timedelta


def generate_rolling_splits(
    start_date: str,
    end_date: str,
    train_years: int = 6,
    test_years: int = 2
) -> List[Dict[str, str]]:
    """
    Generate rolling window splits by year
    
    Args:
        start_date: Start date of available data (YYYY-MM-DD)
        end_date: End date of available data (YYYY-MM-DD)
        train_years: Training period in years (default: 6)
        test_years: Test period in years (default: 2)
        
    Returns:
        List of dicts with 'train_start', 'train_end', 'val_end', 'test_end'
        Ratio is train:val:test = train_years : test_years : test_years
        
    Example:
        If train_years=6, test_years=2:
        - Split 1: Train 6yr, Val 2yr, Test 2yr (total 10yr)
        - Split 2: Train 6yr, Val 2yr, Test 2yr (shift by 2yr)
        - ...
        - Last split: May overlap if remaining < test_years
    """
    
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    total_years = (end - start).days / 365.25
    
    # Period lengths
    val_years = test_years  # val and test have same length
    total_period_years = train_years + val_years + test_years
    
    splits = []
    current_start = start
    
    while True:
        # Calculate split dates
        train_end = current_start + pd.DateOffset(years=train_years)
        val_end = train_end + pd.DateOffset(years=val_years)
        test_end = val_end + pd.DateOffset(years=test_years)
        
        # Check if we have enough data
        if test_end > end:
            # Not enough data for full split
            remaining_years = (end - current_start).days / 365.25
            
            if remaining_years < total_period_years:
                # Create last split with overlap to maintain ratio
                test_end = end
                val_end = end - pd.DateOffset(years=test_years)
                train_end = val_end - pd.DateOffset(years=val_years)
                
                # Ensure train_start doesn't go before start
                if train_end < current_start + pd.DateOffset(years=train_years):
                    # Need to overlap with previous data
                    train_end = val_end - pd.DateOffset(years=val_years)
                    if train_end - pd.DateOffset(years=train_years) < start:
                        # Not enough data, skip this split
                        break
                
                split = {
                    'train_start': (train_end - pd.DateOffset(years=train_years)).strftime('%Y-%m-%d'),
                    'train_end': train_end.strftime('%Y-%m-%d'),
                    'val_end': val_end.strftime('%Y-%m-%d'),
                    'test_end': test_end.strftime('%Y-%m-%d'),
                    'test_start': val_end.strftime('%Y-%m-%d')
                }
                splits.append(split)
            break
        
        # Normal split (no overlap)
        split = {
            'train_start': current_start.strftime('%Y-%m-%d'),
            'train_end': train_end.strftime('%Y-%m-%d'),
            'val_end': val_end.strftime('%Y-%m-%d'),
            'test_end': test_end.strftime('%Y-%m-%d'),
            'test_start': val_end.strftime('%Y-%m-%d')
        }
        
        splits.append(split)
        
        # Move to next test period (non-overlapping by test_years)
        current_start = current_start + pd.DateOffset(years=test_years)
    
    return splits


def print_rolling_splits(splits: List[Dict[str, str]]):
    """Print rolling splits in a readable format"""
    print("\n" + "="*80)
    print(f"Rolling Window Splits (Total: {len(splits)} periods)")
    print("="*80)
    
    for i, split in enumerate(splits, 1):
        train_start = pd.to_datetime(split['train_start'])
        train_end = pd.to_datetime(split['train_end'])
        val_end = pd.to_datetime(split['val_end'])
        test_start = pd.to_datetime(split['test_start'])
        test_end = pd.to_datetime(split['test_end'])
        
        train_years = (train_end - train_start).days / 365.25
        val_years = (val_end - train_end).days / 365.25
        test_years = (test_end - test_start).days / 365.25
        
        print(f"\n📅 Split {i}:")
        print(f"  Train: {split['train_start']} to {split['train_end']} ({train_years:.1f} years)")
        print(f"  Val:   {train_end.strftime('%Y-%m-%d')} to {split['val_end']} ({val_years:.1f} years)")
        print(f"  Test:  {split['test_start']} to {split['test_end']} ({test_years:.1f} years)")
    
    print("="*80 + "\n")


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


if __name__ == "__main__":
    # Test rolling splits generation
    splits = generate_rolling_splits(
        start_date='2005-01-01',
        end_date='2024-12-31',
        train_years=6,
        test_years=2
    )
    
    print_rolling_splits(splits)
