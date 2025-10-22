"""
Run CPP methods with proper 2-step calibration procedure
Following the structure from evaluate.py (Step 1 + Step 2)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from datetime import datetime

from data.data_loader import TimeSeriesDataLoader
from utils.portfolios import Portfolio
from utils.metrics import calculate_portfolio_metrics, compare_methods, print_portfolio_metrics
from utils.visualization import create_all_plots
from utils.evaluate import generate_rolling_splits, print_rolling_splits

# Import cpp solver framework
sys.path.append(os.path.join(os.path.dirname(__file__)))
from solver import solve
from configs import config_basic as config

# Logging utility
class Logger:
    """Logger that writes to both console and file"""
    def __init__(self, log_file='./results/cpp_log.txt'):
        self.log_file = log_file
        self.terminal = sys.stdout
        
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    def write(self, message):
        """Write to both terminal and file"""
        self.terminal.write(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def flush(self):
        """Flush both terminal and file"""
        self.terminal.flush()
    
    def log_header(self):
        """Write a header with timestamp"""
        header = f"\n{'='*80}\n"
        header += f"CPP Experiment Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"{'='*80}\n"
        self.write(header)


class CPPPortfolioOptimizer:
    """
    CPP-based portfolio optimizer with 2-step calibration
    Step 1: Get optimal solution using K samples
    Step 2: Calibrate threshold using L samples
    """
    
    def __init__(self, alpha: float = 0.1, n_assets: int = None):
        """
        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
            n_assets: Number of assets (inferred from data if None)
        """
        self.alpha = alpha
        self.n_assets = n_assets
    
    def get_optimal_solution(self,
                            optimization_returns: np.ndarray,
                            validation_returns: np.ndarray,
                            method: str,
                            omega: float = None,
                            time_limit: float = 300.0) -> Dict:
        """
        Step 1: Solve optimization problem using K samples and validate with V samples
        
        Args:
            optimization_returns: K x n_assets array (training data)
            validation_returns: V x n_assets array (test data)
            method: 'SAA', 'CPP-KKT', 'CPP-MIP'
            omega: SAA parameter
            time_limit: Solver time limit
            
        Returns:
            statistics: {
                'weights': optimal weights,
                'threshold_pre': pre-calibration threshold from solver,
                'solve_time': time taken,
                'status': solver status,
                'coverage_pre': coverage on validation set (before calibration),
                'objective_value': objective value
            }
        """
        K, n_assets = optimization_returns.shape
        V, _ = validation_returns.shape
        
        if self.n_assets is None:
            self.n_assets = n_assets
        
        # Convert to training_Ys format
        training_Ys = [optimization_returns[i, :] for i in range(K)]
        
        # Decision variables: [w_1, ..., w_n, s]
        x_dim = n_assets + 1
        
        # Chance constraint: P(s - w^T r <= 0) >= 1 - alpha
        # Equivalent to: P(w^T r >= s) >= 1 - alpha
        def f(x, Y):
            # x is a dict: x[0], ..., x[n-1] are weights, x[n] is threshold
            # Return SCIP expression (not converted to float)
            s = x[n_assets]
            portfolio_return = sum(x[i] * Y[i] for i in range(n_assets))
            return s - portfolio_return
        
        # Objective: maximize s (minimize -s)
        def J(x):
            # Return SCIP expression (not converted to float)
            return -x[n_assets]
        
        # Inequality constraints: w_i >= 0 (long-only)
        def h_nonneg(i):
            return lambda x: -x[i]  # -w_i <= 0 => w_i >= 0
        
        # Budget constraint: sum(w) = 1
        def g_budget(x):
            return sum(x[i] for i in range(n_assets)) - 1.0
        
        # Build constraint lists
        hs = [h_nonneg(i) for i in range(n_assets)]  # Long-only constraints
        gs = [g_budget]
        
        # Solve
        try:
            solution, solver_time = solve(
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
            
            # Check status
            if isinstance(solution, str):
                return {
                    'weights': None,
                    'threshold_pre': None,
                    'solve_time': solver_time,  # Use solver's internal time
                    'status': solution,
                    'coverage_pre': None,
                    'objective_value': None
                }
            
            # Extract solution
            weights = np.array(solution[:n_assets])
            threshold = solution[n_assets]
            
            # Validate constraints (slightly relaxed tolerance for numerical stability)
            weight_sum = np.sum(weights)
            min_weight = np.min(weights)
            
            if min_weight < -1e-7 or abs(weight_sum - 1.0) > 1e-3:
                return {
                    'weights': None,
                    'threshold_pre': None,
                    'solve_time': solver_time,  # Use solver's internal time
                    'status': 'invalid_solution',
                    'coverage_pre': None,
                    'objective_value': None
                }
            
            # Calculate empirical coverage on validation set (V samples)
            # Before calibration: check if f(x, Y) <= 0
            # Vectorized computation
            portfolio_returns_V = validation_returns @ weights  # Shape: (V,)
            feasible_mask = portfolio_returns_V >= threshold
            empirical_coverage = np.mean(feasible_mask)
            
            return {
                'weights': weights,
                'threshold_pre': threshold,  # Pre-calibration threshold from solver
                'solve_time': solver_time,
                'status': 'optimal',
                'coverage_pre': empirical_coverage,  # Pre-calibration coverage
                'objective_value': -threshold
            }
            
        except Exception as e:
            print(f"❌ Error in solver: {e}")
            import traceback
            traceback.print_exc()
            return {
                'weights': None,
                'threshold_pre': None,
                'solve_time': 0.0,  # No time if error
                'status': f'error: {str(e)}',
                'coverage_pre': None,
                'objective_value': None
            }
    
    def calibrate_threshold(self,
                          weights: np.ndarray,
                          calibration_returns: np.ndarray,
                          validation_returns: np.ndarray) -> Dict:
        """
        Step 2: Calibrate threshold using L samples (conformal prediction)
        
        Args:
            weights: Optimal weights from Step 1
            calibration_returns: L x n_assets array
            validation_returns: V x n_assets array (same as Step 1)
            
        Returns:
            calibration_stats: {
                'threshold_post': post-calibration threshold from conformal prediction,
                'coverage_post': coverage with calibrated threshold
            }
        """
        L, n_assets = calibration_returns.shape
        V, _ = validation_returns.shape
        
        # Compute calibration scores (vectorized)
        calibration_scores = calibration_returns @ weights  # Shape: (L,)
        calibration_scores = np.sort(calibration_scores)
        
        # Compute quantile index (conformal prediction formula)
        # Goal: P(w^T Y >= s) >= 1-alpha
        # So s should be the lower alpha-quantile
        k = int(np.ceil((L + 1) * self.alpha))  # 1-based index
        p = max(0, min(k - 1, L - 1))           # Convert to 0-based and clip
        
        # Calibrated threshold (lower alpha-quantile)
        calibrated_threshold = calibration_scores[p]
        
        # Check posterior coverage on validation set (vectorized)
        portfolio_returns_V = validation_returns @ weights  # Shape: (V,)
        feasible_mask = portfolio_returns_V >= calibrated_threshold
        posterior_coverage = np.mean(feasible_mask)
        
        return {
            'threshold_post': calibrated_threshold,  # Post-calibration threshold
            'coverage_post': posterior_coverage       # Post-calibration coverage
        }


def run_cpp_single_experiment(
    returns_data: np.ndarray,
    method: str,
    alpha: float = 0.1,
    omega: float = 0.05,
    time_limit: float = 300.0,
    verbose: bool = False
) -> Dict:
    """
    Run single CPP experiment with 2-step calibration
    
    Args:
        returns_data: Full return data (to be split into K, L, V)
        method: 'CPP-MIP', 'CPP-KKT', 'SAA'
        alpha: Miscoverage rate
        omega: SAA parameter
        time_limit: Solver time limit
        verbose: Print logs
        
    Returns:
        result: {
            'step1': Step 1 statistics,
            'step2': Step 2 statistics
        }
    """
    K = config.K
    L = config.L
    V = config.V
    
    total_samples = K + L + V
    if len(returns_data) < total_samples:
        raise ValueError(f"Not enough samples: need {total_samples}, got {len(returns_data)}")
    
    # Split data
    optimization_returns = returns_data[:K]
    calibration_returns = returns_data[K:K+L]
    validation_returns = returns_data[K+L:K+L+V]
    
    if verbose:
        print(f"  Data split: K={K}, L={L}, V={V}")
    
    # Initialize optimizer
    optimizer = CPPPortfolioOptimizer(alpha=alpha)
    
    # Step 1: Get optimal solution
    if verbose:
        print(f"  Step 1: Optimization with K={K} samples...")
    
    step1_result = optimizer.get_optimal_solution(
        optimization_returns=optimization_returns,
        validation_returns=validation_returns,
        method=method,
        omega=omega,
        time_limit=time_limit
    )
    
    if step1_result['status'] != 'optimal':
        if verbose:
            print(f"  ⚠️ Step 1 failed: {step1_result['status']}")
        return {
            'step1': step1_result,
            'step2': None
        }
    
    if verbose:
        print(f"  Step 1 complete: coverage={step1_result['coverage_pre']:.3f}, time={step1_result['solve_time']:.2f}s")
    
    # Step 2: Calibrate threshold
    if verbose:
        print(f"  Step 2: Calibration with L={L} samples...")
    
    step2_result = optimizer.calibrate_threshold(
        weights=step1_result['weights'],
        calibration_returns=calibration_returns,
        validation_returns=validation_returns
    )
    
    if verbose:
        print(f"  Step 2 complete: coverage={step2_result['coverage_post']:.3f}")
        print(f"  Threshold: {step1_result['threshold_pre']:.6f} → {step2_result['threshold_post']:.6f}")
    
    return {
        'step1': step1_result,
        'step2': step2_result
    }


def run_cpp_methods(
    returns_data: np.ndarray,
    methods: List[str],
    alpha: float = 0.1,
    omega: float = 0.05,
    time_limit: float = 300.0,
    verbose: bool = True
) -> Dict:
    """
    Run CPP for multiple methods on the same data split
    
    Args:
        returns_data: Full return data (K+L+V samples)
        methods: List of methods to test
        alpha: Miscoverage rate
        omega: SAA parameter
        time_limit: Solver time limit
        verbose: Print logs
        
    Returns:
        results: {method: result_dict}
    """
    results = {}
    
    for method in methods:
        if verbose:
            print(f"\nMethod: {method}")
        
        result = run_cpp_single_experiment(
            returns_data=returns_data,
            method=method,
            alpha=alpha,
            omega=omega,
            time_limit=time_limit,
            verbose=verbose
        )
        
        results[method] = result
    
    return results


def summarize_rolling_results(all_split_results: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
    """
    Summarize statistics across rolling splits
    
    Args:
        all_split_results: List of {method: result} dicts, one per split
        
    Returns:
        summary_df: DataFrame with mean ± std for each method across splits
        detailed_stats: {method: {
            'optimal_solutions': [weights1, weights2, ...],  # List across splits
            'solver_times': [time1, time2, ...],
            'thresholds_pre': [t1, t2, ...],
            'thresholds_post': [c1, c2, ...],
            'coverages_pre': [cov1, cov2, ...],
            'coverages_post': [cov1, cov2, ...]
        }}
    """
    # Collect all methods
    all_methods = set()
    for split_result in all_split_results:
        all_methods.update(split_result.keys())
    
    summary_data = []
    detailed_stats = {method: {
        'optimal_solutions': [],
        'solver_times': [],
        'thresholds_pre': [],
        'thresholds_post': [],
        'coverages_pre': [],
        'coverages_post': [],
        'objective_values': [],
        'successful_splits': 0,
        'failed_splits': 0
    } for method in all_methods}
    
    # Collect results across splits
    for split_result in all_split_results:
        for method in all_methods:
            if method not in split_result:
                detailed_stats[method]['failed_splits'] += 1
                continue
                
            result = split_result[method]
            
            if result['step1']['status'] == 'optimal': # need to fix
                detailed_stats[method]['optimal_solutions'].append(result['step1']['weights'])
                detailed_stats[method]['solver_times'].append(result['step1']['solve_time'])
                detailed_stats[method]['thresholds_pre'].append(result['step1']['threshold_pre'])
                detailed_stats[method]['thresholds_post'].append(result['step2']['threshold_post'])
                detailed_stats[method]['coverages_pre'].append(result['step1']['coverage_pre'])
                detailed_stats[method]['coverages_post'].append(result['step2']['coverage_post'])
                detailed_stats[method]['objective_values'].append(result['step1']['objective_value'])
                detailed_stats[method]['successful_splits'] += 1
            else:
                detailed_stats[method]['failed_splits'] += 1
    
    # Compute summary statistics
    for method in all_methods:
        stats = detailed_stats[method]
        n_success = stats['successful_splits']
        
        if n_success == 0:
            continue
        
        summary_data.append({
            'Method': method,
            'N_Splits_Success': n_success,
            'N_Splits_Failed': stats['failed_splits'],
            'Solve_Time_mean': np.mean(stats['solver_times']),
            'Solve_Time_std': np.std(stats['solver_times']),
            'Threshold_Pre_mean': np.mean(stats['thresholds_pre']),
            'Threshold_Post_mean': np.mean(stats['thresholds_post']),
            'Coverage_Pre_mean': np.mean(stats['coverages_pre']),
            'Coverage_Pre_std': np.std(stats['coverages_pre']),
            'Coverage_Post_mean': np.mean(stats['coverages_post']),
            'Coverage_Post_std': np.std(stats['coverages_post']),
            'Objective_mean': np.mean(stats['objective_values'])
        })
    
    return pd.DataFrame(summary_data), detailed_stats


def run_cpp_rolling_backtest(
    data_path: str = "snp50.csv",
    frequency: str = 'weekly',
    K: int = None,
    L: int = None,
    V: int = None,
    step_size: int = None,
    methods: List[str] = ['KKT'],
    alpha: float = 0.05,
    omega: float = 0.03,
    time_limit: float = 300.0,
    log_file: str = './results/cpp_log.txt'
):
    """
    Run CPP with rolling windows based on K, L, V data points
    
    Args:
        data_path: Path to data file
        frequency: Data frequency
        K: Optimization sample size (default: from config)
        L: Calibration sample size (default: from config)
        V: Validation sample size (default: from config)
        step_size: Rolling step size (default: V)
        methods: Methods to test
        alpha: Miscoverage rate
        omega: SAA parameter
        time_limit: Solver time limit
        log_file: Path to log file (appends to existing file)
    """
    # 📌 Create timestamped result folder
    timestamp = datetime.now().strftime('%m%d%H%M')
    result_folder = f'./results/run_{timestamp}'
    os.makedirs(result_folder, exist_ok=True)
    
    # Update log file path to timestamped folder
    log_file = os.path.join(result_folder, 'cpp_log.txt')
    
    # Setup logger
    logger = Logger(log_file)
    logger.log_header()
    
    # Redirect stdout to logger
    original_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print("🎯CPP Rolling Window Backtest with K-L-V Data Point Splits")
        
        # Set defaults from config
        if K is None:
            K = config.K
        if L is None:
            L = config.L
        if V is None:
            V = config.V
        if step_size is None:
            step_size = V
        
        # Load data
        print("\nLoading data...")
        loader = TimeSeriesDataLoader(data_path=data_path)
        loader.load_data()  # Load raw price data (do NOT preprocess with StandardScaler)
        data_resampled = loader.resample_frequency(loader.raw_data, frequency)
        
        # Convert to returns (from raw prices)
        returns = data_resampled.pct_change().dropna()
        returns_array = returns.values
        dates = returns.index
        asset_names = returns.columns.tolist()  # 📌 Get asset names
        
        print(f"Data loaded: {returns.shape[0]} periods, {returns.shape[1]} assets")
        print(f"Date range: {dates.min()} to {dates.max()}")
        print(f"Assets: {asset_names}")
        
        # Generate rolling splits based on K, L, V data points
        splits = generate_rolling_splits(
            dates=dates,
            K=K,
            L=L,
            V=V,
            step_size=step_size
        )
        
        print_rolling_splits(splits, K=K, L=L, V=V)
        
        # 📌 Initialize Portfolio objects for each method
        portfolios = {method: Portfolio(name=method) for method in methods}
        
        # Run experiments for each split
        all_split_results = []
        
        for i, split in enumerate(splits, 1):
            print(f"\n{'='*60}")
            print(f"Split {i}/{len(splits)}")
            print(f"{'='*60}")
            print(f"K (Optimize):  [{split['K_start_idx']:4d}:{split['K_end_idx']:4d}]  {split['K_start_date']} to {split['K_end_date']}")
            print(f"L (Calibrate): [{split['K_end_idx']:4d}:{split['L_end_idx']:4d}]  {split['K_end_date']} to {split['L_end_date']}")
            print(f"V (Validate):  [{split['L_end_idx']:4d}:{split['V_end_idx']:4d}]  {split['L_end_date']} to {split['V_end_date']}")
            
            # Extract data for this split (use entire K+L+V window)
            K_start = split['K_start_idx']
            V_end = split['V_end_idx']
            split_returns = returns_array[K_start:V_end]
            
            # Run all methods on this split
            split_results = run_cpp_methods(
                returns_data=split_returns,
                methods=methods,
                alpha=alpha,
                omega=omega,
                time_limit=time_limit,
                verbose=True
            )
            
            all_split_results.append(split_results)
            
            # 📌 Calculate realized returns for validation period and update Portfolio objects
            validation_returns = returns_array[split['L_end_idx']:split['V_end_idx']]
            validation_dates = dates[split['L_end_idx']:split['V_end_idx']]
            
            for method in methods:
                result = split_results.get(method)
                
                # Only process if optimization was successful
                if result and result['step1']['status'] == 'optimal':
                    weights = result['step1']['weights']
                    threshold_post = result['step2']['threshold_post']
                    solve_time = result['step1']['solve_time']
                    
                    # Calculate realized return for each day in validation period
                    for t, (date, asset_returns) in enumerate(zip(validation_dates, validation_returns)):
                        realized_return = weights @ asset_returns  # w^T * Y_t
                        
                        # Add to portfolio (only record solve_time on first day of this split)
                        portfolios[method].add_period(
                            date=date,
                            weight=weights,
                            realized_return=realized_return,
                            solve_time=solve_time if t == 0 else 0.0,
                            threshold_post=threshold_post if t == 0 else None
                        )
        
        # Summarize across all splits
        print(f"\n{'='*60}")
        print("📊 Summary Across All Splits")
        print(f"{'='*60}")
        
        summary_df, detailed_stats = summarize_rolling_results(all_split_results)
        
        print(f"\n{summary_df.to_string(index=False)}")
        
        # Print detailed info
        for method, stats in detailed_stats.items():
            print(f"\n  {method} Results Across {stats['successful_splits']} Splits:")
            print(f"    Successful: {stats['successful_splits']}, Failed: {stats['failed_splits']}")
            print(f"    Solver times: {[f'{t:.2f}' for t in stats['solver_times']]}")
            print(f"    Thresholds (post): {[f'{t:.4f}' for t in stats['thresholds_post']]}")
            print(f"    Coverages (pre):  {[f'{c:.3f}' for c in stats['coverages_pre']]}")
            print(f"    Coverages (post): {[f'{c:.3f}' for c in stats['coverages_post']]}")
        
        # Save results
        if len(summary_df) > 0:
            summary_path = os.path.join(result_folder, 'cpp_calibrated_results.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\n💾 Calibration results saved to '{summary_path}'")
        else:
            print("\n⚠️ No valid calibration results!")
        
        # 📌 Calculate and display overall portfolio performance
        print(f"\n{'='*80}")
        print("� OVERALL PORTFOLIO PERFORMANCE (Out-of-Sample)")
        print(f"{'='*80}")
        
        # Determine periods_per_year based on frequency
        freq_to_periods = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4,
            'yearly': 1
        }
        periods_per_year = freq_to_periods.get(frequency.lower(), 252)
        
        # Compare all methods
        performance_df = compare_methods(portfolios, periods_per_year=periods_per_year)
        
        print("\n📈 Performance Comparison:")
        print(performance_df.to_string())
        
        # Print detailed metrics for each method
        for method in methods:
            if len(portfolios[method]) > 0:
                metrics = calculate_portfolio_metrics(portfolios[method], periods_per_year=periods_per_year)
                print_portfolio_metrics(metrics, portfolio_name=method)
        
        # 📌 Save portfolio performance and weights
        performance_path = os.path.join(result_folder, 'cpp_portfolio_performance.csv')
        performance_df.to_csv(performance_path)
        print(f"\n💾 Portfolio performance saved to '{performance_path}'")
        
        # 📌 Save weights for each method
        for method in methods:
            if len(portfolios[method]) > 0:
                weights_df = pd.DataFrame(
                    portfolios[method].weights,
                    index=portfolios[method].dates,
                    columns=asset_names
                )
                weights_path = os.path.join(result_folder, f'{method}_weights.csv')
                weights_df.to_csv(weights_path)
                print(f"💾 {method} weights saved to '{weights_path}'")
        
        # 📌 Create visualization plots
        create_all_plots(portfolios, result_folder, prefix='cpp')
        
        print(f"\n📝 Full log saved to '{log_file}'")
        print(f"📁 All results saved to folder: '{result_folder}'")
        
        result = {
            'summary': summary_df,
            'portfolios': portfolios,
            'performance': performance_df,
            'result_folder': result_folder
        }
    
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
    
    return result


if __name__ == "__main__":
    # Run with calibration using K-L-V from config
    summary = run_cpp_rolling_backtest(
        data_path="snp10.csv",  # Use pre-sampled data
        frequency=config.freq,
        K=config.K,           # From config
        L=config.L,           # From config
        V=config.V,           # From config
        step_size=config.V,   # Roll forward by V points
        methods=config.methods,  # 'CPP-MIP', 'CPP-KKT', 'SAA'
        alpha=config.alpha,
        omega=config.omega,   # From config
        time_limit=config.time_limit  
    )
    
    print("\n✅ Calibrated CPP backtest completed!")
    print(f"📌 Using K={config.K}, L={config.L}, V={config.V}")
    # print(f"📌 SAA omega={config.omega}")
