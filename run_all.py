"""
Run All Methods: CPP, CCPO, and Equal-Weight
Unified experiment runner that compares all portfolio optimization methods
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import unified config
from configs import config

# Import data loader
from data.data_loader import TimeSeriesDataLoader

# Import portfolio utilities
from utils.portfolios import Portfolio, create_equal_weight_portfolio
from utils.metrics import calculate_portfolio_metrics, compare_methods, print_portfolio_metrics
from utils.visualization import create_all_plots
from utils.evaluate import generate_rolling_splits

# Import method-specific modules
from cpp.solver import solve as cpp_solve
from layers.cp_utils import set_seed


# ============================================================================
# LOGGING UTILITY
# ============================================================================

class CombinedLogger:
    """Logger that writes to both console and file"""
    def __init__(self, log_file='./results/combined_log.txt'):
        self.log_file = log_file
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def flush(self):
        self.terminal.flush()
    
    def log_header(self):
        header = f"\n{'='*80}\n"
        header += f"Combined Experiment Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"{'='*80}\n"
        self.write(header)


# ============================================================================
# CPP RUNNER
# ============================================================================

def run_cpp_method(
    K_returns: np.ndarray,
    L_returns: np.ndarray,
    V_returns: np.ndarray,
    method: str,
    alpha: float
) -> Dict:
    """
    Run a single CPP method for one split
    
    Returns:
        {
            'weights': optimal weights,
            'threshold_pre': pre-calibration threshold,
            'threshold_post': post-calibration threshold,
            'coverage_pre': pre-calibration coverage,
            'coverage_post': post-calibration coverage,
            'solve_time': solver time,
            'status': solver status
        }
    """
    from cpp.chance_constraint_encoders import indicator_encoding, big_M_encoding
    
    K, n_assets = K_returns.shape
    L, _ = L_returns.shape
    V, _ = V_returns.shape
    
    # Convert to training_Ys format
    training_Ys = [K_returns[i, :] for i in range(K)]
    
    # Decision variables: [w_1, ..., w_n, s]
    x_dim = n_assets + 1
    
    # Chance constraint: P(s - w^T r <= 0) >= 1 - alpha
    def f(x, Y):
        s = x[n_assets]
        portfolio_return = sum(x[i] * Y[i] for i in range(n_assets))
        return s - portfolio_return
    
    # Objective: maximize s (minimize -s)
    def J(x):
        return -x[n_assets]
    
    # Variable bounds
    bounds = [(0, 1) for _ in range(n_assets)] + [(config.CPP.m, config.CPP.M)]
    
    # Equality constraint: sum of weights = 1
    def h_eq(x):
        return sum(x[i] for i in range(n_assets)) - 1
    
    # Select encoding based on method
    if method == 'CPP-MIP':
        encoder = indicator_encoding
    else:  # SAA
        encoder = None
    
    try:
        # Solve optimization
        solution, solver_time, status = cpp_solve(
            f=f,
            J=J,
            training_Ys=training_Ys,
            x_dim=x_dim,
            bounds=bounds,
            alpha=alpha,
            h_eq=h_eq,
            omega=config.CPP.OMEGA if method == 'SAA' else None,
            encoder=encoder,
            M_val=config.CPP.M,
            m_val=config.CPP.m,
            zeta=config.CPP.ZETA,
            time_limit=config.CPP.TIME_LIMIT
        )
        
        if isinstance(solution, str):
            return {'status': solution, 'weights': None}
        
        # Extract solution
        weights = np.array(solution[:n_assets])
        threshold_pre = solution[n_assets]
        
        # Calculate pre-calibration coverage on V
        portfolio_returns_V = V_returns @ weights
        coverage_pre = np.mean(portfolio_returns_V >= threshold_pre)
        
        # Step 2: Calibrate threshold using L samples
        calibration_scores = L_returns @ weights
        calibration_scores = np.sort(calibration_scores)
        k = int(np.ceil((L + 1) * alpha))
        p = max(0, min(k - 1, L - 1))
        threshold_post = calibration_scores[p]
        
        # Calculate post-calibration coverage on V
        coverage_post = np.mean(portfolio_returns_V >= threshold_post)
        
        return {
            'weights': weights,
            'threshold_pre': threshold_pre,
            'threshold_post': threshold_post,
            'coverage_pre': coverage_pre,
            'coverage_post': coverage_post,
            'solve_time': solver_time,
            'status': 'optimal'
        }
    
    except Exception as e:
        print(f"  ❌ CPP solver error: {e}")
        return {'status': f'error: {str(e)}', 'weights': None}


# ============================================================================
# CCPO RUNNER
# ============================================================================

def run_ccpo_method(
    data_path: str,
    lookback: int,
    K: int,
    L: int,
    V: int,
    start_idx: int,
    alpha: float
) -> Dict:
    """
    Run CCPO method for one split
    
    IMPORTANT: Scale handling (already done internally):
    1. Time series models are trained on StandardScaler-normalized data
    2. compute_residuals() in cp_utils.py performs inverse_transform automatically
    3. Therefore, mu_pred_V and cov_matrix are already in original scale
    4. Portfolio weights are optimized using original-scale predictions
    5. Realized returns are calculated with original (pct_change) returns
    
    Returns:
        {
            'weights': optimal weights (for original returns),
            'threshold': conformal radius,
            'coverage': empirical coverage,
            'status': optimization status
        }
    """
    from run_ccpo import CCPOPortfolioOptimizer
    
    # Load and prepare data
    loader = TimeSeriesDataLoader(data_path=data_path)
    loader.load_data()
    data_resampled = loader.resample_frequency(loader.raw_data, config.FREQUENCY)
    
    # Create K-L-V sequences
    (X_K, X_L, X_V, y_K, y_L, y_V,
     dates_K, dates_L, dates_V, scaler) = loader.create_sequences_KLV(
        data=data_resampled,
        lookback=lookback,
        K=K,
        L=L,
        V=V,
        start_idx=start_idx,
        forecast_horizon=config.CCPO.FORECAST_HORIZON
    )
    
    # Initialize optimizer
    optimizer = CCPOPortfolioOptimizer(
        alpha=alpha,
        model_cls=config.CCPO.MODEL_CLASS,
        device=config.DEVICE,
        r=config.CCPO.LOW_RANK_R,
        use_local_ellipsoid=config.CCPO.USE_LOCAL_ELLIPSOID
    )
    
    try:
        # Step 1 & 2: Fit and calibrate
        calib_result = optimizer.fit_and_calibrate(
            X_K, X_L, X_V, y_K, y_L, y_V,
            loader=loader,
            scaler=scaler,
            B=config.CCPO.B,
            batch_size=config.CCPO.BATCH_SIZE,
            EPOCHS=config.CCPO.EPOCHS,
            lr=config.CCPO.LEARNING_RATE,
            path=config.CCPO.WEIGHTS_PATH,
            patience=config.CCPO.PATIENCE
        )
        
        if calib_result['status'] != 'optimal':
            return {'status': calib_result['status'], 'weights': None}
        
        # Step 3: Portfolio optimization
        opt_result = optimizer.optimize_portfolio_socp(
            mu_hat=calib_result['mu_pred_V'],
            cov_matrix=calib_result['cov_matrix'],
            radius=calib_result['radius'],
            formulation=config.CCPO.FORMULATION
        )
        
        if opt_result['status'] != 'optimal':
            return {'status': opt_result['status'], 'weights': None}
        
        return {
            'weights': opt_result['weights'],
            'threshold': calib_result['radius'],
            'coverage': calib_result['coverage'],
            'status': 'optimal'
        }
    
    except Exception as e:
        print(f"  ❌ CCPO error: {e}")
        return {'status': f'error: {str(e)}', 'weights': None}


# ============================================================================
# MAIN COMBINED EXPERIMENT
# ============================================================================

def run_all_methods(
    data_path: str = None,
    frequency: str = None,
    K: int = None,
    L: int = None,
    V: int = None,
    lookback: int = None,
    step_size: int = None,
    alpha: float = None,
    use_expanding: bool = None,
    K_max: int = None
):
    """
    Run all portfolio optimization methods and compare results
    """
    # Use config defaults if not provided
    data_path = data_path or config.DATA_PATH
    frequency = frequency or config.FREQUENCY
    K = K or config.K
    L = L or config.L
    V = V or config.V
    lookback = lookback or config.LOOKBACK
    step_size = step_size or config.STEP_SIZE
    alpha = alpha or config.ALPHA
    use_expanding = use_expanding if use_expanding is not None else config.USE_EXPANDING_WINDOW
    K_max = K_max or config.K_MAX
    
    # Create timestamped result folder
    timestamp = datetime.now().strftime('%m%d%H%M')
    result_folder = f'./results/run_all_{timestamp}'
    os.makedirs(result_folder, exist_ok=True)
    
    # Setup logger
    log_file = os.path.join(result_folder, 'combined_log.txt')
    logger = CombinedLogger(log_file)
    logger.log_header()
    
    # Redirect stdout
    original_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print("🚀 Running All Portfolio Optimization Methods")
        print(f"\nConfiguration:")
        print(f"  Data: {data_path}")
        print(f"  Frequency: {frequency}")
        print(f"  K={K}, L={L}, V={V}, Lookback={lookback}")
        print(f"  Step Size: {step_size}")
        print(f"  Alpha: {alpha}")
        print(f"  Expanding Window: {use_expanding}")
        if use_expanding:
            print(f"  K_max: {K_max}")
        print()
        
        # Load data
        print("📊 Loading data...")
        loader = TimeSeriesDataLoader(data_path=data_path)
        loader.load_data()
        data_resampled = loader.resample_frequency(loader.raw_data, frequency)
        
        # Convert to returns
        returns = data_resampled.pct_change().dropna()
        returns_array = returns.values
        dates = returns.index
        asset_names = returns.columns.tolist()
        n_assets = len(asset_names)
        
        print(f"Data loaded: {returns.shape[0]} periods, {n_assets} assets")
        print(f"Date range: {dates.min()} to {dates.max()}")
        print(f"Assets: {asset_names}\n")
        
        # Generate rolling splits
        splits = generate_rolling_splits(
            dates=dates,
            K=K,
            L=L,
            V=V,
            step_size=step_size,
            expanding_window=use_expanding,
            K_max=K_max
        )
        
        print(f"Generated {len(splits)} splits")
        print("="*80 + "\n")
        
        # Define all methods
        cpp_methods = config.CPP.METHODS
        ccpo_methods = ['CCPO-CCO']
        baseline_methods = ['Equal-Weight']
        all_methods = cpp_methods + ccpo_methods + baseline_methods
        
        # Initialize Portfolio objects
        portfolios = {method: Portfolio(name=method) for method in all_methods}
        
        # Store split results for coverage analysis
        all_split_results = {method: [] for method in all_methods}
        
        # Run experiments for each split
        for split_idx, split in enumerate(splits, 1):
            print(f"\n{'='*80}")
            print(f"Split {split_idx}/{len(splits)}")
            print(f"{'='*80}")
            print(f"K: [{split['K_start_idx']:4d}:{split['K_end_idx']:4d}] ({split['K_size']:3d} points)")
            print(f"L: [{split['K_end_idx']:4d}:{split['L_end_idx']:4d}] ({L:3d} points)")
            print(f"V: [{split['L_end_idx']:4d}:{split['V_end_idx']:4d}] ({V:3d} points)")
            print(f"Dates: {split['K_start_date']} to {split['V_end_date']}\n")
            
            # Extract data for this split
            K_returns = returns_array[split['K_start_idx']:split['K_end_idx']]
            L_returns = returns_array[split['K_end_idx']:split['L_end_idx']]
            V_returns = returns_array[split['L_end_idx']:split['V_end_idx']]
            V_dates = dates[split['L_end_idx']:split['V_end_idx']]
            
            # --- Run CPP methods ---
            for cpp_method in cpp_methods:
                print(f"  Running {cpp_method}...")
                result = run_cpp_method(K_returns, L_returns, V_returns, cpp_method, alpha)
                all_split_results[cpp_method].append(result)
                
                if result['status'] == 'optimal':
                    weights = result['weights']
                    threshold = result['threshold_post']
                    
                    # Add to portfolio
                    for t, (date, asset_returns) in enumerate(zip(V_dates, V_returns)):
                        realized_return = weights @ asset_returns
                        portfolios[cpp_method].add_period(
                            date=date,
                            weight=weights,
                            realized_return=realized_return,
                            solve_time=result['solve_time'] if t == 0 else 0.0,
                            threshold_post=threshold if t == 0 else None
                        )
                    print(f"    ✅ Coverage: {result['coverage_post']:.3f}")
                else:
                    print(f"    ❌ Failed: {result['status']}")
            
            # --- Run CCPO method ---
            print(f"  Running CCPO-CCO...")
            result = run_ccpo_method(
                data_path=data_path,
                lookback=lookback,
                K=split['K_size'],
                L=L,
                V=V,
                start_idx=split['K_start_idx'],
                alpha=alpha
            )
            all_split_results['CCPO-CCO'].append(result)
            
            if result['status'] == 'optimal':
                weights = result['weights']
                threshold = result['threshold']
                
                # Calculate realized returns on original (unscaled) returns
                # Note: weights are optimized with inverse-transformed predictions,
                # so they're compatible with original return scale
                for t, (date, asset_returns) in enumerate(zip(V_dates, V_returns)):
                    realized_return = weights @ asset_returns
                    portfolios['CCPO-CCO'].add_period(
                        date=date,
                        weight=weights,
                        realized_return=realized_return,
                        solve_time=0.0,
                        threshold_post=threshold if t == 0 else None
                    )
                print(f"    ✅ Coverage: {result['coverage']:.3f}")
            else:
                print(f"    ❌ Failed: {result['status']}")
            
            # --- Run Equal-Weight baseline ---
            print(f"  Running Equal-Weight...")
            equal_weights = np.ones(n_assets) / n_assets
            for date, asset_returns in zip(V_dates, V_returns):
                realized_return = equal_weights @ asset_returns
                portfolios['Equal-Weight'].add_period(
                    date=date,
                    weight=equal_weights,
                    realized_return=realized_return,
                    solve_time=0.0,
                    threshold_post=None
                )
            print(f"    ✅ Completed")
        
        # ========================================================================
        # RESULTS AGGREGATION
        # ========================================================================
        
        print(f"\n{'='*80}")
        print("📊 AGGREGATED RESULTS")
        print(f"{'='*80}\n")
        
        # Calculate periods per year
        periods_per_year = config.get_periods_per_year(frequency)
        
        # Compare all methods
        performance_df = compare_methods(portfolios, periods_per_year=periods_per_year)
        
        print("📈 Performance Comparison:")
        print(performance_df.to_string())
        print()
        
        # Print detailed metrics for each method
        for method in all_methods:
            if len(portfolios[method]) > 0:
                metrics = calculate_portfolio_metrics(portfolios[method], periods_per_year=periods_per_year)
                print_portfolio_metrics(metrics, portfolio_name=method)
        
        # Save results
        performance_path = os.path.join(result_folder, 'combined_performance.csv')
        performance_df.to_csv(performance_path)
        print(f"\n💾 Performance comparison saved to '{performance_path}'")
        
        # Save weights for each method
        for method in all_methods:
            if len(portfolios[method]) > 0:
                weights_df = pd.DataFrame(
                    portfolios[method].weights,
                    index=portfolios[method].dates,
                    columns=asset_names
                )
                weights_path = os.path.join(result_folder, f'{method}_weights.csv')
                weights_df.to_csv(weights_path)
                print(f"💾 {method} weights saved to '{weights_path}'")
        
        # Create combined visualization
        create_all_plots(portfolios, result_folder, prefix='combined')
        
        print(f"\n📁 All results saved to: {result_folder}")
        print(f"📝 Log saved to: {log_file}")
        
        return {
            'portfolios': portfolios,
            'performance': performance_df,
            'result_folder': result_folder
        }
    
    finally:
        sys.stdout = original_stdout


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    set_seed(config.SEED)
    
    results = run_all_methods()
    
    print("\n✅ All methods completed!")
