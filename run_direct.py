"""
Run Direct Evaluation (Single Split, No Rolling)
K=520 weeks (~10 years), L=260 weeks (~5 years), V=remaining

This script performs a single train-calibrate-test split without rolling windows.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict
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

# Import method-specific modules
import cpp.solver as cpp_solver
from layers.cp_utils import set_seed


# ============================================================================
# LOGGING UTILITY
# ============================================================================

class DirectLogger:
    """Logger that writes to both console and file"""
    def __init__(self, log_file='./results/direct_log.txt'):
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
        header += f"Direct Evaluation Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"{'='*80}\n"
        self.write(header)


# ============================================================================
# CPP RUNNER
# ============================================================================

def run_cpp_direct(
    K_returns: np.ndarray,
    L_returns: np.ndarray,
    V_returns: np.ndarray,
    method: str,
    alpha: float
) -> Dict:
    """
    Run a single CPP method for direct evaluation
    
    Returns:
        {
            'weights': optimal weights,
            'threshold_post': post-calibration threshold,
            'coverage_post': post-calibration coverage,
            'solve_time': solver time,
            'status': solver status
        }
    """
    K, n_assets = K_returns.shape
    L, _ = L_returns.shape
    V, _ = V_returns.shape
    
    print(f"  Running {method}...")
    print(f"    K={K}, L={L}, V={V}, Assets={n_assets}")
    
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
    
    # Inequality constraints: w_i >= 0 for all i (long-only)
    hs = []
    for i in range(n_assets):
        hs.append(lambda x, i=i: -x[i])  # -w_i <= 0 => w_i >= 0
    
    # Equality constraints: sum(w) = 1
    def g_budget(x):
        return sum(x[i] for i in range(n_assets)) - 1
    gs = [g_budget]
    
    try:
        # Solve optimization using cpp_solver
        solution, solver_time = cpp_solver.solve(
            x_dim=x_dim,
            delta=alpha,
            training_Ys=training_Ys,
            hs=hs,
            gs=gs,
            f=f,
            J=J,
            method=method,
            omega=config.CPP.OMEGA if method == 'SAA' else None,
            time_limit=config.CPP.TIME_LIMIT
        )
        
        if isinstance(solution, str):
            print(f"    ❌ Failed: {solution}")
            return {'status': solution, 'weights': None}
        
        # Extract solution
        weights = np.array(solution[:n_assets])
        threshold_pre = solution[n_assets]
        
        # Step 2: Calibrate threshold using L samples
        calibration_scores = L_returns @ weights
        calibration_scores = np.sort(calibration_scores)
        k = int(np.ceil((L + 1) * alpha))
        p = max(0, min(k - 1, L - 1))
        threshold_post = calibration_scores[p]
        
        # Calculate post-calibration coverage on V
        portfolio_returns_V = V_returns @ weights
        coverage_post = np.mean(portfolio_returns_V >= threshold_post)
        
        print(f"    ✅ Coverage: {coverage_post:.3f}, Threshold: {threshold_post:.6f}")
        
        return {
            'weights': weights,
            'threshold_pre': threshold_pre,
            'threshold_post': threshold_post,
            'coverage_post': coverage_post,
            'solve_time': solver_time,
            'status': 'optimal'
        }
    
    except Exception as e:
        print(f"    ❌ CPP solver error: {e}")
        return {'status': f'error: {str(e)}', 'weights': None}


# ============================================================================
# CCPO RUNNER
# ============================================================================

def run_ccpo_direct(
    data_path: str,
    lookback: int,
    K: int,
    L: int,
    V: int,
    start_idx: int,
    alpha: float,
    V_dates: pd.DatetimeIndex,
    V_returns: np.ndarray
) -> Dict:
    """
    Run CCPO method for direct evaluation with weekly rebalancing
    
    For each test period, predict next period returns and optimize portfolio.
    This enables dynamic rebalancing based on updated predictions.
    
    Returns:
        {
            'portfolios': list of (date, weights, threshold) for each V period,
            'threshold': average conformal radius,
            'coverage': empirical coverage on L,
            'status': optimization status
        }
    """
    from run_ccpo import CCPOPortfolioOptimizer
    import torch
    
    print(f"  Running CCPO-CCO (Weekly Rebalance)...")
    print(f"    K={K}, L={L}, V={V}, Lookback={lookback}")
    
    # Load and prepare data
    loader = TimeSeriesDataLoader(data_path=data_path)
    loader.load_data()
    data_resampled = loader.resample_frequency(loader.raw_data, config.FREQUENCY)
    
    # Create K-L sequences for training and calibration
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
        # Step 1: Train ensemble models on K
        print(f"    Training {config.CCPO.B} bootstrap models...")
        from layers.multi_cp import SPCI_and_EnbPI
        
        # SPCI_and_EnbPI expects Y to have shape (n_samples, forecast_horizon, n_assets)
        # If y is 2D (n_samples, n_assets), expand to 3D
        if len(y_K.shape) == 2:
            y_K = np.expand_dims(y_K, axis=1)  # (n_samples, 1, n_assets)
            y_L = np.expand_dims(y_L, axis=1)
            y_V = np.expand_dims(y_V, axis=1)
        
        # Convert to torch tensors
        X_K_tensor = torch.FloatTensor(X_K)
        X_L_tensor = torch.FloatTensor(X_L)
        X_V_tensor = torch.FloatTensor(X_V)
        y_K_tensor = torch.FloatTensor(y_K)
        y_L_tensor = torch.FloatTensor(y_L)
        y_V_tensor = torch.FloatTensor(y_V)
        
        conformal_predictor = SPCI_and_EnbPI(
            X_K_tensor, X_L_tensor, X_V_tensor,
            y_K_tensor, y_L_tensor, y_V_tensor,
            model_cls=config.CCPO.MODEL_CLASS,
            loader=loader,
            scaler=scaler,
            device=config.DEVICE,
            r=config.CCPO.LOW_RANK_R,
            use_local_ellipsoid=config.CCPO.USE_LOCAL_ELLIPSOID
        )
        
        conformal_predictor.fit_bootstrap_models_online_multistep(
            B=config.CCPO.B,
            batch_size=config.CCPO.BATCH_SIZE,
            EPOCHS=config.CCPO.EPOCHS,
            lr=config.CCPO.LEARNING_RATE,
            path=config.CCPO.WEIGHTS_PATH,
            patience=config.CCPO.PATIENCE,
            valid_mode=True
        )
        
        # Step 2: Calibrate on L to get radius and covariance
        print(f"    Calibrating conformal prediction intervals...")
        conformal_predictor.compute_Widths_Ensemble_online(
            alpha=alpha,
            smallT=False,
            use_SPCI=config.CCPO.USE_SPCI,
            past_window=config.CCPO.PAST_WINDOW,
            random_state=config.SEED
        )
        
        mean_coverage, mean_volume, coverage_seq, volume_seq, radius_seq = conformal_predictor.get_results()
        radius = np.mean(radius_seq)
        cov_matrix = conformal_predictor.global_cov
        
        print(f"    ✅ Calibration done - Coverage: {mean_coverage:.3f}, Radius: {radius:.6f}")
        print(f"       Volume: {mean_volume:.6f}, Coverage std: {np.std(coverage_seq):.6f}")
        
        # Step 3: For each V period, predict and optimize
        print(f"    Optimizing portfolio for each of {len(V_dates)} test periods...")
        portfolios_list = []
        
        # Store per-period coverage and volume (will use radius_seq if needed for adaptive approach)
        # For now, we use global (mean) radius and cov_matrix for all V periods
        # But we keep coverage_seq and volume_seq for potential post-analysis
        
        # Convert price to returns for inference (consistent with training)
        full_returns = data_resampled.pct_change().dropna()
        full_returns_values = full_returns.values
        full_returns_dates = full_returns.index
        
        # Map V_dates to actual returns dates (they might differ due to lookback adjustment)
        # V_dates are prediction dates from create_sequences_KLV
        # V_returns are actual returns at those dates
        
        for v_idx, (v_date, v_return) in enumerate(zip(V_dates, V_returns)):
            # Find the index of current test date in full returns data
            try:
                date_idx = full_returns_dates.get_loc(v_date)
            except KeyError:
                # If exact date not found, find nearest
                date_idx = full_returns_dates.get_indexer([v_date], method='nearest')[0]
            
            # Create input sequence: last lookback periods of RETURNS before v_date
            if date_idx < lookback:
                print(f"      ⚠️  Skipping {v_date}: not enough history (idx={date_idx}, need {lookback})")
                continue
            
            X_test = full_returns_values[date_idx-lookback:date_idx]  # (lookback, n_assets) RETURNS
            X_test_scaled = scaler.transform(X_test)
            X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(0).to(config.DEVICE)  # (1, lookback, n_assets)
            
            # Predict using ensemble (get mean prediction)
            predictions = []
            for b in range(config.CCPO.B):
                model = conformal_predictor.models[b]
                model.eval()
                with torch.no_grad():
                    pred = model(X_test_tensor)  # Shape depends on model output
                    # Ensure pred is (1, n_assets) or (1, 1, n_assets)
                    if len(pred.shape) == 3:
                        pred = pred.squeeze(1)  # (1, n_assets)
                    predictions.append(pred)
            
            # Mean prediction and inverse transform to original scale
            mean_pred = torch.stack(predictions).mean(dim=0).squeeze()  # (n_assets,)
            
            # Handle case where mean_pred might still have extra dimensions
            if len(mean_pred.shape) == 0:  # scalar
                mean_pred = mean_pred.unsqueeze(0)
            
            mean_pred_np = mean_pred.cpu().numpy()
            if mean_pred_np.ndim == 1:
                mean_pred_np = mean_pred_np.reshape(1, -1)
            
            mu_pred = scaler.inverse_transform(mean_pred_np).flatten()  # (n_assets,)
            
            # Optimize portfolio using predicted mu and calibrated uncertainty
            opt_result = optimizer.optimize_portfolio_socp(
                mu_hat=mu_pred,
                cov_matrix=cov_matrix,
                radius=radius,
                gamma=config.CCPO.GAMMA,
                formulation=config.CCPO.FORMULATION
            )
            
            if opt_result['status'] == 'optimal':
                portfolios_list.append({
                    'date': v_date,
                    'weights': opt_result['weights'],
                    'threshold': opt_result['threshold'],  # Use threshold from optimizer
                    'mu_pred': mu_pred
                })
            else:
                print(f"      ⚠️  Optimization failed for {v_date}: {opt_result['status']}")
                # Use equal weights as fallback
                n_assets = len(mu_pred)
                portfolios_list.append({
                    'date': v_date,
                    'weights': np.ones(n_assets) / n_assets,
                    'threshold': None,  # No threshold for fallback
                    'mu_pred': mu_pred
                })
        
        print(f"    ✅ Completed {len(portfolios_list)}/{len(V_dates)} periods")
        
        return {
            'portfolios': portfolios_list,
            'threshold': radius,
            'coverage': mean_coverage,
            'volume': mean_volume,
            'coverage_seq': coverage_seq,
            'volume_seq': volume_seq,
            'radius_seq': radius_seq,
            'status': 'optimal'
        }
    
    except Exception as e:
        import traceback
        print(f"    ❌ CCPO error: {e}")
        print(traceback.format_exc())
        return {'status': f'error: {str(e)}', 'portfolios': []}


# ============================================================================
# MAIN DIRECT EVALUATION
# ============================================================================

def run_direct_evaluation(
    data_path: str = None,
    frequency: str = None,
    K: int = 520,  # ~10 years weekly
    L: int = 260,  # ~5 years weekly
    lookback: int = None,
    alpha: float = None
):
    """
    Run direct evaluation with single K-L-V split (no rolling)
    
    Args:
        data_path: Path to data file
        frequency: Data frequency (should be 'weekly')
        K: Training size (520 weeks = ~10 years)
        L: Calibration size (260 weeks = ~5 years)
        lookback: History window for CCPO
        alpha: Miscoverage rate
    """
    # Use config defaults if not provided
    data_path = data_path or config.DATA_PATH
    frequency = frequency or config.FREQUENCY
    lookback = lookback or config.LOOKBACK
    alpha = alpha or config.ALPHA
    
    # Create timestamped result folder
    timestamp = datetime.now().strftime('%m%d%H%M')
    result_folder = f'./results/run_direct_{timestamp}'
    os.makedirs(result_folder, exist_ok=True)
    
    # Setup logger
    log_file = os.path.join(result_folder, 'direct_log.txt')
    logger = DirectLogger(log_file)
    logger.log_header()
    
    # Redirect stdout
    original_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print("🎯 Direct Evaluation (Single Split)")
        print(f"\nConfiguration:")
        print(f"  Data: {data_path}")
        print(f"  Frequency: {frequency}")
        print(f"  K={K} (~{K/52:.1f} years), L={L} (~{L/52:.1f} years), V=remaining")
        print(f"  Lookback={lookback}, Alpha={alpha}")
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
        n_total = len(returns)
        
        print(f"Data loaded: {n_total} periods, {n_assets} assets")
        print(f"Date range: {dates.min()} to {dates.max()}")
        print(f"Assets: {asset_names}\n")
        
        # Check if we have enough data
        V = n_total - K - L
        if V <= 0:
            raise ValueError(f"Not enough data! Total={n_total}, K={K}, L={L}, V would be {V}")
        
        print(f"Split sizes:")
        print(f"  K (Train):      {K:4d} periods ({dates[0]} to {dates[K-1]})")
        print(f"  L (Calibrate):  {L:4d} periods ({dates[K]} to {dates[K+L-1]})")
        print(f"  V (Test):       {V:4d} periods ({dates[K+L]} to {dates[n_total-1]})")
        print(f"  Total:          {n_total:4d} periods")
        print()
        
        # Split data
        K_returns = returns_array[:K]
        L_returns = returns_array[K:K+L]
        V_returns = returns_array[K+L:]
        V_dates = dates[K+L:]
        
        # Define all methods
        cpp_methods = config.CPP.METHODS
        ccpo_methods = ['CCPO-CCO']
        baseline_methods = ['Equal-Weight']
        all_methods = cpp_methods + ccpo_methods + baseline_methods
        
        # Initialize Portfolio objects
        portfolios = {method: Portfolio(name=method) for method in all_methods}
        
        # Store results
        results = {}
        
        print("="*80)
        print("RUNNING EXPERIMENTS")
        print("="*80 + "\n")
        
        # --- Run CPP methods ---
        for cpp_method in cpp_methods:
            result = run_cpp_direct(K_returns, L_returns, V_returns, cpp_method, alpha)
            results[cpp_method] = result
            
            if result['status'] == 'optimal':
                weights = result['weights']
                threshold = result['threshold_post']
                
                # Add to portfolio for each period in V
                for date, asset_returns in zip(V_dates, V_returns):
                    realized_return = weights @ asset_returns
                    portfolios[cpp_method].add_period(
                        date=date,
                        weight=weights,
                        realized_return=realized_return,
                        solve_time=result['solve_time'],
                        threshold_post=threshold
                    )
        
        # --- Run CCPO method (Weekly Rebalancing) ---
        result = run_ccpo_direct(
            data_path=data_path,
            lookback=lookback,
            K=K,
            L=L,
            V=V,
            start_idx=0,
            alpha=alpha,
            V_dates=V_dates,
            V_returns=V_returns
        )
        results['CCPO-CCO'] = result
        
        if result['status'] == 'optimal' and len(result['portfolios']) > 0:
            # Add each period's portfolio to the tracking object
            for portfolio_info in result['portfolios']:
                date = portfolio_info['date']
                weights = portfolio_info['weights']
                threshold = portfolio_info['threshold']
                
                # Find corresponding returns for this date
                date_idx = V_dates.get_loc(date)
                asset_returns = V_returns[date_idx]
                realized_return = weights @ asset_returns
                
                portfolios['CCPO-CCO'].add_period(
                    date=date,
                    weight=weights,
                    realized_return=realized_return,
                    solve_time=0.0,
                    threshold_post=threshold
                )
            
            print(f"    ✅ CCPO completed with {len(result['portfolios'])} rebalancing periods")
            print(f"       Calibration - Coverage: {result['coverage']:.3f}, Volume: {result['volume']:.6f}")
            print(f"       Coverage std: {np.std(result['coverage_seq']):.6f}, Volume std: {np.std(result['volume_seq']):.6f}")
        
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
        print("📊 RESULTS")
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
        performance_path = os.path.join(result_folder, 'direct_performance.csv')
        performance_df.to_csv(performance_path)
        print(f"\n💾 Performance comparison saved to '{performance_path}'")
        
        # Save weights for each method
        for method in all_methods:
            if len(portfolios[method]) > 0:
                # Save all weight vectors (one per rebalancing period)
                weights_df = pd.DataFrame(
                    portfolios[method].weights,  # All weight vectors
                    columns=asset_names,
                    index=portfolios[method].dates  # Include dates as index
                )
                weights_path = os.path.join(result_folder, f'{method}_weights.csv')
                weights_df.to_csv(weights_path)  # Keep index (dates)
                print(f"💾 {method} weights saved to '{weights_path}' ({len(weights_df)} periods)")
        
        # Save summary with coverage info
        summary_data = []
        for method in all_methods:
            if method in results and results[method]['status'] == 'optimal':
                summary_entry = {
                    'Method': method,
                    'Status': results[method]['status'],
                }
                if 'coverage_post' in results[method]:
                    summary_entry['Coverage'] = results[method]['coverage_post']
                    summary_entry['Threshold'] = results[method]['threshold_post']
                elif 'coverage' in results[method]:
                    summary_entry['Coverage'] = results[method]['coverage']
                    summary_entry['Threshold'] = results[method]['threshold']
                if 'solve_time' in results[method]:
                    summary_entry['Solve_Time'] = results[method]['solve_time']
                # Add volume info if available (CCPO)
                if 'volume' in results[method]:
                    summary_entry['Volume'] = results[method]['volume']
                summary_data.append(summary_entry)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(result_folder, 'direct_summary.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"💾 Summary saved to '{summary_path}'")
        
        # Save detailed CCPO calibration info if available
        if 'CCPO-CCO' in results and 'coverage_seq' in results['CCPO-CCO']:
            ccpo_result = results['CCPO-CCO']
            ccpo_calib_df = pd.DataFrame({
                'Period': range(len(ccpo_result['coverage_seq'])),
                'Coverage': ccpo_result['coverage_seq'],
                'Volume': ccpo_result['volume_seq'],
                'Radius': ccpo_result['radius_seq']
            })
            ccpo_calib_path = os.path.join(result_folder, 'ccpo_calibration_details.csv')
            ccpo_calib_df.to_csv(ccpo_calib_path, index=False)
            print(f"💾 CCPO calibration details saved to '{ccpo_calib_path}'")
        
        # Create combined visualization
        create_all_plots(portfolios, result_folder, prefix='direct')
        
        print(f"\n📁 All results saved to: {result_folder}")
        print(f"📝 Log saved to: {log_file}")
        
        return {
            'portfolios': portfolios,
            'performance': performance_df,
            'summary': summary_df,
            'result_folder': result_folder
        }
    
    finally:
        sys.stdout = original_stdout


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    set_seed(config.SEED)
    
    results = run_direct_evaluation(
        data_path='snp50.csv',
        frequency='weekly',
        K=config.K,  # ~10 years
        L=config.L,  # ~5 years
        lookback=config.LOOKBACK,
        alpha=config.ALPHA
    )
    
    print("\n✅ Direct evaluation completed!")
