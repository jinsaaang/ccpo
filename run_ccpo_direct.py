"""
Run CCPO Direct Evaluation (Single Split, No Rolling)
K=520 weeks (~10 years), L=260 weeks (~5 years), V=remaining

This script performs CCPO-only evaluation with Equal-Weight baseline.
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
from utils.portfolios import Portfolio
from utils.metrics import calculate_portfolio_metrics, compare_methods, print_portfolio_metrics
from utils.visualization import create_all_plots

# Import CCPO modules
from layers.cp_utils import set_seed


# ============================================================================
# LOGGING UTILITY
# ============================================================================

class CCPODirectLogger:
    """Logger that writes to both console and file"""
    def __init__(self, log_file='./results/ccpo_direct_log.txt'):
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
        header += f"CCPO Direct Evaluation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"{'='*80}\n"
        self.write(header)


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
    Run CCPO with weekly rebalancing for direct evaluation
    
    Returns:
        {
            'portfolios': list of {date, weights, threshold, mu_pred},
            'threshold': calibrated radius,
            'coverage': calibration coverage,
            'volume': calibration volume,
            'coverage_seq': per-period coverage sequence,
            'volume_seq': per-period volume sequence,
            'radius_seq': per-period radius sequence,
            'status': 'optimal' or error message
        }
    """
    from run_ccpo import CCPOPortfolioOptimizer
    import torch
    
    print(f"  Running CCPO-CCO (Weekly Rebalance)...")
    print(f"    K={K}, L={L}, V={V}, Lookback={lookback}")
    print(f"    Gamma (risk preference): {config.CCPO.GAMMA}")
    
    # Load and prepare data
    loader = TimeSeriesDataLoader(data_path=data_path)
    loader.load_data()
    data_resampled = loader.resample_frequency(loader.raw_data, config.FREQUENCY)
    
    # Create K-L sequences for training and calibration
    # Note: We only need K and L sequences for training/calibration
    # V sequences from create_sequences_KLV are NOT used (we use passed V_dates/V_returns instead)
    # So we pass a dummy V value just to satisfy the function signature
    (X_K, X_L, X_V_dummy, y_K, y_L, y_V_dummy,
     dates_K, dates_L, dates_V_dummy, scaler) = loader.create_sequences_KLV(
        data=data_resampled,
        lookback=lookback,
        K=K,
        L=L,
        V=10,  # Dummy value - V sequences are not used, we use passed V_dates/V_returns
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
            y_V_dummy = np.expand_dims(y_V_dummy, axis=1)
        
        # Convert to torch tensors
        X_K_tensor = torch.FloatTensor(X_K)
        X_L_tensor = torch.FloatTensor(X_L)
        X_V_tensor = torch.FloatTensor(X_V_dummy)  # Dummy - not actually used in inference
        y_K_tensor = torch.FloatTensor(y_K)
        y_L_tensor = torch.FloatTensor(y_L)
        y_V_tensor = torch.FloatTensor(y_V_dummy)  # Dummy - not actually used in inference
        
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
# MAIN CCPO DIRECT EVALUATION
# ============================================================================

def run_ccpo_direct_evaluation(
    data_path: str = 'snp10.csv',
    frequency: str = None,
    K: int = 520,
    L: int = 260,
    V: int = None,  # Will use remaining data
    lookback: int = 26,
    alpha: float = None
):
    """
    Run direct evaluation: CCPO vs Equal-Weight baseline
    Single split: K=520, L=260, V=remaining
    """
    
    # Create timestamped result folder
    timestamp = datetime.now().strftime('%m%d%H%M')
    result_folder = f'./results/run_ccpo_direct_{timestamp}'
    os.makedirs(result_folder, exist_ok=True)
    
    # Setup logger
    log_file = os.path.join(result_folder, 'ccpo_direct_log.txt')
    logger = CCPODirectLogger(log_file)
    logger.log_header()
    
    # Redirect stdout to logger
    original_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print("🎯 CCPO Direct Evaluation (Single Split)")
        print(f"Configuration: K={K}, L={L}, V=remaining, Lookback={lookback}")
        print(f"Gamma (risk preference): {config.CCPO.GAMMA}")
        
        # Use config defaults if not specified
        if frequency is None:
            frequency = config.FREQUENCY
        if alpha is None:
            alpha = config.ALPHA
        
        # Load data
        print("\n" + "="*80)
        print("LOADING DATA")
        print("="*80)
        
        loader = TimeSeriesDataLoader(data_path=data_path)
        loader.load_data()
        data_resampled = loader.resample_frequency(loader.raw_data, frequency)
        
        # Convert to returns
        returns = data_resampled.pct_change().dropna()
        returns_array = returns.values
        dates = returns.index
        asset_names = returns.columns.tolist()
        n_assets = len(asset_names)
        
        print(f"\nData Info:")
        print(f"  Total periods: {len(returns)}")
        print(f"  Assets: {n_assets}")
        print(f"  Asset names: {asset_names}")
        print(f"  Date range: {dates.min()} to {dates.max()}")
        print(f"  Frequency: {frequency}")
        
        # Calculate V if not specified
        total_needed = lookback + K + L
        if V is None:
            V = len(returns) - total_needed
            if V <= 0:
                raise ValueError(f"Not enough data. Need at least {total_needed} points, got {len(returns)}")
        
        print(f"\nSplit Configuration:")
        print(f"  K (train): {K} periods")
        print(f"  L (calibration): {L} periods")
        print(f"  V (test): {V} periods")
        print(f"  Lookback: {lookback} periods")
        print(f"  Total needed: {lookback + K + L + V} periods")
        print(f"  Alpha (miscoverage): {alpha}")
        
        # Extract K, L, V splits
        K_start = lookback
        K_end = K_start + K
        L_start = K_end
        L_end = L_start + L
        V_start = L_end
        V_end = V_start + V
        
        K_returns = returns_array[K_start:K_end]
        L_returns = returns_array[L_start:L_end]
        V_returns = returns_array[V_start:V_end]
        
        K_dates = dates[K_start:K_end]
        L_dates = dates[L_start:L_end]
        V_dates = dates[V_start:V_end]
        
        print(f"\nActual Splits:")
        print(f"  K: {K_dates[0]} to {K_dates[-1]} ({len(K_returns)} periods)")
        print(f"  L: {L_dates[0]} to {L_dates[-1]} ({len(L_returns)} periods)")
        print(f"  V: {V_dates[0]} to {V_dates[-1]} ({len(V_returns)} periods)")
        
        # ========================================================================
        # RUN EXPERIMENTS
        # ========================================================================
        
        print("\n" + "="*80)
        print("RUNNING EXPERIMENTS")
        print("="*80 + "\n")
        
        # Initialize portfolios
        methods = ['CCPO-CCO', 'Equal-Weight']
        portfolios = {method: Portfolio(name=method) for method in methods}
        results = {}
        
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
        print(f"\n  Running Equal-Weight...")
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
        for method in methods:
            if len(portfolios[method]) > 0:
                metrics = calculate_portfolio_metrics(portfolios[method], periods_per_year=periods_per_year)
                print_portfolio_metrics(metrics, portfolio_name=method)
        
        # Save results
        performance_path = os.path.join(result_folder, 'ccpo_direct_performance.csv')
        performance_df.to_csv(performance_path)
        print(f"\n💾 Performance comparison saved to '{performance_path}'")
        
        # Save weights for each method
        for method in methods:
            if len(portfolios[method]) > 0:
                weights_df = pd.DataFrame(
                    portfolios[method].weights,
                    columns=asset_names,
                    index=portfolios[method].dates
                )
                weights_path = os.path.join(result_folder, f'{method}_weights.csv')
                weights_df.to_csv(weights_path)
                print(f"💾 {method} weights saved to '{weights_path}' ({len(weights_df)} periods)")
        
        # Save summary with coverage info
        summary_data = []
        for method in methods:
            if method in results and results[method]['status'] == 'optimal':
                summary_entry = {
                    'Method': method,
                    'Status': results[method]['status'],
                }
                if 'coverage' in results[method]:
                    summary_entry['Coverage'] = results[method]['coverage']
                    summary_entry['Threshold'] = results[method]['threshold']
                if 'volume' in results[method]:
                    summary_entry['Volume'] = results[method]['volume']
                summary_data.append(summary_entry)
        
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(result_folder, 'ccpo_direct_summary.csv')
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
        
        # Create visualization plots
        print("\n📊 Creating visualization plots...")
        create_all_plots(portfolios, result_folder, prefix='ccpo_direct')
        print(f"✅ Plots saved to '{result_folder}'")
        
        print(f"\n{'='*80}")
        print("✅ CCPO DIRECT EVALUATION COMPLETED")
        print(f"{'='*80}")
        print(f"📝 Full log: '{log_file}'")
        print(f"📁 Results folder: '{result_folder}'")
        
    finally:
        # Restore original stdout
        sys.stdout = original_stdout
    
    return {
        'portfolios': portfolios,
        'performance': performance_df,
        'results': results,
        'result_folder': result_folder
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    set_seed(config.SEED)
    
    # Run CCPO direct evaluation
    results = run_ccpo_direct_evaluation(
        data_path='snp10.csv',
        frequency='weekly',
        K=460,
        L=300,
        V=None,  # Use remaining data
        lookback=26,
        alpha=0.05
    )
    
    print("\n✅ All done!")
