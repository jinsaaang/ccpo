"""
Run CCPO (Conformal Prediction + Portfolio Optimization) with rolling windows
Following run_cpp.py structure with time series prediction + conformal calibration + SOCP optimization
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from datetime import datetime
import torch
import cvxpy as cp

from data.data_loader import TimeSeriesDataLoader
from utils.portfolios import Portfolio
from utils.metrics import calculate_portfolio_metrics, compare_methods, print_portfolio_metrics
from utils.visualization import create_all_plots
from utils.evaluate import generate_rolling_splits, print_rolling_splits

# Import CCPO-specific modules
from layers.cp_utils import set_seed, train_models, compute_residuals
from layers.multi_cp import SPCI_and_EnbPI
from layers.predictors import MLP, DLinear, LSTMModel
import configs.config_cp as config_cp


# Logging utility (same as run_cpp)
class Logger:
    """Logger that writes to both console and file"""
    def __init__(self, log_file='./results/ccpo_log.txt'):
        self.log_file = log_file
        self.terminal = sys.stdout
        
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
        header += f"CCPO Experiment Run - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"{'='*80}\n"
        self.write(header)


class CCPOPortfolioOptimizer:
    """
    CCPO-based portfolio optimizer with 3 steps:
    Step 1: Time series prediction (Bootstrap ensemble)
    Step 2: Conformal calibration (Ellipsoid construction)
    Step 3: SOCP portfolio optimization
    """
    
    def __init__(self, 
                 alpha: float = 0.1,
                 model_cls=DLinear,
                 device=None,
                 r: int = None,
                 use_local_ellipsoid: bool = False):
        """
        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
            model_cls: Time series model class (DLinear, LSTM, MLP)
            device: Torch device
            r: Low-rank approximation for covariance
            use_local_ellipsoid: Whether to use local covariance
        """
        self.alpha = alpha
        self.model_cls = model_cls
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.r = r
        self.use_local_ellipsoid = use_local_ellipsoid
        
    def fit_and_calibrate(self,
                         X_K: np.ndarray,
                         X_L: np.ndarray,
                         X_V: np.ndarray,
                         y_K: np.ndarray,
                         y_L: np.ndarray,
                         y_V: np.ndarray,
                         loader: TimeSeriesDataLoader,
                         scaler,
                         B: int = 30,
                         batch_size: int = 32,
                         EPOCHS: int = 100,
                         lr: float = 1e-3,
                         path: str = './weights/',
                         patience: int = 10) -> Dict:
        """
        Step 1 & 2: Fit time series models and calibrate with conformal prediction
        
        Args:
            X_K, y_K: Training data (K samples)
            X_L, y_L: Calibration data (L samples)
            X_V, y_V: Validation data (V samples)
            loader: Data loader for preprocessing
            scaler: Fitted scaler
            B: Number of bootstrap models
            
        Returns:
            result: {
                'mu_pred_L': predicted mean on L set,
                'mu_pred_V': predicted mean on V set,
                'cov_matrix': covariance matrix,
                'radius': conformal radius,
                'coverage': empirical coverage on V set,
                'y_pred_V': predictions on V set,
                'y_true_V': true values on V set
            }
        """
        # Convert to torch tensors
        X_K_t = torch.FloatTensor(X_K)
        y_K_t = torch.FloatTensor(y_K).unsqueeze(1) if y_K.ndim == 2 else torch.FloatTensor(y_K).unsqueeze(1).unsqueeze(2)
        X_L_t = torch.FloatTensor(X_L)
        y_L_t = torch.FloatTensor(y_L).unsqueeze(1) if y_L.ndim == 2 else torch.FloatTensor(y_L).unsqueeze(1).unsqueeze(2)
        X_V_t = torch.FloatTensor(X_V)
        y_V_t = torch.FloatTensor(y_V).unsqueeze(1) if y_V.ndim == 2 else torch.FloatTensor(y_V).unsqueeze(1).unsqueeze(2)
        
        # Initialize conformal predictor
        conformal_predictor = SPCI_and_EnbPI(
            X_K_t, X_L_t, X_V_t,
            y_K_t, y_L_t, y_V_t,
            model_cls=self.model_cls,
            loader=loader,
            scaler=scaler,
            device=self.device,
            r=self.r,
            use_local_ellipsoid=self.use_local_ellipsoid,
            bins=config_cp.QRF_BINS,
            n_estimators=config_cp.QRF_N_ESTIMATORS,
            max_d=config_cp.QRF_MAX_DEPTH,
            criterion=config_cp.CRITERION
        )
        
        # Fit bootstrap models
        print("  Fitting bootstrap models...")
        results_fit = conformal_predictor.fit_bootstrap_models_online_multistep(
            B=B,
            batch_size=batch_size,
            EPOCHS=EPOCHS,
            lr=lr,
            path=path,
            patience=patience,
            valid_mode=True
        )
        
        # Compute prediction intervals
        print("  Computing conformal prediction intervals...")
        conformal_predictor.compute_Widths_Ensemble_online(
            alpha=self.alpha,
            smallT=False,
            use_SPCI=config_cp.USE_SPCI,
            past_window=config_cp.PAST_WINDOW,
            random_state=config_cp.SEED
        )
        
        # Get results
        mean_coverage, mean_volume, coverage_seq, volume_seq, radius_seq = conformal_predictor.get_results()
        
        # Extract key information
        # Use mean of predictions as mu_hat
        # NOTE: compute_residuals() already performs inverse_transform in cp_utils.py
        # So valid_pred and test_pred are already in original scale!
        mu_pred_L = conformal_predictor.valid_pred.mean(dim=0).squeeze().detach().cpu().numpy()  # (d,)
        mu_pred_V = conformal_predictor.test_pred.mean(dim=0).squeeze().detach().cpu().numpy()   # (d,)
        
        # Covariance matrix
        # NOTE: global_cov is computed from residuals in original scale (after inverse_transform)
        # See multi_cp.py line 163: np.cov(residuals.T) where residuals = Yv_inv - Pv_inv
        cov_matrix = conformal_predictor.global_cov  # (d, d) - already in original scale!
        
        # Radius (use mean of radius sequence)
        # Radius is in scaled space, but we'll keep it as-is for conformal calibration check
        radius = np.mean(radius_seq)
        
        # Predictions and true values on V (keep scaled for coverage calculation)
        y_pred_V = conformal_predictor.test_pred.mean(dim=0).squeeze().detach().cpu().numpy()
        y_true_V = conformal_predictor.Y_predict.squeeze().numpy()
        
        return {
            'mu_pred_L': mu_pred_L,
            'mu_pred_V': mu_pred_V,
            'cov_matrix': cov_matrix,
            'radius': radius,
            'coverage': mean_coverage,
            'volume': mean_volume,
            'y_pred_V': y_pred_V,
            'y_true_V': y_true_V,
            'status': 'optimal'
        }
    
    def optimize_portfolio_socp(self,
                               mu_hat: np.ndarray,
                               cov_matrix: np.ndarray,
                               radius: float,
                               gamma: float = 1.0,
                               formulation: str = 'cco', # 'cco' or 'target'
                               s0: float = None) -> Dict:
        """
        Step 3: SOCP portfolio optimization
        
        CCO formulation: max s  s.t. gamma * mu^T w - sqrt(q) * ||L^T w||_2 >= s
        Target formulation: max mu^T w  s.t. mu^T w - sqrt(q) * ||L^T w||_2 >= s0
        
        Args:
            mu_hat: Expected return vector (d,)
            cov_matrix: Covariance matrix (d, d)
            radius: Conformal radius (q)
            gamma: Risk preference factor (higher = more risk-averse)
            formulation: 'cco' or 'target'
            s0: Threshold for target formulation
            
        Returns:
            result: {
                'weights': optimal weights,
                'objective_value': objective value,
                'status': solver status
            }
        """
        d = len(mu_hat)
        
        # Cholesky decomposition: Σ = L L^T
        try:
            L = np.linalg.cholesky(cov_matrix)
        except np.linalg.LinAlgError:
            # If not positive definite, use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(cov_matrix)
            eigvals = np.maximum(eigvals, 1e-8)  # Ensure positive
            L = eigvecs @ np.diag(np.sqrt(eigvals))
        
        # CVXPY variables
        w = cp.Variable(d)
        s = cp.Variable()  # Threshold variable
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Budget constraint
            w >= 0           # Long-only
        ]

        if formulation == 'cco':
            # CCO formulation: max s  s.t. mu^T w - radius * ||L^T w||_2 >= s
            # Note: radius from CP already equals sqrt(q), not q itself
            constraints.append(
                gamma * mu_hat @ w - cp.norm(L.T @ w, 2) * radius >= s
            )

            objective = cp.Maximize(s)
        else:
            # Target formulation: max mu^T w  s.t. mu^T w - sqrt(q) * ||L^T w||_2 >= s0
            # Note: radius from CP already equals sqrt(q), not q itself
            if s0 is None:
                raise ValueError("s0 must be provided for target formulation")
            
            constraints.append(
                mu_hat @ w - cp.norm(L.T @ w, 2) * radius >= s0
            )
            objective = cp.Maximize(mu_hat @ w)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                result = {
                    'weights': w.value,
                    'objective_value': problem.value,
                    'status': 'optimal'
                }
                # For CCO formulation, the threshold is the objective value (s)
                if formulation == 'cco':
                    result['threshold'] = problem.value  # This is s* (optimal threshold)
                else:
                    result['threshold'] = s0  # For target formulation, threshold is given
                return result
            else:
                return {
                    'weights': None,
                    'objective_value': None,
                    'threshold': None,
                    'status': problem.status
                }
        except Exception as e:
            print(f"  ❌ SOCP solver error: {e}")
            return {
                'weights': None,
                'objective_value': None,
                'threshold': None,
                'status': f'error: {str(e)}'
            }


def run_ccpo_single_split(
    returns_data: np.ndarray,
    dates: pd.DatetimeIndex,
    data_path: str,
    K: int,
    L: int,
    V: int,
    lookback: int,
    start_idx: int,
    alpha: float = 0.1,
    formulation: str = 'cco',
    s0_list: List[float] = None,
    verbose: bool = True
) -> Dict:
    """
    Run CCPO for a single split
    
    Args:
        returns_data: Full return data
        dates: DatetimeIndex
        data_path: Path to data file (e.g., 'snp10.csv')
        K, L, V: Split sizes
        lookback: History window
        start_idx: Starting index
        alpha: Miscoverage rate
        formulation: 'cco' or 'target'
        s0_list: List of thresholds for target formulation
        verbose: Print logs
        
    Returns:
        result: {method_name: {weights, threshold, coverage, ...}}
    """
    if verbose:
        print(f"\n  Creating K-L-V data split...")
    
    # Load and prepare data
    loader = TimeSeriesDataLoader(data_path=data_path)
    loader.load_data()
    data_resampled = loader.resample_frequency(loader.raw_data, config_cp.FREQUENCY)
    
    # Create K-L-V sequences
    (X_K, X_L, X_V, y_K, y_L, y_V,
     dates_K, dates_L, dates_V, scaler) = loader.create_sequences_KLV(
        data=data_resampled,
        lookback=lookback,
        K=K,
        L=L,
        V=V,
        start_idx=start_idx,
        forecast_horizon=1
    )
    
    if verbose:
        print(f"  Data prepared: K={len(X_K)}, L={len(X_L)}, V={len(X_V)}")
    
    # Initialize optimizer
    optimizer = CCPOPortfolioOptimizer(
        alpha=alpha,
        model_cls=config_cp.MODEL_CLASS,
        device=config_cp.DEVICE,
        r=config_cp.LOW_RANK_R,
        use_local_ellipsoid=config_cp.USE_LOCAL_ELLIPSOID
    )
    
    # Step 1 & 2: Fit and calibrate
    if verbose:
        print(f"  Step 1&2: Time series prediction + Conformal calibration...")
    
    calib_result = optimizer.fit_and_calibrate(
        X_K, X_L, X_V, y_K, y_L, y_V,
        loader=loader,
        scaler=scaler,
        B=config_cp.B,
        batch_size=config_cp.BATCH_SIZE,
        EPOCHS=config_cp.EPOCHS,
        lr=config_cp.LEARNING_RATE,
        path=config_cp.WEIGHTS_PATH,
        patience=config_cp.PATIENCE
    )
    
    if calib_result['status'] != 'optimal':
        if verbose:
            print(f"  ⚠️ Calibration failed: {calib_result['status']}")
        return {}
    
    if verbose:
        print(f"  Coverage: {calib_result['coverage']:.3f}, Radius: {calib_result['radius']:.6f}")
    
    # Step 3: Portfolio optimization
    results = {}
    
    if formulation == 'cco' or s0_list is None:
        # CCO formulation
        if verbose:
            print(f"  Step 3: SOCP optimization (CCO formulation)...")
        
        opt_result = optimizer.optimize_portfolio_socp(
            mu_hat=calib_result['mu_pred_V'],
            cov_matrix=calib_result['cov_matrix'],
            radius=calib_result['radius'],
            gamma=config_cp.GAMMA,
            formulation='cco'
        )
        
        if opt_result['status'] == 'optimal':
            results['CCPO-CCO'] = {
                'weights': opt_result['weights'],
                'threshold': opt_result['threshold'],  # Use threshold from optimizer (s*)
                'coverage': calib_result['coverage'],
                'objective_value': opt_result['objective_value'],
                'status': 'optimal',
                'y_pred': calib_result['y_pred_V'],
                'y_true': calib_result['y_true_V']
            }
    
    if formulation == 'target' and s0_list is not None:
        # Target formulation with multiple s0 values
        for s0 in s0_list:
            if verbose:
                print(f"  Step 3: SOCP optimization (Target formulation, s0={s0:.4f})...")
            
            opt_result = optimizer.optimize_portfolio_socp(
                mu_hat=calib_result['mu_pred_V'],
                cov_matrix=calib_result['cov_matrix'],
                radius=calib_result['radius'],
                gamma=config_cp.GAMMA,
                formulation='target',
                s0=s0
            )
            
            if opt_result['status'] == 'optimal':
                method_name = f'CCPO-Target-s{s0:.3f}'
                results[method_name] = {
                    'weights': opt_result['weights'],
                    'threshold': s0,
                    'coverage': calib_result['coverage'],
                    'objective_value': opt_result['objective_value'],
                    'status': 'optimal',
                    'y_pred': calib_result['y_pred_V'],
                    'y_true': calib_result['y_true_V']
                }
    
    return results


def run_ccpo_rolling_backtest(
    data_path: str = "snp10.csv",
    frequency: str = 'weekly',
    K: int = None,
    L: int = None,
    V: int = None,
    lookback: int = 26,
    step_size: int = None,
    alpha: float = 0.1,
    formulation: str = 'cco',
    s0_list: List[float] = None,
    log_file: str = './results/ccpo_log.txt'
):
    """
    Run CCPO with rolling windows (similar to run_cpp_rolling_backtest)
    
    Args:
        data_path: Path to data file
        frequency: Data frequency
        K: Training sample size
        L: Calibration sample size  
        V: Validation sample size
        lookback: History window size
        step_size: Rolling step size (default: V)
        alpha: Miscoverage rate
        formulation: '1' or '2'
        s0_list: List of thresholds for formulation 2
        log_file: Path to log file
    """
    # Create timestamped result folder
    timestamp = datetime.now().strftime('%m%d%H%M')
    result_folder = f'./results/run_{timestamp}'
    os.makedirs(result_folder, exist_ok=True)
    
    # Update log file path
    log_file = os.path.join(result_folder, 'ccpo_log.txt')
    
    # Setup logger
    logger = Logger(log_file)
    logger.log_header()
    
    # Redirect stdout to logger
    original_stdout = sys.stdout
    sys.stdout = logger
    
    try:
        print("🎯 CCPO Rolling Window Backtest with K-L-V Data Point Splits")
        
        # Set defaults from config
        if K is None:
            K = getattr(config_cp, 'K', 63)
        if L is None:
            L = getattr(config_cp, 'L', 42)
        if V is None:
            V = getattr(config_cp, 'V', 42)
        if step_size is None:
            step_size = V
        
        # Load data
        print("\nLoading data...")
        loader = TimeSeriesDataLoader(data_path=data_path)
        loader.load_data()
        data_resampled = loader.resample_frequency(loader.raw_data, frequency)
        
        # Convert to returns
        returns = data_resampled.pct_change().dropna()
        returns_array = returns.values
        dates = returns.index
        asset_names = returns.columns.tolist()
        
        print(f"Data loaded: {returns.shape[0]} periods, {returns.shape[1]} assets")
        print(f"Date range: {dates.min()} to {dates.max()}")
        print(f"Assets: {asset_names}")
        
        # Generate rolling splits
        total_window = lookback + K + L + V
        n_data = len(returns)
        
        if n_data < total_window:
            raise ValueError(f"Not enough data. Need {total_window} points, got {n_data}")
        
        splits = []
        current_start = 0
        
        while True:
            split_end = current_start + total_window
            if split_end > n_data:
                break
            
            splits.append({
                'start_idx': current_start,
                'end_idx': split_end,
                'start_date': dates[current_start].strftime('%Y-%m-%d'),
                'end_date': dates[split_end - 1].strftime('%Y-%m-%d')
            })
            
            current_start += step_size
        
        print(f"\nRolling Window Splits (Total: {len(splits)} periods)")
        print(f"Window Size: lookback={lookback}, K={K}, L={L}, V={V}")
        print(f"Total window: {total_window} points")
        print("="*60)
        
        # Determine method names
        if formulation == 'cco' or s0_list is None:
            methods = ['CCPO-CCO']
        else:
            methods = [f'CCPO-Target-s{s0:.3f}' for s0 in s0_list]
        
        # Initialize Portfolio objects
        portfolios = {method: Portfolio(name=method) for method in methods}
        
        # Run experiments for each split
        all_split_results = []
        
        for i, split in enumerate(splits, 1):
            print(f"\n{'='*60}")
            print(f"Split {i}/{len(splits)}")
            print(f"{'='*60}")
            print(f"Window: [{split['start_idx']:4d}:{split['end_idx']:4d}]  {split['start_date']} to {split['end_date']}")
            
            # Run CCPO for this split
            split_results = run_ccpo_single_split(
                returns_data=returns_array,
                dates=dates,
                data_path=data_path,
                K=K,
                L=L,
                V=V,
                lookback=lookback,
                start_idx=split['start_idx'],
                alpha=alpha,
                formulation=formulation,
                s0_list=s0_list,
                verbose=True
            )
            
            all_split_results.append(split_results)
            
            # Calculate realized returns for validation period
            V_start_idx = split['start_idx'] + lookback + K + L
            V_end_idx = V_start_idx + V
            validation_returns = returns_array[V_start_idx:V_end_idx]
            validation_dates = dates[V_start_idx:V_end_idx]
            
            for method in methods:
                result = split_results.get(method)
                
                if result and result['status'] == 'optimal':
                    weights = result['weights']
                    threshold = result['threshold']
                    
                    # Calculate realized return for each day in validation period
                    for t, (date, asset_returns) in enumerate(zip(validation_dates, validation_returns)):
                        realized_return = weights @ asset_returns
                        
                        portfolios[method].add_period(
                            date=date,
                            weight=weights,
                            realized_return=realized_return,
                            solve_time=0.0,  # Time series training time not tracked per day
                            threshold_post=threshold if t == 0 else None
                        )
        
        # Summarize results
        print(f"\n{'='*60}")
        print("📊 Summary Across All Splits")
        print(f"{'='*60}")
        
        # Calculate global (aggregate) coverage across all splits
        global_coverage_data = {}
        for method in methods:
            all_y_pred = []
            all_y_true = []
            all_radius = []
            
            for r in all_split_results:
                if method in r and r[method]['status'] == 'optimal':
                    all_y_pred.append(r[method]['y_pred'])
                    all_y_true.append(r[method]['y_true'])
                    all_radius.append(r[method]['threshold'])
            
            if len(all_y_pred) > 0:
                # Concatenate all predictions and true values
                all_y_pred = np.concatenate(all_y_pred, axis=0)
                all_y_true = np.concatenate(all_y_true, axis=0)
                
                # Calculate global coverage
                # Check if predictions are within ellipsoid
                n_assets = all_y_pred.shape[-1] if all_y_pred.ndim > 1 else 1
                n_total = len(all_y_pred)
                
                # Simple coverage: check prediction errors
                errors = all_y_pred - all_y_true
                if errors.ndim > 1:
                    # For multivariate, use mean radius
                    mean_radius = np.mean(all_radius)
                    # Simplified: check if within mean prediction error bounds
                    covered = np.sum(np.abs(errors) <= mean_radius * 2, axis=1) == n_assets
                    global_coverage = np.mean(covered)
                else:
                    mean_radius = np.mean(all_radius)
                    global_coverage = np.mean(np.abs(errors) <= mean_radius)
                
                global_coverage_data[method] = {
                    'n_predictions': n_total,
                    'global_coverage': global_coverage
                }
        
        # Save summary statistics (coverage, optimization status, etc.)
        summary_data = []
        for method in methods:
            n_success = sum(1 for r in all_split_results if method in r and r[method]['status'] == 'optimal')
            n_failed = len(splits) - n_success
            
            if n_success > 0:
                coverages = [r[method]['coverage'] for r in all_split_results if method in r and r[method]['status'] == 'optimal']
                thresholds = [r[method]['threshold'] for r in all_split_results if method in r and r[method]['status'] == 'optimal']
                
                summary_entry = {
                    'Method': method,
                    'N_Splits_Success': n_success,
                    'N_Splits_Failed': n_failed,
                    'Coverage_mean': np.mean(coverages),
                    'Coverage_std': np.std(coverages),
                    'Threshold_mean': np.mean(thresholds),
                    'Threshold_std': np.std(thresholds)
                }
                
                # Add global coverage if available
                if method in global_coverage_data:
                    summary_entry['Global_Coverage'] = global_coverage_data[method]['global_coverage']
                    summary_entry['N_Total_Predictions'] = global_coverage_data[method]['n_predictions']
                
                summary_data.append(summary_entry)
        
        summary_df = pd.DataFrame(summary_data)
        print(f"\n{summary_df.to_string(index=False)}")
        
        # Save summary
        if len(summary_df) > 0:
            summary_path = os.path.join(result_folder, 'ccpo_calibration_results.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\n💾 Calibration results saved to '{summary_path}'")
        
        # Calculate and display overall portfolio performance
        print(f"\n{'='*80}")
        print("📊 OVERALL PORTFOLIO PERFORMANCE (Out-of-Sample)")
        print(f"{'='*80}")
        
        # Determine periods_per_year
        freq_to_periods = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12
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
        
        # Save portfolio performance and weights
        performance_path = os.path.join(result_folder, 'ccpo_portfolio_performance.csv')
        performance_df.to_csv(performance_path)
        print(f"\n💾 Portfolio performance saved to '{performance_path}'")
        
        # Save weights for each method
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
        
        # Create visualization plots
        create_all_plots(portfolios, result_folder, prefix='ccpo')
        
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
    set_seed(config_cp.SEED)
    
    # Run CCPO rolling backtest
    summary = run_ccpo_rolling_backtest(
        data_path="snp10.csv",
        frequency=config_cp.FREQUENCY,
        K=63,  # Or from config
        L=42,
        V=42,
        lookback=config_cp.LOOKBACK,
        step_size=42,
        alpha=config_cp.ALPHA,
        formulation=config_cp.FORMULATION,  # 'cco' or 'target'
        s0_list=None  # [-0.005, 0.000, 0.005] for 'target' formulation
    )
    
    print("\n✅ CCPO backtest completed!")
