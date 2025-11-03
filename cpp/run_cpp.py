import sys
import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from datetime import datetime
from data.data_loader_final import TimeSeriesDataLoader
from utils.portfolios import Portfolio
from utils.metrics import calculate_portfolio_metrics, compare_methods, print_portfolio_metrics
from utils.visualization import create_all_plots
from utils.evaluate import generate_rolling_splits, print_rolling_splits
from .solver import solve
from configs import config_revised as config
from utils.evaluation_utils import DirectLogger, aggregate_and_save_results

class CPPPortfolioOptimizer:
    """
    CPP-based portfolio optimizer with 2-step calibration:
    Step 1: Solve optimization using K samples
    Step 2: Calibrate threshold using L samples (conformal prediction)
    """
    def __init__(self, alpha: float = None, n_assets: int = None):
        """
        Args:
            alpha: Miscoverage rate (e.g., 0.1 for 90% coverage)
            n_assets: Number of assets (optional, determined from data if None)
        """
        self.alpha = alpha if alpha is not None else config.ALPHA
        self.n_assets = n_assets

    def get_optimal_solution(self,
                             optimization_returns: np.ndarray,
                             validation_returns: np.ndarray,
                             method: str,
                             omega: float = None,
                             time_limit: float = None) -> Dict:
        """
        Step 1: Solve chance-constrained optimization (K samples) and evaluate on validation (V)
        Returns pre-calibration threshold and performance.
        """
        K, n_assets = optimization_returns.shape
        V = validation_returns.shape[0]
        if self.n_assets is None:
            self.n_assets = n_assets

        # Prepare data for solver
        training_Ys = [optimization_returns[i, :] for i in range(K)]
        x_dim = n_assets + 1

        def f(x, Y):
            s = x[n_assets]
            portfolio_return = sum(x[i] * Y[i] for i in range(n_assets))
            return s - portfolio_return

        def J(x):
            return -x[n_assets]

        # Constraints
        hs = [lambda x, i=i: -x[i] for i in range(n_assets)]
        gs = [lambda x: sum(x[i] for i in range(n_assets)) - 1.0]

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
                time_limit=time_limit if time_limit is not None else config.CPP.TIME_LIMIT
            )
            if isinstance(solution, str):
                return {
                    'weights': None,
                    'threshold_pre': None,
                    'solve_time': solver_time,
                    'status': solution,
                    'coverage_pre': None,
                    'objective_value': None
                }
            # Extract solution
            weights = np.array(solution[:n_assets])
            threshold_pre = solution[n_assets]

            # Validate basic constraints (weights sum ~1 and non-negative)
            weight_sum = weights.sum()
            if weights.min() < -1e-7 or abs(weight_sum - 1.0) > 1e-3:
                return {
                    'weights': None,
                    'threshold_pre': None,
                    'solve_time': solver_time,
                    'status': 'invalid_solution',
                    'coverage_pre': None,
                    'objective_value': None
                }

            # Calculate empirical coverage on validation set (pre-calibration)
            portfolio_returns_V = validation_returns @ weights
            coverage_pre = float(np.mean(portfolio_returns_V >= threshold_pre))
            return {
                'weights': weights,
                'threshold_pre': threshold_pre,
                'solve_time': solver_time,
                'status': 'optimal',
                'coverage_pre': coverage_pre,
                'objective_value': -threshold_pre
            }
        except Exception as e:
            print(f"❌ Error in solver: {e}")
            traceback.print_exc()
            return {
                'weights': None,
                'threshold_pre': None,
                'solve_time': 0.0,
                'status': f'error: {str(e)}',
                'coverage_pre': None,
                'objective_value': None
            }

    def calibrate_threshold(self,
                            weights: np.ndarray,
                            calibration_returns: np.ndarray,
                            validation_returns: np.ndarray) -> Dict:
        """
        Step 2: Calibrate threshold using L samples (conformal prediction formula).
        Returns calibrated threshold and post-calibration coverage.
        """
        L, n_assets = calibration_returns.shape
        # Compute sorted calibration scores
        calibration_scores = np.sort(calibration_returns @ weights)
        k = int(np.ceil((L + 1) * self.alpha))
        p = max(0, min(k - 1, L - 1))
        calibrated_threshold = calibration_scores[p]

        # Compute coverage on validation set with calibrated threshold
        portfolio_returns_V = validation_returns @ weights
        coverage_post = float(np.mean(portfolio_returns_V >= calibrated_threshold))
        return {
            'threshold_post': calibrated_threshold,
            'coverage_post': coverage_post
        }

def run_cpp_single_experiment(
    returns_data: np.ndarray,
    method: str,
    alpha: float = None,
    omega: float = None,
    time_limit: float = None,
    verbose: bool = False
) -> Dict:
    """
    Run a single CPP experiment with 2-step calibration on provided return data.
    Data is split into K/L/V based on config values.
    """
    K = config.TRAIN_LENGTH
    L = config.LEN_K
    V = config.LEN_V
    total_samples = K + L + V
    if returns_data.shape[0] < total_samples:
        raise ValueError(f"Not enough samples: need {total_samples}, got {len(returns_data)}")

    optimization_returns = returns_data[:K]
    calibration_returns = returns_data[K:K+L]
    validation_returns = returns_data[K+L:K+L+V]

    if verbose:
        print(f"  Data split: K={K}, L={L}, V={V}")

    optimizer = CPPPortfolioOptimizer(alpha=alpha)
    if verbose:
        print(f"  Step 1: Optimization with K={K} samples...")
    step1_result = optimizer.get_optimal_solution(
        optimization_returns=optimization_returns,
        validation_returns=validation_returns,
        method=method,
        omega=omega if omega is not None else config.CPP.OMEGA,
        time_limit=time_limit if time_limit is not None else config.CPP.TIME_LIMIT
    )
    if step1_result['status'] != 'optimal':
        if verbose:
            print(f"  ⚠️ Step 1 failed: {step1_result['status']}")
        return {'step1': step1_result, 'step2': None}
    if verbose:
        print(f"  Step 1 complete: coverage={step1_result['coverage_pre']:.3f}, time={step1_result['solve_time']:.2f}s")

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
    return {'step1': step1_result, 'step2': step2_result}

def run_cpp_methods(
    returns_data: np.ndarray,
    methods: List[str],
    alpha: float = None,
    omega: float = None,
    time_limit: float = None,
    verbose: bool = True
) -> Dict:
    """
    Run multiple CPP methods on the same data split (with calibration).
    Returns results dict for each method.
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
    Summarize statistics (coverage, threshold, etc.) across multiple splits.
    Returns a summary DataFrame and detailed stats per method.
    """
    detailed_stats: Dict[str, Dict] = {}
    summary_rows = []
    for split_res in all_split_results:
        for method, res in split_res.items():
            if method not in detailed_stats:
                detailed_stats[method] = {
                    'thresholds_pre': [], 'thresholds_post': [],
                    'coverages_pre': [], 'coverages_post': [], 'solver_times': [],
                    'successful_splits': 0, 'failed_splits': 0
                }
            # 누적 통계 갱신
            if res and res['step1']['status'] == 'optimal':
                detailed_stats[method]['successful_splits'] += 1
                detailed_stats[method]['solver_times'].append(res['step1']['solve_time'])
                detailed_stats[method]['thresholds_pre'].append(res['step1']['threshold_pre'])
                detailed_stats[method]['thresholds_post'].append(res['step2']['threshold_post'] if res['step2'] else None)
                detailed_stats[method]['coverages_pre'].append(res['step1']['coverage_pre'])
                detailed_stats[method]['coverages_post'].append(res['step2']['coverage_post'] if res['step2'] else None)
            else:
                detailed_stats[method]['failed_splits'] += 1

    for method, stats in detailed_stats.items():
        if stats['successful_splits'] > 0:
            summary_rows.append({
                'Method': method,
                'Threshold_Pre_mean': np.mean(stats['thresholds_pre']),
                'Threshold_Post_mean': np.mean([x for x in stats['thresholds_post'] if x is not None]),
                'Coverage_Pre_mean': np.mean(stats['coverages_pre']),
                'Coverage_Pre_std': np.std(stats['coverages_pre']),
                'Coverage_Post_mean': np.mean([c for c in stats['coverages_post'] if c is not None]),
                'Coverage_Post_std': np.std([c for c in stats['coverages_post'] if c is not None])
            })
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, detailed_stats

def run_cpp_rolling_backtest(
    data_path: str = None,
    frequency: str = None,
    K: int = None,
    L: int = None,
    V: int = None,
    step_size: int = None,
    methods: List[str] = None,
    alpha: float = None,
    omega: float = None,
    time_limit: float = None
) -> Dict:
    """
    Run CPP rolling window backtest using K/L/V splits (counts mode).
    """
    # 결과 폴더 및 로거 세팅
    timestamp = datetime.now().strftime('%m%d%H%M')
    result_folder = f'./results/run_{timestamp}'
    os.makedirs(result_folder, exist_ok=True)
    log_file = os.path.join(result_folder, 'cpp_log.txt')
    logger = DirectLogger(log_file)
    logger.log_header(title="CPP Rolling Backtest Run")
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("🎯CPP Rolling Window Backtest with K-L-V Data Point Splits")
        # 필요한 기본값 설정
        K = K if K is not None else config.LEN_K
        L = L if L is not None else 0
        V = V if V is not None else config.LEN_V
        step_size = step_size if step_size is not None else V
        data_path = data_path or os.path.join(config.DATA_PATH, "snp10.csv")
        frequency = frequency or config.FREQUENCY
        methods = methods if methods is not None else config.CPP.METHODS

        # 데이터 로드
        print("\nLoading data...")
        loader = TimeSeriesDataLoader(data_path=data_path if os.path.isabs(data_path) else os.path.join(config.DATA_PATH, data_path))
        loader.load_data()
        data_resampled = loader.resample_frequency(loader.raw_data, frequency)

        # 수익률 산출
        returns = data_resampled.pct_change().dropna()
        dates = returns.index
        asset_names = returns.columns.tolist()
        returns_array = returns.values
        print(f"Data loaded: {returns.shape[0]} periods, {returns.shape[1]} assets")
        print(f"Date range: {dates.min()} to {dates.max()}")

        # Rolling window split 생성 (counts 기반)
        splits = generate_rolling_splits(
            dates=dates,
            K=K,
            L=L,
            V=V,
            step_size=step_size
        )
        print_rolling_splits(splits, K=K, L=L, V=V)

        # Portfolio 객체 초기화
        portfolios = {method: Portfolio(name=method) for method in methods}
        all_split_results = []

        # 각 split에 대해 실험 실행
        for i, split in enumerate(splits, 1):
            print(f"\n{'='*60}")
            print(f"Split {i}/{len(splits)}")
            print(f"{'='*60}")
            print(f"K (Optimize):  [{split['K_start_idx']:4d}:{split['K_end_idx']:4d}]  {split['K_start_date']} to {split['K_end_date']}")
            print(f"L (Calibrate): [{split['K_end_idx']:4d}:{split['L_end_idx']:4d}]  {split['K_end_date']} to {split['L_end_date']}")
            print(f"V (Test):      [{split['L_end_idx']:4d}:{split['V_end_idx']:4d}]  {split['L_end_date']} to {split['V_end_date']}")

            K_start = split['K_start_idx']
            V_end = split['V_end_idx']
            split_returns = returns_array[K_start:V_end]

            split_results = run_cpp_methods(
                returns_data=split_returns,
                methods=methods,
                alpha=alpha if alpha is not None else config.ALPHA,
                omega=omega if omega is not None else config.CPP.OMEGA,
                time_limit=time_limit if time_limit is not None else config.CPP.TIME_LIMIT,
                verbose=True
            )
            all_split_results.append(split_results)

            # V 기간 실제 수익률 추출하여 포트폴리오에 반영
            validation_returns = returns_array[split['L_end_idx']:split['V_end_idx']]
            validation_dates = dates[split['L_end_idx']:split['V_end_idx']]
            for method in methods:
                res = split_results.get(method)
                if res and res['step1']['status'] == 'optimal':
                    weights = res['step1']['weights']
                    threshold_post = res['step2']['threshold_post'] if res['step2'] else None
                    solve_time = res['step1']['solve_time']
                    for t, (date, asset_ret) in enumerate(zip(validation_dates, validation_returns)):
                        portfolios[method].add_period(
                            date=date,
                            weight=weights,
                            realized_return=float(weights @ asset_ret),
                            solve_time=solve_time/len(validation_dates) if t == 0 else 0.0,
                            threshold_post=threshold_post if t == 0 else None
                        )

        # 모든 split 결과 요약
        print(f"\n{'='*60}")
        print("📊 Summary Across All Splits")
        print(f"{'='*60}")
        summary_df, detailed_stats = summarize_rolling_results(all_split_results)
        if not summary_df.empty:
            print(f"\n{summary_df.to_string(index=False)}")
        for method, stats in detailed_stats.items():
            print(f"\n  {method} Results Across {stats['successful_splits']} Splits:")
            print(f"    Successful: {stats['successful_splits']}, Failed: {stats['failed_splits']}")
            print(f"    Solver times: {[f'{t:.2f}' for t in stats['solver_times']]}")
            print(f"    Thresholds (post): {[f'{t:.4f}' for t in stats['thresholds_post'] if t is not None]}")
            print(f"    Coverages (pre):  {[f'{c:.3f}' for c in stats['coverages_pre']]}")
            print(f"    Coverages (post): {[f'{c:.3f}' for c in stats['coverages_post'] if c is not None]}")

        if not summary_df.empty:
            summary_path = os.path.join(result_folder, 'cpp_calibrated_results.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\n💾 Calibration results saved to '{summary_path}'")

        aggregate_and_save_results(
            portfolios=portfolios, result_folder=result_folder,
            asset_names=asset_names, prefix="cpp_portfolio",
            cfg=config, results=None
        )
        print(f"📝 Full log saved to '{log_file}'")
        print(f"📁 All results saved to folder: '{result_folder}'")

        return {
            'summary': summary_df,
            'portfolios': portfolios,
            'performance': None,
            'result_folder': result_folder
        }
    finally:
        sys.stdout = original_stdout
