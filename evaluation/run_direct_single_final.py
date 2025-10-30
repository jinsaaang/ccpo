"""
Run Direct Evaluation (Single Split, No Rolling)
K/V split only (no rolling, no explicit L in opt_res)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')
from configs import config_revised as config
from data.data_loader_final import TimeSeriesDataLoader
from utils.portfolios import Portfolio
from utils.metrics import calculate_portfolio_metrics, compare_methods, print_portfolio_metrics
from utils.visualization import create_all_plots

# Import method-specific modules
import cpp.solver as cpp_solver
from layers.cp_utils import set_seed

from run_ccpo import CCPOPortfolioOptimizer
import torch


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
# Helper
# ============================================================================

def _build_create_all_kwargs(cfg):
    """
    config.MODE에 따라 loader.create_all 인자를 구성하고, 누락 필드를 검증한다.
    - MODE="dates": TRAIN_END_DATES, VALID_END_DATES (TEST_END_DATES는 선택)
    - MODE="counts": TRAIN_LENGTH, LEN_K, LEN_V
    공통 인자: lookback, batch_size, shuffle_train, use_scaler, resample_freq
    """
    base_kwargs = dict(
        lookback=cfg.LOOKBACK,
        batch_size=cfg.BATCH_SIZE,
        shuffle_train=True,
        use_scaler=True,
        resample_freq=cfg.FREQUENCY,
    )

    if cfg.MODE == "dates":
        missing = [k for k in ["TRAIN_END_DATES", "VALID_END_DATES"] if getattr(cfg, k, None) is None]
        if missing:
            raise ValueError(f"[MODE=dates] needs these configurations: {missing}")
        return dict(
            mode="dates",
            train_end_date=cfg.TRAIN_END_DATES,
            val_end_date=cfg.VALID_END_DATES,
            test_end_date=getattr(cfg, "TEST_END_DATES", None),
            **base_kwargs,
        )

    if cfg.MODE == "counts":
        missing = [k for k in ["TRAIN_LENGTH", "LEN_K", "LEN_V"] if getattr(cfg, k, None) is None]
        if missing:
            raise ValueError(f"[MODE=counts] needs these configurations: {missing}")
        return dict(
            mode="counts",
            train_len=cfg.TRAIN_LENGTH,
            K=cfg.LEN_K,
            V=cfg.LEN_V,
            start_idx=0,
            **base_kwargs,
        )

    raise ValueError(f"Unknown MODE: {cfg.MODE}. Use 'dates' or 'counts'.")


# ============================================================================
# CPP RUNNER
# ============================================================================

def run_cpp_direct(
    K_returns: np.ndarray,
    L_returns: Optional[np.ndarray],
    V_returns: np.ndarray,
    method: str,
    alpha: float
) -> Dict:
    """
    Run a single CPP method for direct evaluation (K/V; optional L for calibration)
    Returns:
        {
            'weights': optimal weights,
            'threshold_post': post-calibration threshold,
            'coverage_post': post-calibration coverage (on V),
            'solve_time': solver time,
            'status': solver status
        }
    """
    K, n_assets = K_returns.shape
    L = 0 if L_returns is None else L_returns.shape[0]
    V = V_returns.shape[0]
    
    print(f"  Running {method}...")
    print(f"    K={K}, L={L}, V={V}, Assets={n_assets}")
    
    # Convert to training_Ys format
    training_Ys = [K_returns[i, :] for i in range(K)]
    
    # Decision variables: [w_1, ..., w_n, s]
    x_dim = n_assets + 1
    
    # Chance constraint
    def f(x, Y):
        s = x[n_assets]
        portfolio_return = sum(x[i] * Y[i] for i in range(n_assets))
        return s - portfolio_return
    
    # Objective
    def J(x):
        return -x[n_assets]
    
    # Inequality constraints
    hs = []
    for i in range(n_assets):
        hs.append(lambda x, i=i: -x[i])  # -w_i <= 0 => w_i >= 0
    
    # Equality constraints
    def g_budget(x):
        return sum(x[i] for i in range(n_assets)) - 1
    gs = [g_budget]
    
    try:
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
        
        if L_returns is not None and L > 0: 
            calibration_scores = L_returns @ weights
            calibration_scores = np.sort(calibration_scores)
            k = int(np.ceil((L + 1) * alpha))
            p = max(0, min(k - 1, L - 1))
            threshold_post = calibration_scores[p]
        else:
            threshold_post = threshold_pre
        
        portfolio_returns_V = V_returns @ weights
        coverage_post = float(np.mean(portfolio_returns_V >= threshold_post))
        
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
    """

    print(f"  Running CCPO-CCO (Weekly Rebalance)...")
    print(f"    K={K}, L={L}, V={V}, Lookback={lookback}")
    
    # Load and prepare data
    loader = TimeSeriesDataLoader(base_path="./data", num_assets=config.NUM_ASSETS)
    loader.load_data()
    
    if config.MODE == 'dates':    
        res = loader.create_all(
            mode="dates",
            lookback=config.LOOKBACK,
            train_end_date=config.TRAIN_END_DATES,
            val_end_date=config.VALID_END_DATES,
            test_end_date=getattr(config, "TEST_END_DATES", None),
            batch_size=config.BATCH_SIZE,
            shuffle_train=True,
            use_scaler=True,
            resample_freq=config.FREQUENCY
        )
        
    elif config.MODE == 'counts':
        res = loader.create_all(
            mode="counts",
            lookback=config.LOOKBACK,
            train_len=config.TRAIN_LENGTH,
            K=config.LEN_K,
            V=config.LEN_V,
            start_idx=0,
            batch_size=config.BATCH_SIZE,
            shuffle_train=True,
            use_scaler=True,
            resample_freq=config.FREQUENCY
        )
    else:
        raise ValueError(f"Unsupported MODE: {config.MODE}")
    
    
    train_loader = res['model']['train_loader']
    valid_loader = res['model']['valid_loader']
    test_loader  = res['model']['test_loader']
    scaler = res['scaler']
     
    # Initialize optimizer
    optimizer = CCPOPortfolioOptimizer(
        alpha=alpha,
        model_cls=config.CCPO.MODEL_CLASS,
        device=config.DEVICE,
        r=config.CCPO.LOW_RANK_R,
        use_local_ellipsoid=config.CCPO.USE_LOCAL_ELLIPSOID,
        bins = config.CCPO.QRF_BINS,
        n_estimators = config.CCPO.QRF_N_ESTIMATORS,
        max_d = config.CCPO.QRF_MAX_DEPTH,
        criterion = config.CCPO.CRITERION
    )
    
    try:
        # Step 1: Train ensemble models
        print(f"    Training {config.CCPO.B} bootstrap models...")
        from layers.multi_cp import SPCI_and_EnbPI
        
        # Convert to tensors from loaders
        X_train, Y_train = train_loader.dataset.X, train_loader.dataset.y
        X_valid, Y_valid = valid_loader.dataset.X, valid_loader.dataset.y
        X_predict, Y_predict = test_loader.dataset.X, test_loader.dataset.y
        
        conformal_predictor = SPCI_and_EnbPI(
            X_train, X_valid, X_predict,
            Y_train, Y_valid, Y_predict,
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
        
        # Step 2: Calibrate conformal intervals
        print(f"    Calibrating conformal prediction intervals...")
        conformal_predictor.compute_Widths_Ensemble_online(
            alpha=alpha,
            smallT=False,
            use_SPCI=config.CCPO.USE_SPCI,
            past_window=config.CCPO.PAST_WINDOW,
            random_state=config.SEED
        )
        
        mean_coverage, mean_volume, coverage_seq, volume_seq, radius_seq = conformal_predictor.get_results()
        radius = float(np.mean(radius_seq))
        cov_matrix = conformal_predictor.global_cov
        
        print(f"    ✅ Calibration done - Coverage: {mean_coverage:.3f}, Radius: {radius:.6f}")
        print(f"       Volume: {mean_volume:.6f}, Coverage std: {np.std(coverage_seq):.6f}")
        
        # Step 3: For each V period, predict and optimize
        print(f"    Optimizing portfolio for each of {len(V_dates)} test periods...")
        portfolios_list = []
        
        # Full return series for inference context
        full_returns = loader.resample_frequency(loader.raw_data, config.FREQUENCY)
        full_returns_values = full_returns.values
        full_returns_dates = full_returns.index
        
        for v_idx, (v_date, v_return) in enumerate(zip(V_dates, V_returns)):
            # Find index of v_date in full series
            try:
                date_idx = full_returns_dates.get_loc(v_date)
            except KeyError:
                date_idx = full_returns_dates.get_indexer([v_date], method='nearest')[0]
            
            # Build lookback window
            if date_idx < lookback:
                print(f"      ⚠️  Skipping {v_date}: not enough history (idx={date_idx}, need {lookback})")
                continue
            
            X_test = full_returns_values[date_idx - lookback: date_idx]  # (lookback, n_assets) RETURNS
            X_test_scaled = scaler.transform(X_test)
            X_test_tensor = torch.FloatTensor(X_test_scaled).unsqueeze(0).to(config.DEVICE)  # (1, lookback, n_assets)
            
            # Predict via ensemble
            predictions = []
            for b in range(config.CCPO.B):
                model = conformal_predictor.models[b]
                model.eval()
                with torch.no_grad():
                    pred = model(X_test_tensor)
                    if len(pred.shape) == 3:
                        pred = pred.squeeze(1)  # (1, n_assets)
                    predictions.append(pred)
            
            mean_pred = torch.stack(predictions).mean(dim=0).squeeze()  # (n_assets,)
            if len(mean_pred.shape) == 0:
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
                    'threshold': opt_result['threshold'],
                    'mu_pred': mu_pred
                })
            else:
                print(f"      ⚠️  Optimization failed for {v_date}: {opt_result['status']}")
                n_assets = len(mu_pred)
                portfolios_list.append({
                    'date': v_date,
                    'weights': np.ones(n_assets) / n_assets,
                    'threshold': None,
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
    lookback: int = None,
    alpha: float = None,
):
    """
    Direct evaluation with a single split using our TimeSeriesDataLoader.create_all pipeline.
    - opt_res에는 y_L이 없다는 설계를 따른다(K/V만 사용).
    - CPP: L-calibration 생략(L_returns=None) → threshold_post = threshold_pre.
    - CCPO: train/valid/test_loader 내부 로직으로 캘리브레이션 진행.
    """
    # -------------------------
    # 0) 기본 설정
    # -------------------------
    data_path = data_path or config.DATA_PATH
    frequency = frequency or config.FREQUENCY
    lookback = lookback or config.LOOKBACK
    alpha = alpha or config.ALPHA

    # 결과 폴더 & 로거
    timestamp = datetime.now().strftime("%m%d%H%M")
    result_folder = f"./results/run_direct_{timestamp}"
    os.makedirs(result_folder, exist_ok=True)

    log_file = os.path.join(result_folder, "direct_log.txt")
    logger = DirectLogger(log_file)
    logger.log_header()

    # stdout 리다이렉션
    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print("🎯 Direct Evaluation (Single Split via create_all)")
        print("\nConfiguration:")
        print(f"  MODE: {config.MODE}")
        print(f"  Data: {data_path}")
        print(f"  Frequency: {frequency}")
        print(f"  K: from loader (MODE-dependent), V: remaining (no L in opt)")
        print(f"  Lookback={lookback}, Alpha={alpha}\n")

        # -------------------------
        # 1) 로더 준비 및 데이터 생성
        # -------------------------
        loader = TimeSeriesDataLoader(base_path="./data", num_assets=config.NUM_ASSETS)
        loader.load_data()

        create_kwargs = _build_create_all_kwargs(config)
        print(f"create_all kwargs: {create_kwargs}")
        res = loader.create_all(**create_kwargs)

        # res 구조 가정:
        # res['opt']: {'y_K','y_V','dates_K','dates_V'}  # y_L 없음(의도)
        # res['model']: {'train_loader','valid_loader','test_loader','dates'}
        # res['scaler']: StandardScaler 등
        opt_res = res["opt"]
        model_res = res["model"]
        scaler = res["scaler"]

        # K/V 리턴과 날짜
        K_returns = np.asarray(opt_res["y_K"])             # (K, n_assets)
        V_returns = np.asarray(opt_res["y_V"])             # (V, n_assets)
        dates_K = pd.DatetimeIndex(opt_res["dates_K"])
        V_dates = pd.DatetimeIndex(opt_res["dates_V"])

        asset_names = loader.raw_data.columns.tolist()
        n_assets = len(asset_names)
        K_eff, n_assets_k = K_returns.shape
        V_eff, n_assets_v = V_returns.shape
        assert n_assets_k == n_assets_v == n_assets, "Asset dimension mismatch between K and V."

        print("Data loaded via loader.create_all:")
        print(f"  Assets ({n_assets}): {asset_names}")
        print(f"  K={K_eff}  [{dates_K.min()} ~ {dates_K.max()}]")
        print(f"  V={V_eff}  [{V_dates.min()} ~ {V_dates.max()}]\n")

        # -------------------------
        # 2) 메서드 및 포트폴리오 객체
        # -------------------------
        cpp_methods = config.CPP.METHODS
        ccpo_methods = ["CCPO-CCO"]
        baseline_methods = ["Equal-Weight"]
        all_methods = cpp_methods + ccpo_methods + baseline_methods

        portfolios = {m: Portfolio(name=m) for m in all_methods}
        results = {}

        print("=" * 80)
        print("RUNNING EXPERIMENTS")
        print("=" * 80 + "\n")

        # -------------------------
        # 3) CPP 메서드 실행 (L 없음)
        # -------------------------
        for cpp_method in cpp_methods:
            cpp_res = run_cpp_direct(
                K_returns=K_returns,
                L_returns=None,       
                V_returns=V_returns,
                method=cpp_method,
                alpha=alpha,
            )
            results[cpp_method] = cpp_res

            if cpp_res.get("status") == "optimal":
                weights = cpp_res["weights"]
                threshold = cpp_res["threshold_post"]  # == threshold_pre
                for date, asset_ret in zip(V_dates, V_returns):
                    realized_return = float(weights @ asset_ret)
                    portfolios[cpp_method].add_period(
                        date=date,
                        weight=weights,
                        realized_return=realized_return,
                        solve_time=cpp_res.get("solve_time", 0.0),
                        threshold_post=threshold,
                    )

        # -------------------------
        # 4) CCPO 실행 (주별 리밸런싱)
        # -------------------------
        ccpo_res = run_ccpo_direct(
            data_path=data_path,
            lookback=lookback,
            K=K_eff,
            L=0,        
            V=V_eff,
            start_idx=0,
            alpha=alpha,
            V_dates=V_dates,
            V_returns=V_returns,
        )
        results["CCPO-CCO"] = ccpo_res

        if ccpo_res.get("status") == "optimal" and len(ccpo_res.get("portfolios", [])) > 0:
            for pinfo in ccpo_res["portfolios"]:
                date = pinfo["date"]
                weights = pinfo["weights"]
                threshold = pinfo["threshold"]
                idx = V_dates.get_loc(date)
                realized_return = float(weights @ V_returns[idx])
                portfolios["CCPO-CCO"].add_period(
                    date=date,
                    weight=weights,
                    realized_return=realized_return,
                    solve_time=0.0,
                    threshold_post=threshold,
                )
            print(f"    ✅ CCPO completed with {len(ccpo_res['portfolios'])} rebalancing periods")
            print(
                f"       Calibration - Coverage: {ccpo_res['coverage']:.3f}, "
                f"Volume: {ccpo_res['volume']:.6f}"
            )
            print(
                f"       Coverage std: {np.std(ccpo_res['volume_seq']):.6f}, "
                f"Volume std: {np.std(ccpo_res['volume_seq']):.6f}"
            )

        # -------------------------
        # 5) Equal-Weight 베이스라인
        # -------------------------
        print("  Running Equal-Weight...")
        equal_w = np.ones(n_assets) / n_assets
        for date, asset_ret in zip(V_dates, V_returns):
            portfolios["Equal-Weight"].add_period(
                date=date,
                weight=equal_w,
                realized_return=float(equal_w @ asset_ret),
                solve_time=0.0,
                threshold_post=None,
            )
        print("    ✅ Completed")

        # -------------------------
        # 6) 결과 집계/저장
        # -------------------------
        print(f"\n{'=' * 80}")
        print("📊 RESULTS")
        print(f"{'=' * 80}\n")

        periods_per_year = config.get_periods_per_year(frequency)
        performance_df = compare_methods(portfolios, periods_per_year=periods_per_year)

        print("📈 Performance Comparison:")
        print(performance_df.to_string())
        print()

        # 상세 지표 출력
        for method in all_methods:
            if len(portfolios[method]) > 0:
                metrics = calculate_portfolio_metrics(
                    portfolios[method], periods_per_year=periods_per_year
                )
                print_portfolio_metrics(metrics, portfolio_name=method)

        # 성능 저장
        performance_path = os.path.join(result_folder, "direct_performance.csv")
        performance_df.to_csv(performance_path)
        print(f"\n💾 Performance comparison saved to '{performance_path}'")

        # 각 메서드 가중치 저장
        for method in all_methods:
            if len(portfolios[method]) > 0:
                weights_df = pd.DataFrame(
                    portfolios[method].weights,
                    columns=asset_names,
                    index=portfolios[method].dates,
                )
                weights_path = os.path.join(result_folder, f"{method}_weights.csv")
                weights_df.to_csv(weights_path)
                print(f"💾 {method} weights saved to '{weights_path}' ({len(weights_df)} periods)")

        # 요약 저장 (커버리지/임계값/볼륨 등)
        summary_rows = []
        for method in all_methods:
            if method in results and results[method].get("status") == "optimal":
                row = {"Method": method, "Status": "optimal"}
                if "coverage_post" in results[method]:
                    row["Coverage"] = results[method]["coverage_post"]
                    row["Threshold"] = results[method]["threshold_post"]
                if "coverage" in results[method]:
                    row["Coverage"] = results[method]["coverage"]
                    row["Threshold"] = results[method].get("threshold")
                    if "volume" in results[method]:
                        row["Volume"] = results[method]["volume"]
                if "solve_time" in results[method]:
                    row["Solve_Time"] = results[method]["solve_time"]
                summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(result_folder, "direct_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"💾 Summary saved to '{summary_path}'")

        # CCPO 캘리브레이션 상세 저장
        if "CCPO-CCO" in results and "coverage_seq" in results["CCPO-CCO"]:
            c = results["CCPO-CCO"]
            ccpo_calib_df = pd.DataFrame(
                {
                    "Period": np.arange(len(c["coverage_seq"])),
                    "Coverage": c["coverage_seq"],
                    "Volume": c["volume_seq"],
                    "Radius": c["radius_seq"],
                }
            )
            ccpo_calib_path = os.path.join(result_folder, "ccpo_calibration_details.csv")
            ccpo_calib_df.to_csv(ccpo_calib_path, index=False)
            print(f"💾 CCPO calibration details saved to '{ccpo_calib_path}'")

        # 시각화
        create_all_plots(portfolios, result_folder, prefix="direct")

        print(f"\n📁 All results saved to: {result_folder}")
        print(f"📝 Log saved to: {log_file}")

        return {
            "portfolios": portfolios,
            "performance": performance_df,
            "summary": summary_df,
            "result_folder": result_folder,
        }

    finally:
        # 항상 stdout 원복
        sys.stdout = original_stdout


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    set_seed(config.SEED)
    results = run_direct_evaluation(
        data_path=config.DATA_PATH,
        frequency=config.FREQUENCY,
        lookback=config.LOOKBACK,
        alpha=config.ALPHA
    )
    print("\n✅ Direct evaluation completed!")
