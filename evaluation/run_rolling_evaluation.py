import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from pandas.tseries.offsets import DateOffset
from configs import config_revised as config
from data.data_loader_final import TimeSeriesDataLoader
from utils.portfolios import Portfolio
from utils.evaluation_utils import DirectLogger, aggregate_and_save_results
from evaluation.evaluation_runners import run_cpp_direct, run_ccpo_rolling_counts, run_ccpo_rolling_dates

def run_rolling_evaluation(
    data_path: str = None,
    frequency: str = None,
    lookback: int = None,
    alpha: float = None,
    cfg: config = config
):
    """
    Rolling window evaluation (Sliding or Expanding) based on config diagram.
    Splits into distinct Train / K (Calib) / V (Test) periods.
    """
    frequency = frequency or cfg.FREQUENCY
    lookback = lookback or cfg.LOOKBACK
    alpha = alpha or cfg.ALPHA
    cfg_roll = cfg.ROLLING

    timestamp = datetime.now().strftime("%m%d%H%M")
    result_folder = os.path.join(
        os.path.dirname(__file__),
        "..", "results",
        f"run_rolling_{cfg_roll.WINDOW_TYPE}_{cfg.MODE}_{timestamp}"
    )
    os.makedirs(result_folder, exist_ok=True)

    log_file = os.path.join(result_folder, "rolling_log.txt")
    logger = DirectLogger(log_file)
    logger.log_header(title=f"Rolling Evaluation ({cfg_roll.WINDOW_TYPE} / {cfg.MODE})")

    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print(f"🎯 Rolling Window Evaluation ({cfg_roll.WINDOW_TYPE} / {cfg.MODE})")
        print(f"  Frequency: {frequency}, Lookback: {lookback}, Alpha: {alpha}")

        loader = TimeSeriesDataLoader(base_path=cfg.DATA_PATH, num_assets=cfg.NUM_ASSETS)
        loader.load_data()
        full_data_resampled = loader.resample_frequency(loader.raw_data, frequency)
        total_len, n_assets = full_data_resampled.shape
        asset_names = full_data_resampled.columns.tolist()
        print(f"\nFull data loaded: {total_len} periods, {n_assets} assets")
        print(f"  [{full_data_resampled.index.min().date()} ~ {full_data_resampled.index.max().date()}]")

        cpp_methods = cfg.CPP.METHODS
        ccpo_methods = ["CCPO-CCO"]
        baseline_methods = ["Equal-Weight"]
        all_methods = cpp_methods + ccpo_methods + baseline_methods
        portfolios = {m: Portfolio(name=m) for m in all_methods}

        window_definitions = []

        if cfg.MODE == "counts":
            cfg_roll_cnt = cfg.ROLLING.COUNTS
            print(f"\nRolling Config (counts) - {cfg_roll.WINDOW_TYPE} window:")
            print(f"  TrainLen={cfg_roll_cnt.TRAIN_LEN}, TestLen={cfg_roll_cnt.TEST_LEN}, Step={cfg_roll_cnt.STEP_SIZE}")

            train_raw_len = lookback + cfg_roll_cnt.TRAIN_LEN
            test_raw_len = cfg_roll_cnt.TEST_LEN
            step_size = cfg_roll_cnt.STEP_SIZE

            # Initial window start
            initial_train_start_idx = 0
            
            # Expanding: test_end moves forward by step_size
            # Sliding: both train_start and test_end move forward by step_size
            current_test_end_idx = initial_train_start_idx + train_raw_len + test_raw_len
            
            while True:
                if cfg_roll.WINDOW_TYPE == "expanding":
                    # Expanding: train_start fixed at 0, train grows with each window
                    train_start_idx = initial_train_start_idx
                    test_end_idx = current_test_end_idx
                    train_end_idx = test_end_idx - test_raw_len
                    
                elif cfg_roll.WINDOW_TYPE == "sliding":
                    # Sliding: fixed window size, moves forward
                    test_end_idx = current_test_end_idx
                    train_end_idx = test_end_idx - test_raw_len
                    train_start_idx = train_end_idx - train_raw_len
                else:
                    raise ValueError(f"Unknown ROLLING.WINDOW_TYPE: {cfg_roll.WINDOW_TYPE}")

                # Check if we've reached the end of data
                if test_end_idx > total_len:
                    print("\n--- Reached end of data (counts). Stopping. ---")
                    break

                if train_end_idx <= train_start_idx or test_end_idx <= train_end_idx or train_start_idx < 0:
                    print(f"--- Skipping invalid window: train[{train_start_idx}:{train_end_idx}], test[{train_end_idx}:{test_end_idx}] ---")
                    current_test_end_idx += step_size
                    continue

                window_definitions.append({
                    "mode": "counts",
                    "train_start_idx": train_start_idx,
                    "train_len": cfg_roll_cnt.TRAIN_LEN,
                    "test_len": cfg_roll_cnt.TEST_LEN,
                    "train_start_raw_idx": train_start_idx,
                    "train_end_raw_idx": train_end_idx,
                    "test_start_raw_idx": train_end_idx,
                    "test_end_raw_idx": test_end_idx,
                })

                # Move test_end forward for next window
                current_test_end_idx += step_size

        elif cfg.MODE == "dates":
            cfg_roll_dt = cfg.ROLLING.DATES
            print(f"\nRolling Config (dates):")
            print(f"  Train Offset={cfg_roll_dt.MODEL_TRAIN_OFFSET}, K Period={cfg_roll_dt.K_PERIOD_OFFSET}, V Period={cfg_roll_dt.V_PERIOD_OFFSET}, Step={cfg_roll_dt.STEP_OFFSET}")
            print(f"  Rolling Start={cfg_roll_dt.ROLLING_START_DATE}, Rolling End={cfg_roll_dt.ROLLING_END_DATE}")

            train_offset = pd.tseries.frequencies.to_offset(cfg_roll_dt.MODEL_TRAIN_OFFSET)
            k_offset = pd.tseries.frequencies.to_offset(cfg_roll_dt.K_PERIOD_OFFSET)
            v_offset = pd.tseries.frequencies.to_offset(cfg_roll_dt.V_PERIOD_OFFSET)
            step_offset = pd.tseries.frequencies.to_offset(cfg_roll_dt.STEP_OFFSET)

            first_possible_train_start = pd.to_datetime(cfg_roll_dt.ROLLING_START_DATE) if cfg_roll_dt.ROLLING_START_DATE else full_data_resampled.index[0]
            last_possible_v_end = pd.to_datetime(cfg_roll_dt.ROLLING_END_DATE) if cfg_roll_dt.ROLLING_END_DATE else full_data_resampled.index[-1]

            first_k_end = first_possible_train_start + train_offset + k_offset
            first_v_start = first_k_end

            current_v_start_date = first_v_start
            while True:
                v_end_date = current_v_start_date + v_offset
                if v_end_date > last_possible_v_end + pd.Timedelta(days=1):
                    print(f"\n--- Reached end date {last_possible_v_end.date()}. Stopping. ---")
                    break

                k_end_date = current_v_start_date
                train_end_date = k_end_date - k_offset

                if cfg_roll.WINDOW_TYPE == "expanding":
                    train_start_date = first_possible_train_start
                elif cfg_roll.WINDOW_TYPE == "sliding":
                    train_start_date = train_end_date - train_offset
                else:
                    raise ValueError(f"Unknown ROLLING.WINDOW_TYPE: {cfg_roll.WINDOW_TYPE}")

                min_data_date_for_seq = full_data_resampled.index[lookback]
                if train_start_date < min_data_date_for_seq or train_end_date <= train_start_date or k_end_date <= train_end_date:
                    print(f"--- Skipping window V=[{current_v_start_date.date()} ~ {v_end_date.date()}]: Insufficient history or invalid Train/K dates. ---")
                    current_v_start_date += step_offset
                    continue

                window_definitions.append({
                    "mode": "dates",
                    "train_start_date": train_start_date,
                    "train_end_date": train_end_date,
                    "k_start_date": train_end_date,
                    "k_end_date": k_end_date,
                    "v_start_date": k_end_date,
                    "v_end_date": v_end_date
                })

                current_v_start_date += step_offset

        else:
            raise ValueError(f"Unknown MODE: {cfg.MODE}")

        print(f"\nTotal valid windows defined: {len(window_definitions)}")

        # Initialize accumulator for test residuals (for covariance update)
        accumulated_test_residuals_list = []

        for i, window in enumerate(window_definitions):
            window_num = i + 1

            print(f"\n{'='*80}")
            print(f"RUNNING WINDOW {window_num}/{len(window_definitions)} ({cfg_roll.WINDOW_TYPE} / {cfg.MODE})")

            if window["mode"] == "counts":
                train_start_idx = window["train_start_raw_idx"]
                train_end_idx = window["train_end_raw_idx"]
                test_start_idx = window["test_start_raw_idx"]
                test_end_idx = window["test_end_raw_idx"]
                
                train_data_raw = full_data_resampled.iloc[train_start_idx : train_end_idx]
                test_data_raw = full_data_resampled.iloc[test_start_idx : test_end_idx]
                
                train_s, train_e = train_start_idx, train_end_idx
                test_s, test_e = test_start_idx, test_end_idx
                print(f"  Raw Idx: Train=[{train_s}:{train_e}], Test=[{test_s}:{test_e}]")

            else:
                k_start, k_end = window["k_start_date"], window["k_end_date"]
                v_start, v_end = window["v_start_date"], window["v_end_date"]
                K_data_raw = full_data_resampled.loc[k_start : k_end - pd.Timedelta(nanoseconds=1)]
                V_data_raw = full_data_resampled.loc[v_start : v_end - pd.Timedelta(nanoseconds=1)]
                train_s, train_e = window["train_start_date"], window["train_end_date"]
                print(f"  Dates: Train=[{train_s.date()}~{train_e.date()}], K=[{k_start.date()}~{k_end.date()}], V=[{v_start.date()}~{v_end.date()}]")

            if window["mode"] == "counts":
                if train_data_raw.empty or test_data_raw.empty:
                    print(f"--- Skipping window {window_num}: Empty train or test data. ---")
                    continue
                
                train_returns_raw = train_data_raw.values
                test_returns_raw = test_data_raw.values
                test_dates = test_data_raw.index

                print(f"  Train Period Data: {len(train_data_raw)} obs [{train_data_raw.index.min().date()} ~ {train_data_raw.index.max().date()}]")
                print(f"  Test Period Data: {len(test_data_raw)} obs [{test_dates.min().date()} ~ {test_dates.max().date()}]")
            else:
                if K_data_raw.empty or V_data_raw.empty:
                    print(f"--- Skipping window {window_num}: Empty K or V data. ---")
                    continue
                    
                train_returns_raw = K_data_raw.values
                test_returns_raw = V_data_raw.values
                test_dates = V_data_raw.index

            print(f"{'='*80}\n")

            # 4.2) CPP 실행 (train 기간 데이터 사용)
            window_results = {}
            for cpp_method in cpp_methods:
                cpp_res = run_cpp_direct(
                    K_returns=train_returns_raw,
                    L_returns=None,
                    V_returns=test_returns_raw,
                    method=cpp_method,
                    alpha=alpha,
                )
                window_results[cpp_method] = cpp_res

            # 4.3) CCPO for counts mode
            if window["mode"] == "counts":
                # Prepare accumulated test residuals from previous windows
                accumulated_residuals = None
                if cfg.CCPO.USE_COV_UPDATE and len(accumulated_test_residuals_list) > 0:
                    accumulated_residuals = np.vstack(accumulated_test_residuals_list)
                    print(f"  Using {len(accumulated_residuals)} accumulated test residuals from {len(accumulated_test_residuals_list)} previous windows")
                
                ccpo_res = run_ccpo_rolling_counts(
                    data_path=cfg.DATA_PATH, lookback=lookback, alpha=alpha,
                    train_len=window["train_len"],
                    test_len=window["test_len"],
                    start_idx=window["train_start_idx"],
                    test_dates=test_dates,
                    test_returns_raw=test_returns_raw,
                    cfg=cfg,
                    accumulated_test_residuals=accumulated_residuals
                )
                
                # Store current window's test residuals for next window
                if cfg.CCPO.USE_COV_UPDATE and ccpo_res.get('test_residuals') is not None:
                    accumulated_test_residuals_list.append(ccpo_res['test_residuals'])
                    print(f"  ✅ Stored {len(ccpo_res['test_residuals'])} test residuals for future windows")
                
            else:  # dates mode
                ccpo_res = run_ccpo_rolling_dates(
                    data_path=cfg.DATA_PATH, lookback=lookback, alpha=alpha,
                    train_start_date=window["train_start_date"],
                    train_end_date=window["train_end_date"],
                    K_end_date=window["k_end_date"],
                    V_dates=test_dates,
                    V_returns_raw=test_returns_raw,
                    cfg=cfg
                )
            window_results["CCPO-CCO"] = ccpo_res

            # 4.4) Equal-Weight
            print("  Running Equal-Weight...")
            if n_assets > 0:
                equal_w = np.ones(n_assets) / n_assets
                window_results["Equal-Weight"] = {
                    'weights': equal_w,
                    'threshold_post': None,
                    'status': 'optimal',
                    'solve_time': 0.0
                }
                print(f"    ✅ Completed.")
            else:
                window_results["Equal-Weight"] = {'status': 'skipped'}
                print(f"    Skipped (no assets).")

            # 4.5) 현 윈도우 결과를 전역 포트폴리오에 추가 (test 기간)
            print("\n  Adding window results to portfolio logs...")
            for method in all_methods:
                result = window_results.get(method)
                
                if result and result.get('status') == 'optimal' and method == 'CCPO-CCO':
                    portfolio = result['portfolios']
                    
                    for portfolio_item in portfolio:
                        weights = portfolio_item.get('weights')
                        threshold = portfolio_item.get('threshold')
                        
                    solve_time = result['calibration_time'] + result['optimization_time']
                    
                    if weights is not None:
                        for date, asset_ret_raw in zip(test_dates, test_returns_raw):
                            portfolios[method].add_period(
                                date=date, weight=weights,
                                realized_return=float(weights @ asset_ret_raw),
                                solve_time=solve_time / len(test_dates) if len(test_dates) > 0 else 0.0,
                                threshold_post=threshold,
                            )
                        print(f"    Method '{method}': Added {len(test_dates)} periods.")
                    else:
                        print(f"    Method '{method}': Skipped (no weights).")                            
                    
                                 
                elif result and result.get('status') == 'optimal' and method != 'CCPO-CCO':
                    weights = result.get('weights')
                    threshold = result.get('threshold_post', result.get('threshold'))
                    solve_time = result.get('solve_time', 0.0)
                    
                    if weights is not None:
                        for date, asset_ret_raw in zip(test_dates, test_returns_raw):
                            portfolios[method].add_period(
                                date=date, weight=weights,
                                realized_return=float(weights @ asset_ret_raw),
                                solve_time=solve_time / len(test_dates) if len(test_dates) > 0 else 0.0,
                                threshold_post=threshold,
                            )
                        print(f"    Method '{method}': Added {len(test_dates)} periods.")
                    else:
                        print(f"    Method '{method}': Skipped (no weights).")
                else:
                    status = result.get('status') if isinstance(result, dict) else 'error'
                    print(f"    Method '{method}': Skipped (status: {status}).")
                    
        # 5) Aggregate all results for whole rolling periods
        aggregate_and_save_results(
            portfolios=portfolios, result_folder=result_folder,
            asset_names=asset_names, prefix="rolling_agg",
            cfg=cfg, results=None
        )

        print(f"📝 Log saved to: {log_file}")

        return { "portfolios": portfolios, "result_folder": result_folder }

    finally:
        sys.stdout = original_stdout
