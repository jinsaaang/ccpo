import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
from configs import config_revised as config
from data.data_loader_final import TimeSeriesDataLoader
from utils.portfolios import Portfolio
from utils.evaluation_utils import DirectLogger, _build_create_all_kwargs, aggregate_and_save_results
from evaluation.evaluation_runners import run_cpp_direct, run_ccpo_direct

def run_direct_evaluation(
    data_path: str = None, 
    frequency: str = None,
    lookback: int = None,
    alpha: float = None,
    cfg: config = config
):
    """
    Direct evaluation with a single split using config settings.
    """
    # 0) 기본 설정 로드
    frequency = frequency or cfg.FREQUENCY
    lookback = lookback or cfg.LOOKBACK
    alpha = alpha or cfg.ALPHA

    # 결과 폴더 & 로거
    timestamp = datetime.now().strftime("%m%d%H%M")
    result_folder = os.path.join(
        os.path.dirname(__file__),
        "..", "results",
        f"run_direct_{cfg.MODE}_{timestamp}"
    )
    os.makedirs(result_folder, exist_ok=True)

    log_file = os.path.join(result_folder, "direct_log.txt")
    logger = DirectLogger(log_file)
    logger.log_header(title=f"Direct Evaluation (MODE={cfg.MODE})")

    original_stdout = sys.stdout
    sys.stdout = logger

    try:
        print(f"🎯 Direct Evaluation (Single Split via create_all, MODE={cfg.MODE})")
        print("\nConfiguration:")
        print(f"  Frequency: {frequency}")
        print(f"  Lookback={lookback}, Alpha={alpha}\n")

        create_kwargs = _build_create_all_kwargs(cfg)
        print(f"create_all kwargs (direct): {create_kwargs}")

        # 데이터 로드 (자산 이름 파악용)
        temp_loader = TimeSeriesDataLoader(base_path=cfg.DATA_PATH, num_assets=cfg.NUM_ASSETS)
        temp_loader.load_data()
        asset_names = temp_loader.raw_data.columns.tolist()
        n_assets = len(asset_names)
        del temp_loader

        print("Fetching K and V returns for CPP...")
        temp_loader_for_cpp = TimeSeriesDataLoader(base_path=cfg.DATA_PATH, num_assets=cfg.NUM_ASSETS)
        res_for_cpp = temp_loader_for_cpp.create_all(**create_kwargs)
        K_returns_raw = res_for_cpp['opt']['y_K']
        V_returns_raw = res_for_cpp['opt']['y_V']
        K_dates = pd.DatetimeIndex(res_for_cpp['opt']['dates_K'])
        V_dates = pd.DatetimeIndex(res_for_cpp['opt']['dates_V'])
        del temp_loader_for_cpp, res_for_cpp

        print("Data Split Info (from direct config):")
        print(f"  Assets ({n_assets}): {asset_names}")
        print(f"  K Period Opt Data: {len(K_returns_raw)} obs [{K_dates.min().date() if len(K_dates)>0 else 'N/A'} ~ {K_dates.max().date() if len(K_dates)>0 else 'N/A'}]")
        print(f"  V Period Opt Data: {len(V_returns_raw)} obs [{V_dates.min().date() if len(V_dates)>0 else 'N/A'} ~ {V_dates.max().date() if len(V_dates)>0 else 'N/A'}]\n")

        cpp_methods = cfg.CPP.METHODS
        ccpo_methods = ["CCPO-CCO"]  # 현재 CCPO-CCO 하나만 사용
        baseline_methods = ["Equal-Weight"]
        all_methods = cpp_methods + ccpo_methods + baseline_methods

        portfolios = {m: Portfolio(name=m) for m in all_methods}
        results = {}

        print("=" * 80)
        print("RUNNING EXPERIMENTS")
        print("=" * 80 + "\n")

        for cpp_method in cpp_methods:
            cpp_res = run_cpp_direct(
                K_returns=K_returns_raw,
                L_returns=None,
                V_returns=V_returns_raw,
                method=cpp_method,
                alpha=alpha,
            )
            results[cpp_method] = cpp_res

            if cpp_res.get("status") == "optimal":
                weights = cpp_res["weights"]
                threshold = cpp_res["threshold_post"]
                for date, asset_ret in zip(V_dates, V_returns_raw):
                    portfolios[cpp_method].add_period(
                        date=date, weight=weights,
                        realized_return=float(weights @ asset_ret),
                        solve_time=cpp_res.get("solve_time", 0.0) / len(V_dates) if len(V_dates)>0 else 0.0,
                        threshold_post=threshold,
                    )

        ccpo_res = run_ccpo_direct(
            data_path=cfg.DATA_PATH,
            lookback=lookback,
            alpha=alpha,
            cfg=cfg
        )
        
        results["CCPO-CCO"] = ccpo_res

        if ccpo_res.get("status") == "optimal" and "portfolios" in ccpo_res:
            for pinfo in ccpo_res["portfolios"]:
                date = pinfo["date"]
                weights = pinfo["weights"]
                threshold = pinfo["threshold"]
                try:
                    idx = V_dates.get_loc(date)
                    asset_ret_raw = V_returns_raw[idx]
                    realized_return = float(weights @ asset_ret_raw)
                    portfolios["CCPO-CCO"].add_period(
                        date=date, weight=weights,
                        realized_return=realized_return,
                        solve_time=ccpo_res.get("optimization_time", 0.0) / len(V_dates) if len(V_dates)>0 else 0.0,
                        threshold_post=threshold,
                    )
                except KeyError:
                    print(f"  ⚠️ CCPO Warning: Date {date.date()} from optimization not found in V_dates. Skipping portfolio log.")
                except Exception as e:
                    print(f"  ⚠️ Error processing CCPO result for date {date.date()}: {e}")

            print(f"    ✅ CCPO completed. Processed {len(portfolios['CCPO-CCO'])} V periods.")
            if "coverage" in ccpo_res:
                print(f"       Calibration Info (on K) - Coverage: {ccpo_res['coverage']:.3f}, Radius: {ccpo_res.get('threshold', 'N/A'):.6f}")

        # Equal-Weight 베이스라인
        print("  Running Equal-Weight...")
        if n_assets > 0:
            equal_w = np.ones(n_assets) / n_assets
            for date, asset_ret in zip(V_dates, V_returns_raw):
                portfolios["Equal-Weight"].add_period(
                    date=date, weight=equal_w,
                    realized_return=float(equal_w @ asset_ret),
                    solve_time=0.0, threshold_post=None,
                )
            print(f"    ✅ Completed {len(V_dates)} periods.")
        else:
            print("    Skipped (no assets).")

        aggregate_and_save_results(
            portfolios=portfolios, result_folder=result_folder,
            asset_names=asset_names, prefix="direct",
            cfg=cfg, results=results
        )
        print(f"📝 Log saved to: {log_file}")

        return { "portfolios": portfolios, "result_folder": result_folder }

    finally:
        sys.stdout = original_stdout