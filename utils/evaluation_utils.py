import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

# Import unified config
from configs import config_revised_prev as config

# Import portfolio utilities
from utils.portfolios import Portfolio
from utils.metrics import calculate_portfolio_metrics, compare_methods, print_portfolio_metrics
from utils.visualization import create_all_plots

# ============================================================================
# LOGGING UTILITY
# ============================================================================

class DirectLogger:
    """Logger that writes to both console and file"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Clear log file if it exists
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("")
        
    def write(self, message):
        self.terminal.write(message)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(message)
    
    def flush(self):
        self.terminal.flush()
    
    def log_header(self, title="Evaluation Run"):
        header = f"\n{'='*80}\n"
        header += f"{title} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"{'='*80}\n"
        self.write(header)


# ============================================================================
# CONFIG HELPER
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
            raise ValueError(f"[MODE=dates] 다음 설정이 필요합니다: {missing}")
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
            raise ValueError(f"[MODE=counts] 다음 설정이 필요합니다: {missing}")
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
# RESULT AGGREGATION & SAVING
# ============================================================================

def aggregate_and_save_results(
    portfolios: Dict[str, Portfolio],
    result_folder: str,
    asset_names: List[str],
    prefix: str,
    cfg: config,
    results: Dict = None
):
    """
    모든 평가(direct, rolling)의 결과를 집계, 저장, 시각화하는 공통 함수
    """
    print(f"\n{'=' * 80}")
    print(f"📊 {prefix.upper()} RESULTS (Aggregated)")
    print(f"{'=' * 80}\n")
    
    all_methods = list(portfolios.keys())
    periods_per_year = cfg.get_periods_per_year(cfg.FREQUENCY)
    
    # 비어있는 포트폴리오 제거
    valid_portfolios = {m: p for m, p in portfolios.items() if len(p) > 0}
    if not valid_portfolios:
        print("⚠️ No valid portfolio results found to aggregate.")
        return

    performance_df = compare_methods(valid_portfolios, periods_per_year=periods_per_year)

    print(f"📈 Performance Comparison ({prefix}):")
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
    performance_path = os.path.join(result_folder, f"{prefix}_performance.csv")
    performance_df.to_csv(performance_path)
    print(f"\n💾 Performance comparison saved to '{performance_path}'")

    # 가중치 저장
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

    # 요약 저장 (결과가 있는 경우, 주로 'direct' 모드용)
    if results:
        summary_rows = []
        for method in all_methods:
            if method in results and results[method].get("status") == "optimal":
                row = {"Method": method, "Status": "optimal"}
                if "coverage_post" in results[method]:
                    row["Coverage"] = results[method]["coverage_post"]
                    row["Threshold"] = results[method]["threshold_post"]
                if "coverage" in results[method]: # CCPO
                    row["Coverage"] = results[method]["coverage"]
                    row["Threshold"] = results[method].get("threshold")
                    if "volume" in results[method]:
                        row["Volume"] = results[method]["volume"]
                if "solve_time" in results[method]:
                    row["Solve_Time"] = results[method]["solve_time"]
                summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = os.path.join(result_folder, f"{prefix}_summary.csv")
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
            ccpo_calib_path = os.path.join(result_folder, f"{prefix}_ccpo_calibration.csv")
            ccpo_calib_df.to_csv(ccpo_calib_path, index=False)
            print(f"💾 CCPO calibration details saved to '{ccpo_calib_path}'")

    # 시각화
    create_all_plots(valid_portfolios, result_folder, prefix=prefix)
    
    print(f"\n📁 All results saved to: {result_folder}")