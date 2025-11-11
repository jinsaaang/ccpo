"""
CCPO만 단독으로 Rolling Window Evaluation 실행
CPP 메서드 없이 CCPO만 전체 rolling으로 테스트
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import config_revised as config
from layers.cp_utils import set_seed
from evaluation.evaluation_runners import run_ccpo_rolling_counts
from data.data_loader_final import TimeSeriesDataLoader
from utils.portfolios import Portfolio
from utils.metrics import calculate_portfolio_metrics
from utils.evaluation_utils import aggregate_and_save_results
import pandas as pd
import numpy as np
from datetime import datetime

def test_ccpo_rolling():
    """CCPO Rolling Window 테스트 (CPP 없이)"""
    print("="*80)
    print("CCPO 단독 Rolling Window Evaluation")
    print("="*80)
    
    set_seed(config.SEED)
    
    # 데이터 로드
    loader = TimeSeriesDataLoader(base_path=config.DATA_PATH, num_assets=config.NUM_ASSETS)
    loader.load_data()  # raw_data 로드
    full_data_resampled = loader.resample_frequency(loader.raw_data, config.FREQUENCY)
    
    total_len, n_assets = full_data_resampled.shape
    print(f"\nFull data loaded: {total_len} periods, {n_assets} assets")
    print(f"  [{full_data_resampled.index.min().date()} ~ {full_data_resampled.index.max().date()}]")
    
    # Rolling window 설정
    cfg_roll = config.ROLLING
    cfg_roll_cnt = config.ROLLING.COUNTS
    
    print(f"\nRolling Config (counts) - {cfg_roll.WINDOW_TYPE} window:")
    print(f"  TrainLen={cfg_roll_cnt.TRAIN_LEN}, TestLen={cfg_roll_cnt.TEST_LEN}, Step={cfg_roll_cnt.STEP_SIZE}")
    
    lookback = config.LOOKBACK
    train_raw_len = lookback + cfg_roll_cnt.TRAIN_LEN
    test_raw_len = cfg_roll_cnt.TEST_LEN
    step_size = cfg_roll_cnt.STEP_SIZE
    
    # Window definitions 생성
    window_definitions = []
    initial_train_start_idx = 0
    current_test_end_idx = initial_train_start_idx + train_raw_len + test_raw_len
    
    print(f"\nGenerating windows...")
    while True:
        if cfg_roll.WINDOW_TYPE == "expanding":
            train_start_idx = initial_train_start_idx
            test_end_idx = current_test_end_idx
            train_end_idx = test_end_idx - test_raw_len
        elif cfg_roll.WINDOW_TYPE == "sliding":
            test_end_idx = current_test_end_idx
            train_end_idx = test_end_idx - test_raw_len
            train_start_idx = train_end_idx - train_raw_len
        else:
            raise ValueError(f"Unknown WINDOW_TYPE: {cfg_roll.WINDOW_TYPE}")
        
        if test_end_idx > total_len:
            break
        
        window_definitions.append({
            "train_start_idx": train_start_idx,
            "train_end_idx": train_end_idx,
            "test_start_idx": train_end_idx,
            "test_end_idx": test_end_idx,
        })
        current_test_end_idx += step_size
    
    print(f"Total windows: {len(window_definitions)}")
    
    # CCPO Portfolio 생성
    ccpo_portfolio = Portfolio(name="CCPO-CCO")
    accumulated_test_residuals_list = []
    
    # 각 window별로 CCPO 실행
    for window_idx, window in enumerate(window_definitions):
        window_num = window_idx + 1
        print(f"\n{'='*80}")
        print(f"WINDOW {window_num}/{len(window_definitions)}")
        print(f"{'='*80}")
        
        train_start_idx = window["train_start_idx"]
        train_end_idx = window["train_end_idx"]
        test_start_idx = window["test_start_idx"]
        test_end_idx = window["test_end_idx"]
        
        train_data = full_data_resampled.iloc[train_start_idx:train_end_idx]
        test_data = full_data_resampled.iloc[test_start_idx:test_end_idx]
        
        print(f"Train: [{train_start_idx}:{train_end_idx}] = {len(train_data)} periods")
        print(f"       {train_data.index.min().date()} ~ {train_data.index.max().date()}")
        print(f"Test:  [{test_start_idx}:{test_end_idx}] = {len(test_data)} periods")
        print(f"       {test_data.index.min().date()} ~ {test_data.index.max().date()}")
        
        test_returns_raw = test_data.values
        test_dates = test_data.index
        
        # Accumulated residuals 준비
        accumulated_residuals = None
        if config.CCPO.USE_COV_UPDATE and len(accumulated_test_residuals_list) > 0:
            accumulated_residuals = np.vstack(accumulated_test_residuals_list)
            print(f"Using {len(accumulated_residuals)} accumulated residuals from previous windows")
        
        try:
            ccpo_result = run_ccpo_rolling_counts(
                data_path=config.DATA_PATH,
                lookback=config.LOOKBACK,
                alpha=config.ALPHA,
                train_len=cfg_roll_cnt.TRAIN_LEN,
                test_len=cfg_roll_cnt.TEST_LEN,
                start_idx=train_start_idx,
                test_dates=test_dates,
                test_returns_raw=test_returns_raw,
                cfg=config,
                accumulated_test_residuals=accumulated_residuals
            )
            
            if ccpo_result.get('status') == 'optimal':
                print(f"✅ Window {window_num} Success!")
                print(f"   Coverage: {ccpo_result.get('coverage_calib', 0):.3f}")
                print(f"   Radius: {ccpo_result.get('radius', 0):.6f}")
                print(f"   Portfolios: {len(ccpo_result.get('portfolios', []))}")
                
                # Portfolio에 추가
                for portfolio_data in ccpo_result['portfolios']:
                    # 해당 날짜의 실제 수익률 계산
                    date_idx = full_data_resampled.index.get_loc(portfolio_data['date'])
                    realized_return = float(full_data_resampled.iloc[date_idx].values @ portfolio_data['weights'])
                    
                    ccpo_portfolio.add_period(
                        date=portfolio_data['date'],
                        weight=portfolio_data['weights'],
                        realized_return=realized_return,
                        solve_time=0.0,  # 개별 period는 solve time 없음
                        threshold_post=portfolio_data.get('threshold')
                    )
                
                # Residuals 저장
                if config.CCPO.USE_COV_UPDATE and ccpo_result.get('test_residuals') is not None:
                    accumulated_test_residuals_list.append(ccpo_result['test_residuals'])
            else:
                print(f"❌ Window {window_num} Failed: {ccpo_result.get('status')}")
        
        except Exception as e:
            print(f"❌ Window {window_num} Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    # 최종 결과 요약
    print(f"\n{'='*80}")
    print("CCPO Rolling Evaluation Summary")
    print(f"{'='*80}")
    print(f"Total windows: {len(window_definitions)}")
    print(f"Total portfolio entries: {len(ccpo_portfolio)}")
    
    if len(ccpo_portfolio) > 0:
        # 성과 계산
        periods_per_year = config.get_periods_per_year(config.FREQUENCY)
        metrics = calculate_portfolio_metrics(ccpo_portfolio, periods_per_year=periods_per_year)
        
        print(f"\nCCPO-CCO Performance:")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  Annual Return: {metrics.get('annual_return', 0)*100:.2f}%")
        print(f"  Annual Volatility: {metrics.get('annual_volatility', 0)*100:.2f}%")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0)*100:.2f}%")
        
        # 결과 저장
        timestamp = datetime.now().strftime("%m%d%H%M")
        result_folder = os.path.join("results", f"test_ccpo_only_{timestamp}")
        os.makedirs(result_folder, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("Saving Results...")
        print(f"{'='*80}")
        
        # Asset names 가져오기
        asset_names = [f"Asset_{i+1}" for i in range(n_assets)]
        
        # Aggregate and save
        portfolios = {"CCPO-CCO": ccpo_portfolio}
        aggregate_and_save_results(
            portfolios=portfolios,
            result_folder=result_folder,
            asset_names=asset_names,
            prefix="test_ccpo_only",
            cfg=config,
            results=None
        )
        
        return ccpo_portfolio, metrics, result_folder
    else:
        print("No portfolios generated!")
        return None, None, None


if __name__ == "__main__":
    print("\n🎯 CCPO 단독 테스트 시작\n")
    
    print(f"Config 설정:")
    print(f"  - Frequency: {config.FREQUENCY}")
    print(f"  - Lookback: {config.LOOKBACK}")
    print(f"  - Alpha: {config.ALPHA}")
    print(f"  - Assets: {config.NUM_ASSETS}")
    print(f"  - USE_MULTISTEP: {config.CCPO.USE_MULTISTEP}")
    print(f"  - AGGREGATION_METHOD: {config.CCPO.AGGREGATION_METHOD}")
    print(f"  - LOSS_AGGREGATION: {config.CCPO.LOSS_AGGREGATION}")
    print(f"  - USE_COV_UPDATE: {config.CCPO.USE_COV_UPDATE}")
    print(f"  - Bootstrap models (B): {config.CCPO.B}")
    print(f"  - Epochs: {config.CCPO.EPOCHS}")
    
    portfolio, metrics, result_folder = test_ccpo_rolling()
    
    if result_folder:
        print(f"\n✅ 테스트 완료! 결과가 저장되었습니다: {result_folder}")
    
    if portfolio and len(portfolio) > 0:
        print("\n" + "="*80)
        print("✅ CCPO Rolling Test Complete!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ CCPO Rolling Test Failed - No portfolios generated")
        print("="*80)
