# run_experiment.py

import os
import sys
import pandas as pd
import numpy as np
import torch
import time
import json 

sys.path.append(os.path.join(os.getcwd(), '.'))
from layers.cp_utils import set_seed
from data.data_loader import TimeSeriesDataLoader
from layers.multi_cp import SPCI_and_EnbPI
import configs.config_cp as config_cp

os.makedirs(config_cp.RESULTS_PATH, exist_ok=True)
os.makedirs(config_cp.WEIGHTS_PATH, exist_ok=True)

def run_single_experiment(cfg):
    """For running a single experiment based on the provided configuration."""
    print("--- Start Experiments ---")
    print(f'Seed: {cfg.SEED}')
    print(f"Data Path: {cfg.DATA_PATH}")
    print(f"Model Type: {cfg.MODEL_CLASS.__name__}")
    print(f"Miscoverage Level: {cfg.ALPHA}")
    print(f"Use SPCI: {cfg.USE_SPCI}")
    print(f"Low Rank r: {cfg.LOW_RANK_R}")
    print(f"Use Local Ellipsoid: {cfg.USE_LOCAL_ELLIPSOID}")
    print(f"QRF Parameters: {cfg.QRF_N_ESTIMATORS}, Max Depth: {cfg.QRF_MAX_DEPTH}, Bins: {cfg.QRF_BINS}, Criterion: {cfg.CRITERION},")
    print(f"Epochs: {cfg.EPOCHS}, Batch Size: {cfg.BATCH_SIZE}, Lookback: {cfg.LOOKBACK}")
    print("-" * 20)

    start_time = time.time()
    set_seed(cfg.SEED)  # Seed Setting

    # 1. Data Loading and Preprocessing
    print("Data Loading...")
    data_loader = TimeSeriesDataLoader(data_path=os.path.basename(cfg.DATA_PATH),
                                       base_path=os.path.dirname(cfg.DATA_PATH))

    train_loader, val_loader, test_loader, dates, scaler = data_loader.create_dataloaders(
        frequency=cfg.FREQUENCY,
        lookback=cfg.LOOKBACK,
        forecast_horizon=cfg.FORECAST_HORIZON,
        train_end_date=cfg.TRAIN_END_DATE,
        val_end_date=cfg.VAL_END_DATE,
        test_end_date=cfg.TEST_END_DATE,
        batch_size=cfg.BATCH_SIZE,
        use_scaler=cfg.USE_SCALER
    )
    
    data_loader.load_data()
    data_resampled = data_loader.resample_frequency(data_loader.raw_data, frequency=cfg.FREQUENCY)
    hist_cov = data_resampled.ewm(alpha=cfg.WEIGHT_DECAY).cov()
    
    idx = pd.IndexSlice
    hist_cov_resampled = hist_cov.loc[idx[dates['test'], :], :]
    
    tickers = list(hist_cov_resampled.columns)

    cov_list = []
    for _, block in hist_cov_resampled.groupby(level=0):
        B = block.droplevel(0)               
        B = B.reindex(index=tickers, columns=tickers)  
        cov_list.append(B.to_numpy().tolist())
        
    print("Data Loading Completed.")

    # 2. Conformal Predictor Initialization
    print("Conformal Predictor Initialization...")
    X_train, Y_train = train_loader.dataset.X, train_loader.dataset.y
    X_valid, Y_valid = val_loader.dataset.X, val_loader.dataset.y
    X_predict, Y_predict = test_loader.dataset.X, test_loader.dataset.y
    print(f"Train set: {X_train.shape}, Validation set: {X_valid.shape}, Test set: {X_predict.shape}")
    print(stop)

    conformal_predictor = SPCI_and_EnbPI(
        X_train, X_valid, X_predict,
        Y_train, Y_valid, Y_predict,
        model_cls=cfg.MODEL_CLASS,
        loader=data_loader,
        scaler=scaler,
        device=cfg.DEVICE,
        r=cfg.LOW_RANK_R,
        use_local_ellipsoid=cfg.USE_LOCAL_ELLIPSOID,
        bins=cfg.QRF_BINS, 
        n_estimators=cfg.QRF_N_ESTIMATORS, 
        max_d=cfg.QRF_MAX_DEPTH,
        criterion=cfg.CRITERION
    )

    print("Conformal Predictor is initialized.")
    
    # 3. Fitting Bootstrap Models and compute residuals
    print("Start fitting bootstrap models...")
    results_fit = conformal_predictor.fit_bootstrap_models_online_multistep(
        B=cfg.B,
        batch_size=cfg.BATCH_SIZE,
        EPOCHS=cfg.EPOCHS,
        lr=cfg.LEARNING_RATE,
        path=cfg.WEIGHTS_PATH,
        patience=cfg.PATIENCE,
        valid_mode=cfg.VALID_MODE 
    )

    test_trues = results_fit['test']['y_true'].squeeze(1).numpy()
    test_preds = results_fit['test']['y_pred'].squeeze(1).numpy()
    
    
    # 4.Compute Prediction Intervals
    print("Computing Prediction Intervals...")
    conformal_predictor.compute_Widths_Ensemble_online(
        alpha=cfg.ALPHA,
        smallT=cfg.SMALL_T,
        use_SPCI=cfg.USE_SPCI,
        past_window=cfg.PAST_WINDOW,
        random_state=cfg.SEED
    )
    print("Prediction Intervals Computed.")

    # 5. Results Calculation
    print("Calculating Overall Results...")
    mean_coverage, mean_volume, coverage_seq, volume_seq, radius_seq = conformal_predictor.get_results()
    print("Overall Results Calculated.")

    end_time = time.time()
    total_time = end_time - start_time

    print("-" * 20)
    print(f"Average Coverage: {mean_coverage:.4f}")
    print(f"Average Volume: {mean_volume:.4e}")
    print(f"Running Time: {total_time:.2f} 초")
    print("--- Experiment is finished ---")


    # historcial cov (EWMA 0.94, 0.92), ellipsoid의 cov
    results_dict = {
            'config': {k: str(v) if not isinstance(v, (int, float, bool, type(None), list, dict)) else v for k, v in vars(cfg).items() if not k.startswith('__')}, # 설정값 저장
            'dates': dates['test'].astype(str).tolist(),
            'mean_coverage': mean_coverage,
            'mean_volume': mean_volume,
            'total_time_seconds': total_time,
            'sequence_radius': radius_seq,
            'sequence_coverage': coverage_seq, 
            'sequence_volume': [vol if not np.isnan(vol) else None for vol in volume_seq],
            'historical_cov': cov_list,
            'valid_resid_cov': conformal_predictor.global_cov.tolist(),
            'y_trues': test_trues.tolist(),
            'y_preds': test_preds.tolist()
        }

    return results_dict

# 추가사항
# 1. rolling window
# 2. cp 결과 중간저장 (저장 경로 cpp 랑 동일하게). 저장은 metric에 대해서만, cov나 값들은 저장은하지말고, 반환만.
# 3. 최적화에 투입하여 weights 구하고 평가
# 3.a. formulation 1: max mu_hat^T w - sqrt(q)*||L^T w||_2  s.t. 1^T w = 1, w >= 0, mu_hat in ellipsoid (3.a 우선)
# 3.b. formulation 2: max mu_hat^T w  s.t. mu_hat^T w - sqrt(q)*||L^T w||_2 >= s0, 1^T w = 1, w >= 0, mu_hat in ellipsoid
# => 3.b 의 경우에는 s0 값이 config로 부터 list 형태로 주어짐. 각각의 형태에 대해 최적화 수행
# 4. 포트폴리오 평가 지표 출력 및 저장하는 함수 형태로 마무리. run_ccpo_rolling_backtest 라는 함수로.



if __name__ == "__main__":
    experiment_results = run_single_experiment(config_cp)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_{config_cp.MODEL_CLASS.__name__}_{timestamp}.json"
    results_filepath = os.path.join(config_cp.RESULTS_PATH, results_filename)

    try:
        if 'sequence_coverage' in experiment_results and experiment_results['sequence_coverage'] is not None:
             experiment_results['sequence_coverage'] = [bool(c) for c in experiment_results['sequence_coverage']]
        
        with open(results_filepath, 'w') as f:
            json.dump(experiment_results, f, indent=4)
        print(f"Results are saved at {results_filepath}.")
    except Exception as e:
        print(f"Error: {e}")