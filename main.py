# ccpo/main.py

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# --- 중요 ---
# main.py가 있는 'ccpo' 폴더를 파이썬 경로에 추가합니다.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 설정 및 시드 고정
from configs import config_revised as config
from layers.cp_utils import set_seed

# --- 수정된 import 경로 ---
from evaluation.run_direct_evaluation import run_direct_evaluation
from evaluation.run_rolling_evaluation import run_rolling_evaluation

if __name__ == "__main__":
    set_seed(config.SEED)

    # config.EVALUATION_MODE에 따라 분기
    if config.EVALUATION_MODE == "direct":
        print(f"Running Direct (Single Split) Evaluation (MODE={config.MODE})...")
        results = run_direct_evaluation(
            data_path=config.DATA_PATH,
            frequency=config.FREQUENCY,
            lookback=config.LOOKBACK,
            alpha=config.ALPHA,
            cfg=config
        )
    elif config.EVALUATION_MODE == "rolling":
        print(f"Running Rolling Window Evaluation (MODE={config.MODE}, TYPE={config.ROLLING.WINDOW_TYPE})...")
        results = run_rolling_evaluation(
            data_path=config.DATA_PATH,
            frequency=config.FREQUENCY,
            lookback=config.LOOKBACK,
            alpha=config.ALPHA,
            cfg=config
        )
    else:
        raise ValueError(
            f"Unknown EVALUATION_MODE in config: {config.EVALUATION_MODE}. "
            f"Use 'direct' or 'rolling'."
        )

    print("\n✅ Evaluation completed!")