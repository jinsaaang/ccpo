import torch
from layers.predictors import LSTMModel

# ---- 기본 전역 설정 ----
MODE = "counts"                  # "dates" | "counts"
DATA_PATH = "./data"          # (로더가 base_path를 사용할 경우, ./data 아래에 위치)
FREQUENCY = "weekly"
LOOKBACK = 52
NUM_ASSETS = 10
ALPHA = 0.05
SEED = 2025
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available else 'cpu'         

# ---- dates 모드 전용 ----
TRAIN_END_DATES = None           # 예: "2015-12-31"
VALID_END_DATES = None           # 예: "2020-12-31"
TEST_END_DATES  = None           # (선택)

# ---- counts 모드 전용 ----
TRAIN_LENGTH = 780               # 총 학습 길이(예시: 15년 주간)
LEN_K = 520                      # K 길이 (~10y)
LEN_V = 260                      # V 길이 (~5y)

# ---- CPP 설정 ----
class CPP:
    METHODS = ['CPP-MIP', 'SAA']            # 예: ['CPP-MIP', 'SAA']
    OMEGA = 0.03                 # SAA 샘플 수 등 메서드별 의미에 맞게 사용
    TIME_LIMIT = 60             # 시간 제한
    M = 0.99
    m = -M
    zeta = 1e-6

# ---- CCPO 설정 ----
class CCPO:
    # 모델 클래스는 실제 구현에 맞춰 import path 내 클래스 전달 필요
    # 예: from models.lstm import LSTMModel; MODEL_CLASS = LSTMModel
    MODEL_CLASS = LSTMModel

    LOW_RANK_R = 8
    USE_LOCAL_ELLIPSOID = False

    B = 5                         # 부트스트랩 개수
    BATCH_SIZE = 128
    EPOCHS = 20
    LEARNING_RATE = 1e-3
    WEIGHTS_PATH = "./weights/ccpo"  # 체크포인트 저장 경로
    PATIENCE = 5

    USE_SPCI = True
    PAST_WINDOW = 26              # 캘리브레이션에 사용할 과거창(예시)

    GAMMA = 1.0                   # SOCP 내 정규화/리스크 계수
    FORMULATION = "cco"


# ---- 유틸 함수 ----
def get_periods_per_year(freq: str) -> int:
    """주기 문자열에 따른 연간 기간 수 반환"""
    f = (freq or "").lower()
    if f in ("w", "week", "weekly"):
        return 52
    if f in ("d", "day", "daily"):
        return 252
    if f in ("m", "month", "monthly"):
        return 12
    # 기본값: 주간
    return 52
