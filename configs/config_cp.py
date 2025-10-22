
import torch
from layers.predictors import MLP, DLinear, LSTMModel

# --- Seed Configuration ---
SEED = 2025

# --- Data Configuration ---
DATA_PATH = './data/snp50.csv'
FREQUENCY = 'weekly' # ('daily', 'weekly', 'monthly')
LOOKBACK = 26
FORECAST_HORIZON = 1

# K-L-V Split (for rolling backtest)
K = 104  # Training samples
L = 52  # Calibration samples
V = 26  # Validation(Test) samples 

# Legacy date-based split (for compatibility)
# TRAIN_END_DATE = '2012-12-31'
# VAL_END_DATE = '2018-12-31'
# TEST_END_DATE = '2024-12-31'

USE_SCALER = True
WEIGHT_DECAY = 0.06

# --- Model Configuration ---
MODEL_CLASS = LSTMModel
FORMULATION = 'cco'  # 'cco' or 'target'

# --- Training Configuration ---
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4
PATIENCE = 10 
WEIGHTS_PATH = './weights/'
VALID_MODE = True

# --- Conformal Prediction Configuration ---
B = 3  # Number of bootstrap models
ALPHA = 0.05        # Miscoverage level
USE_SPCI = True
SMALL_T = False
PAST_WINDOW = 20
LOW_RANK_R = 8  # Low-rank approximation for covariance matrix
STRIDE = 1
USE_LOCAL_ELLIPSOID = False

# --- Quantile Regression Configuration ---
QRF_BINS = 10
QRF_N_ESTIMATORS = 50
QRF_MAX_DEPTH = 5
CRITERION = 'squared_error'  # {'absolute_error', 'squared_error', 'friedman_mse', 'poisson'}

# --- Device Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Save Path Configuration ---
RESULTS_PATH = './results/'