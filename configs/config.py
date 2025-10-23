"""
Unified Configuration for CPP and CCPO
All methods share common settings with method-specific parameters
"""

import torch
import numpy as np
from layers.predictors import MLP, DLinear, LSTMModel

# ============================================================================
# GENERAL SETTINGS
# ============================================================================

SEED = 2025
np.random.seed(SEED)

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_PATH = './data/snp10.csv'
FREQUENCY = 'weekly'  # 'daily', 'weekly', 'monthly'

# ============================================================================
# ROLLING WINDOW CONFIGURATION
# ============================================================================

# K-L-V Split sizes
K = 468  # Training/Optimization samples
L = 312   # Calibration samples
V = 13   # Validation/Test samples

# Rolling window settings
STEP_SIZE = V  # How many points to move forward (default: V)
LOOKBACK = 26   # History window for time series (CCPO only)

# Expanding window settings (optional)
USE_EXPANDING_WINDOW = False  # If True, K grows over time
K_MAX = 300  # Maximum K size for expanding window (None = unlimited)

# ============================================================================
# CHANCE CONSTRAINT CONFIGURATION
# ============================================================================

ALPHA = 0.05  # Miscoverage level (1 - coverage target)

# ============================================================================
# CPP-SPECIFIC CONFIGURATION
# ============================================================================

class CPP:
    """CPP (Chance-constrained Programming with Prediction sets) settings"""
    
    # Methods to run
    METHODS = ['CPP-MIP', 'SAA']  # Options: 'CPP-MIP', 'CPP-KKT', 'SAA'
    
    # Big-M values for MIP encoding
    M = 0.99      # Maximum possible portfolio return
    m = -0.99     # Minimum possible portfolio return
    ZETA = 1e-6   # Threshold value
    
    # Solver settings
    TIME_LIMIT = 360  # seconds
    
    # SAA parameter
    OMEGA = 0.03  # Conservative parameter (~alpha/2)

# ============================================================================
# CCPO-SPECIFIC CONFIGURATION
# ============================================================================

class CCPO:
    """CCPO (Conformal Prediction + Portfolio Optimization) settings"""
    
    # Model Configuration
    MODEL_CLASS = LSTMModel  # Options: DLinear, LSTMModel, MLP
    FORMULATION = 'cco'  # 'cco' or 'target'
    GAMMA = 0.1  # Risk preference factor (higher = more risk-averse)
    
    # Training Configuration
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    PATIENCE = 10
    WEIGHT_DECAY = 0.06
    WEIGHTS_PATH = './weights/'
    VALID_MODE = True
    
    # Conformal Prediction Configuration
    B = 20  # Number of bootstrap models
    USE_SPCI = True  # Use Sequential Predictive Conformal Inference
    SMALL_T = False
    PAST_WINDOW = 100  # Must be < L
    LOW_RANK_R = 8  # Low-rank approximation for covariance
    STRIDE = 1
    USE_LOCAL_ELLIPSOID = False
    
    # Quantile Regression Configuration
    QRF_BINS = 10
    QRF_N_ESTIMATORS = 50
    QRF_MAX_DEPTH = 5
    CRITERION = 'squared_error'  # 'absolute_error', 'squared_error', 'friedman_mse', 'poisson'
    
    # Data preprocessing
    USE_SCALER = True
    FORECAST_HORIZON = 1

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

RESULTS_PATH = './results/'
LOG_FILE_CPP = 'cpp_log.txt'
LOG_FILE_CCPO = 'ccpo_log.txt'
LOG_FILE_ALL = 'combined_log.txt'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_periods_per_year(frequency: str) -> int:
    """Get number of periods per year based on frequency"""
    freq_map = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12
    }
    return freq_map.get(frequency.lower(), 252)

def validate_config():
    """Validate configuration settings"""
    
    # Check PAST_WINDOW
    if CCPO.USE_SPCI and CCPO.PAST_WINDOW >= min(L, V):
        raise ValueError(
            f"CCPO.PAST_WINDOW ({CCPO.PAST_WINDOW}) must be less than min(L, V) = {min(L, V)}"
        )
    
    # Check expanding window settings
    if USE_EXPANDING_WINDOW and K_MAX is not None and K_MAX < K:
        raise ValueError(
            f"K_MAX ({K_MAX}) must be >= initial K ({K})"
        )
    
    print("✅ Configuration validated successfully")

# Validate on import
# validate_config()
