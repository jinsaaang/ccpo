import torch
from layers.predictors import LSTMModel, DLinear, MLP


# ==============================================================================
# EVALUATION SETTINGS
# ==============================================================================

# ---- Evaluation Mode ----
EVALUATION_MODE = "rolling"     # ["direct", "rolling"]
                                # "direct": Single split evaluation (Train/K/V)
                                # "rolling": Rolling window evaluation (recommended)

# ---- Data Split Mode ----
MODE = "counts"                 # ["dates", "counts"]
                                # "dates": Split by specific dates
                                # "counts": Split by sequence counts (recommended)

# ==============================================================================
# DATA SETTINGS
# ==============================================================================

DATA_PATH = "./data"
FREQUENCY = "weekly"            # ["daily", "weekly", "monthly"]
LOOKBACK = 52                   # Lookback window for model input
NUM_ASSETS = 10                 # Number of assets [5, 10, 30, 49]
ALPHA = 0.05                    # Miscoverage rate (1-ALPHA = coverage level)
SEED = 2025
BATCH_SIZE = 32 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ==============================================================================
# DIRECT MODE SETTINGS (EVALUATION_MODE="direct" only)
# ==============================================================================

# ---- 'dates' Mode: Split by end dates ----
TRAIN_END_DATES = "2015-12-31"  # Model Train End
VALID_END_DATES = "2020-12-31"  # Calibration (K) End
TEST_END_DATES  = "2023-12-31"  # Test (V) End

# ---- 'counts' Mode: Split by sequence counts ----
TRAIN_LENGTH = 780              # Number of Model Train Sequences
LEN_K = 520                     # Number of Calibration (K) Sequences
LEN_V = 260                     # Number of Test (V) Sequences


# ==============================================================================
# ROLLING MODE SETTINGS (EVALUATION_MODE="rolling" only)
# ==============================================================================

class ROLLING:
    WINDOW_TYPE = "expanding"   # ["sliding", "expanding"]
                                # "sliding": Fixed window size, moves forward
                                # "expanding": Growing window, train_start fixed
    
    # --- 'counts' Mode Rolling Configuration (Recommended) ---
    class COUNTS:
        TRAIN_LEN = 52 * 15                 # Initial train length = 15 years = 780 weeks
        TEST_LEN = int(52 * 5)              # Test length per window = 5 years = 260 weeks
        STEP_SIZE = int(52 * 5)             # Retrain cycle = 5 years = 260 weeks
        
        # Expanding Window Example (40 years of data):
        # Window 1: Train [0Y~15Y],  Test [15Y~20Y] → 5Y evaluation
        # Window 2: Train [0Y~20Y],  Test [20Y~25Y] → 5Y evaluation (retrain every 5Y)
        # Window 3: Train [0Y~25Y],  Test [25Y~30Y] → 5Y evaluation
        # Window 4: Train [0Y~30Y],  Test [30Y~35Y] → 5Y evaluation
        # Window 5: Train [0Y~35Y],  Test [35Y~40Y] → 5Y evaluation
        # Total evaluation period: 25 years (15Y~40Y)
    
    # --- 'dates' Mode Rolling Configuration ---
    class DATES:
        MODEL_TRAIN_OFFSET = "10Y"      # Train period length
        K_PERIOD_OFFSET = "5Y"          # Calibration (K) period length
        V_PERIOD_OFFSET = "1Y"          # Test (V) period length
        STEP_OFFSET = "1Y"              # Rolling step (should match V_PERIOD_OFFSET)
        
        ROLLING_START_DATE = "2000-01-01"   # First training period start
        ROLLING_END_DATE = None             # Last test period end (None = use all data)


# ==============================================================================
# CPP (Chance-constrained Portfolio) SETTINGS
# ==============================================================================

class CPP:
    METHODS = ['CPP-MIP', 'SAA']    # CPP methods to evaluate
    OMEGA = 0.03                    # SAA regularization parameter
    TIME_LIMIT = 60                 # Solver time limit (seconds)
    M = 0.99                        # Upper bound for portfolio weights
    m = -M                          # Lower bound (allow short if negative)
    zeta = 1e-6                     # Numerical stability parameter

# ==============================================================================
# CCPO (Conformal Prediction + Portfolio Optimization) SETTINGS
# ==============================================================================

class CCPO:
    # ---- Model Settings ----
    MODEL_CLASS = LSTMModel         # [LSTMModel, DLinear, MLP]
    B = 2                           # Number of Bootstrap models
    BATCH_SIZE = 32                 # Training batch size
    EPOCHS = 30                     # Training epochs
    LEARNING_RATE = 1e-3            # Learning rate
    PATIENCE = 10                   # Early stopping patience
    WEIGHTS_PATH = "./weights/ccpo" # Model weights save path
    
    # ---- Conformal Prediction Settings ----
    LOW_RANK_R = 8                  # Low-rank approximation for covariance matrix
    USE_LOCAL_ELLIPSOID = False     # Use local (KNN-based) covariance vs global
    USE_SPCI = True                 # Use SPCI (quantile regression) for intervals
    PAST_WINDOW = 52                # Max past residuals for quantile computation
    
    # QRF (Quantile Random Forest) parameters (when USE_SPCI=True)
    QRF_BINS = 10                   # Number of bins for binning
    QRF_N_ESTIMATORS = 50           # Number of trees
    QRF_MAX_DEPTH = 5               # Max tree depth
    CRITERION = "squared_error"     # Split criterion
    
    # ---- Portfolio Optimization Settings ----
    GAMMA = 1.0                     # Risk preference (higher = more risk-averse)
    FORMULATION = "cco"             # Portfolio formulation ["cco", "target"]
    
    # ---- Multi-step Prediction Settings ----
    USE_MULTISTEP = True            # Use multi-step prediction (daily input -> multi-horizon)
    HORIZON_MAP = {                 # Prediction horizon for each frequency
        "daily": 1,                 # Daily: predict t+1
        "weekly": 5,                # Weekly: predict t+1~t+5, then aggregate
        "monthly": 20               # Monthly: predict t+1~t+20, then aggregate
    }
    AGGREGATION_METHOD = "mean"     # How to aggregate multi-step predictions ["mean", "last"]
                                    # "mean": Average across horizon (robust, recommended)
                                    # "last": Use only last horizon step
    
    # ---- Training Settings ----
    LOSS_AGGREGATION = "mean"       # How to compute loss for multi-step ["mean", "last"]
                                    # "mean": MSE across all horizon steps (stable training)
                                    # "last": MSE only on last horizon step
    
    # ---- Covariance Update Settings ----
    USE_COV_UPDATE = True           # Update covariance with past test residuals (rolling windows)
                                    # True: Accumulate residuals -> better estimation (recommended for expanding)
                                    # False: Use only current train residuals
    
    # ---- Conformal Residual Mode Settings ----
    CP_RESIDUAL_MODE = "aggregated" # How to use residuals for conformal prediction ["all_horizon", "aggregated"]
                                    # "all_horizon": Use all horizon steps as individual samples
                                    #   - Pros: More samples (N*horizon), tighter intervals
                                    #   - Cons: Violates exchangeability, inconsistent with optimization target
                                    #   - Use case: Maximum sample size, exploratory analysis
                                    # "aggregated": Aggregate residuals same as AGGREGATION_METHOD
                                    #   - Pros: Theoretically valid, consistent with optimization, faster
                                    #   - Cons: Fewer samples (N only)
                                    #   - Use case: Production, theoretical soundness (RECOMMENDED)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def get_periods_per_year(freq: str) -> int:
    """Get number of periods per year for a given frequency."""
    f = (freq or "").lower()
    if f in ("w", "week", "weekly"): return 52
    if f in ("d", "day", "daily"): return 252
    if f in ("m", "month", "monthly"): return 12
    return 52