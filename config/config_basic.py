# Configuration for CPP optimization

import numpy as np

# Big-M values for MIP encoding
M = 0.99    # Maximum possible portfolio return (%)
m = -M   # Minimum possible portfolio return (%)
zeta = 1e-6   # Threshold value

# Solver settings
time_limit = 60  # seconds
config_seed = 42

# CPP Experiment settings (K-L-V rolling splits)
methods = ['CPP-MIP']  # Methods to evaluate: 'CPP-MIP', 'CPP-KKT', 'SAA'
K = 63         # Optimization sample size (training data for solver)
L = 42         # Calibration sample size (for conformal prediction)
V = 42         # Validation sample size (for coverage testing)
alpha = 0.05

# Portfolio settings
freq = 'daily'

# SAA parameter
omega = 0.03  # Conservative parameter for SAA (0.01~0.05)

# Random seed
np.random.seed(config_seed)
