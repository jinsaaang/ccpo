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
methods = ['CPP-MIP', 'SAA']  # Methods to evaluate: 'CPP-MIP', 'CPP-KKT', 'SAA'
alpha = 0.05

# Portfolio settings
# K = 52         # Optimization sample size (training data for solver)
# L = 52         # Calibration sample size (for conformal prediction)
# V = 1          # Validation sample size (for coverage testing)
# freq = 'weekly'

K = 126         # Optimization sample size (training data for solver)
L = 126         # Calibration sample size (for conformal prediction)
V = 63          # Validation sample size (for coverage testing)
freq = 'daily'

# SAA parameter
omega = 0.03  # Conservative parameter for SAA (~alpha/2)

# Random seed
np.random.seed(config_seed)
