# Configuration for CPP optimization

import numpy as np

# Big-M values for MIP encoding
M = 100.0    # Maximum possible portfolio return (%)
m = -100.0   # Minimum possible portfolio return (%)
zeta = 0.0   # Threshold value

# Solver settings
time_limit = 3600  # 1 hour
config_seed = 42

# Random seed
np.random.seed(config_seed)
