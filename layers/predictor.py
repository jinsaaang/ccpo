# several types of prediction models: linear regression, ridge regression, random forest regression, etc.

import pandas as pd
import numpy as np
import math
import time as time
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn_quantile import RandomForestQuantileRegressor, SampleRandomForestQuantileRegressor
from numpy.lib.stride_tricks import sliding_window_view
warnings.filterwarnings("ignore")


