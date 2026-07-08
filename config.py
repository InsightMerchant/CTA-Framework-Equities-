# config.py
import numpy as np

# ---------------------------------------------------------------------
# 1. GLOBAL SETTINGS & SANDBOX
# ---------------------------------------------------------------------
OPTIMIZATION_START = "2020-01-01"
OPTIMIZATION_END = "2022-12-31"

# ---------------------------------------------------------------------
# 2. CONSTRAINED 3D PARAMETER GRIDS
# ---------------------------------------------------------------------

EXIT_CONFIG = {
    'exit_mode': 'conf_cross_zero',  # 'flip', 'conf_cross_zero', 'fixed_bars'
    'exit_bars': 4,                   # only used if exit_mode='fixed_bars'
}

CONSTRAINED_GRID = {
    'DIRECTIONAL': {
        'BIAS_Q': {
            'n': np.arange(20, 301, 10),
            'dir_thresh_symmetric': np.array([0.70, 0.80, 0.85, 0.90]),
        },
        'VWAP_BIAS_Q': {
            'n': np.arange(20, 301, 10),
            'dir_thresh_symmetric': np.array([0.02, 0.03, 0.04, 0.05]), 
        },
        'CCI': {
            'n': np.arange(20, 301, 10),
            'dir_thresh_symmetric': np.array([100, 125, 150, 175, 200]), 
        },
        'MA_CROSS': {
            'n': np.arange(20, 201, 10),
            'dir_thresh_symmetric': np.array([1.0]) 
        }
    },
    
    'CONFIRMATION': {
        'TURNOVER_MEAN_Z_LESSER': {
            'conf_thresh': np.array([-1.0, -1.5, -2.0, -2.5, -3.0]),
            'conf_mode': ['lesser']
        },
        'TURNOVER_MEAN_Z_GREATER': {
            'conf_thresh': np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
            'conf_mode': ['greater'] 
        },
        'LIQUIDITY_PREMIUM_Z_LESSER': {
            'conf_thresh': np.array([-1.0, -1.5, -2.0, -2.5, -3.0]),
            'conf_mode': ['lesser']
        },
        'LIQUIDITY_PREMIUM_Z_GREATER': {
            'conf_thresh': np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
            'conf_mode': ['greater'] 
        }
    }
}

# ---------------------------------------------------------------------
# 3. UNCONSTRAINED PARAMETER GRIDS (5D Data Mining Mode)
# ---------------------------------------------------------------------
UNCONSTRAINED_GRID = {
    'DIRECTIONAL': {
        'BIAS_Q': {
            'n1': np.arange(20, 401, 40),
            'dir_thresh_long': np.arange(0.6, 1.01, 0.1),
            'dir_thresh_short': np.arange(0.0, 0.41, 0.1),
        },
        'VWAP_BIAS_Q': {
            'n1': np.arange(20, 401, 40),
            'dir_thresh_long': np.arange(0.01, 0.05, 0.01),
            'dir_thresh_short': np.arange(-0.05, -0.01, 0.01),
        },
        'CCI': {
            'n1': np.arange(20, 401, 40),
            'dir_thresh_long': np.arange(50, 201, 50),
            'dir_thresh_short': np.arange(-200, -49, 50),
        },
    },
    
    'CONFIRMATION': {
        'TURNOVER_MEAN_Z_LESSER': {
            'n2': np.arange(20, 401, 40),
            'conf_thresh': np.arange(-3.0, -0.51, 0.5),
            'conf_mode': ['lesser']
        }
    }
}