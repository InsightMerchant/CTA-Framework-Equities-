# config.py
import numpy as np

# ---------------------------------------------------------------------
# 1. GLOBAL SETTINGS & SANDBOX
# ---------------------------------------------------------------------
OPTIMIZATION_START = "2020-01-01"
OPTIMIZATION_END = "2023-12-31"

# ---------------------------------------------------------------------
# 2. EXIT CONFIGURATION
# ---------------------------------------------------------------------
EXIT_CONFIG = {
    'exit_mode': 'conf_cross_zero',  # 'flip', 'conf_cross_zero', 'fixed_bars'
    'exit_bars': 18,                  # only used if exit_mode='fixed_bars'
}

# ---------------------------------------------------------------------
# 3. CONSTRAINED 3D PARAMETER GRIDS (Symmetric — validate hypothesis)
# ---------------------------------------------------------------------
CONSTRAINED_GRID = {
    'DIRECTIONAL': {
        'BIAS_Q': {
            'n': np.arange(10, 301, 10),
            'dir_thresh_symmetric': np.array([0.70, 0.80, 0.90]),
        },
        'VWAP_BIAS': {
            'n': np.arange(10, 301, 10),
            'dir_thresh_symmetric': np.array([0.02, 0.04, 0.06, 0.08, 0.1]),
        },
        'CCI': {
            'n': np.arange(10, 301, 10),
            'dir_thresh_symmetric': np.array([100, 125, 150, 175, 200]),
        },
        'MA_CROSS': {
            'n': np.arange(10, 301, 10),
            'dir_thresh_symmetric': np.array([1.0])
        },
        'ORDER_IMBALANCE': {
            'n': np.arange(10, 301, 10),
            'dir_thresh_symmetric': np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
        },
    },

    'CONFIRMATION': {
        'TURNOVER_MEAN_Z_LESSER': {
            'conf_thresh': np.array([-0.5, -1.0, -1.5, -2.0, -2.5, -3.0]),
            'conf_mode': ['lesser'],
            'exit_threshold': [0],
        },
        'TURNOVER_MEAN_Z_GREATER': {
            'conf_thresh': np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
            'conf_mode': ['greater'],
            'exit_threshold': [0],
        },
        'LIQUIDITY_PREMIUM_Z_LESSER': {
            'conf_thresh': np.array([-1.0, -1.5, -2.0, -2.5, -3.0]),
            'conf_mode': ['lesser'],
            'exit_threshold': [0],
        },
        'LIQUIDITY_PREMIUM_Z_GREATER': {
            'conf_thresh': np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
            'conf_mode': ['greater'],
            'exit_threshold': [0],
        },
        'RETURN_VOLATILITY_LESSER': {
            'conf_thresh': np.array([-0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0]),
            'conf_mode': ['lesser'],
            'exit_threshold': [0],
        },
        'RETURN_VOLATILITY_GREATER': {
            'conf_thresh': np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
            'conf_mode': ['greater'],
            'exit_threshold': [0],
        },
        'PARKINSON_VOLATILITY_LESSER': {
            'conf_thresh': np.array([-0.5, -0.75, -1.0, -1.25, -1.5, -1.75, -2.0]),
            'conf_mode': ['lesser'],
            'exit_threshold': [0],
        },
        'PARKINSON_VOLATILITY_GREATER': {
            'conf_thresh': np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
            'conf_mode': ['greater'],
            'exit_threshold': [0],
        },
        'VOLATILITY_FRICTION_RATIO_GREATER': {
            'conf_thresh': np.array([0.7, 0.75, 0.8, 0.85, 0.9]),
            'conf_mode': ['greater'],
            'exit_threshold': [0.5],
        },
        'VOLATILITY_FRICTION_RATIO_LESSER': {
            'conf_thresh': np.array([0.1, 0.15, 0.2, 0.25, 0.3]),
            'conf_mode': ['lesser'],
            'exit_threshold': [0.5],
        },
    }
}

# ---------------------------------------------------------------------
# 4. UNCONSTRAINED PARAMETER GRIDS (Asymmetric — optimize per side)
#    Use after constrained mode validates the hypothesis.
#    Allows independent long/short thresholds for the directional factor.
# ---------------------------------------------------------------------
UNCONSTRAINED_GRID = {
    'DIRECTIONAL': {
        'BIAS_Q': {
            'n': np.arange(20, 301, 20),
            'dir_thresh_long': np.arange(0.6, 0.91, 0.1),
            'dir_thresh_short': np.arange(0.1, 0.41, 0.1),
        },
        'VWAP_BIAS': {
            'n': np.arange(20, 301, 20),
            'dir_thresh_long': np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]),
            'dir_thresh_short': np.array([-0.10, -0.08, -0.06, -0.05, -0.04, -0.03, -0.02]),
        },
        'CCI': {
            'n': np.arange(20, 301, 20),
            'dir_thresh_long': np.arange(75, 226, 25),
            'dir_thresh_short': np.arange(-225, -74, 25),
        },
        'ORDER_IMBALANCE': {
            'n': np.arange(20, 301, 20),
            'dir_thresh_long': np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0]),
            'dir_thresh_short': np.array([-2.0, -1.5, -1.25, -1.0, -0.75, -0.5]),
        },
        'MA_CROSS': {
            'n': np.arange(20, 301, 20),
            'dir_thresh_long': np.array([1.0]),
            'dir_thresh_short': np.array([-1.0]),
        },
    },

    'CONFIRMATION': {
        'TURNOVER_MEAN_Z_LESSER': {
            'conf_thresh': np.array([-1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5]),
            'conf_mode': ['lesser'],
            'exit_threshold': np.arange(-0.5, 0.01, 0.1),     # exits as conf recovers toward 0
        },
        'TURNOVER_MEAN_Z_GREATER': {
            'conf_thresh': np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]),
            'conf_mode': ['greater'],
            'exit_threshold': np.arange(0.0, 0.51, 0.1),      # exits as conf fades toward 0
        },
        'LIQUIDITY_PREMIUM_Z_LESSER': {
            'conf_thresh': np.array([-1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5]),
            'conf_mode': ['lesser'],
            'exit_threshold': np.arange(-0.5, 0.01, 0.1),     # exits as liq_prem recovers toward 0
        },
        'LIQUIDITY_PREMIUM_Z_GREATER': {
            'conf_thresh': np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]),
            'conf_mode': ['greater'],
            'exit_threshold': np.arange(0.0, 0.51, 0.1),      # exits as liq_prem fades toward 0
        },
        'RETURN_VOLATILITY_LESSER': {
            'conf_thresh': np.array([-1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5]),
            'conf_mode': ['lesser'],
            'exit_threshold': np.arange(-0.5, 0.01, 0.1),     # exits as vol recovers toward 0
        },
        'RETURN_VOLATILITY_GREATER': {
            'conf_thresh': np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]),
            'conf_mode': ['greater'],
            'exit_threshold': np.arange(0.0, 0.51, 0.1),      # exits as vol fades toward 0
        },
        'PARKINSON_VOLATILITY_LESSER': {
            'conf_thresh': np.array([-1.0, -1.25, -1.5, -1.75, -2.0, -2.25, -2.5]),
            'conf_mode': ['lesser'],
            'exit_threshold': np.arange(-0.3, 0.01, 0.1),     # exits as parkinson recovers toward 0
        },
        'PARKINSON_VOLATILITY_GREATER': {
            'conf_thresh': np.array([1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]),
            'conf_mode': ['greater'],
            'exit_threshold': np.arange(0.0, 0.31, 0.1),      # exits as parkinson fades toward 0
        },
        'VOLATILITY_FRICTION_RATIO_LESSER': {
            'conf_thresh': np.array([0.1, 0.2, 0.3, 0.4]),
            'conf_mode': ['lesser'],
            'exit_threshold': np.arange(0.4, 0.61, 0.1),     # midpoint ~0.5, exit above it
        },
        'VOLATILITY_FRICTION_RATIO_GREATER': {
            'conf_thresh': np.array([0.6, 0.7, 0.8, 0.9]),
            'conf_mode': ['greater'],
            'exit_threshold': np.arange(0.4, 0.61, 0.1),     # midpoint ~0.5, exit below it
        },
    }
}
