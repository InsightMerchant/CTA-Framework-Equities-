# engine/strategy_combinator.py
"""
Generates all valid (directional_factor × confirmation_factor × logic_type) 
combinations from config, without manual code changes.
"""
import pandas as pd
from factors import compute_factor
from entry_exit_logic.two_factor_trend import two_factor_trend
from entry_exit_logic.two_factor_reverse import two_factor_reverse

LOGIC_REGISTRY = {
    'trend': two_factor_trend,
    'reverse': two_factor_reverse,
}

def build_strategy(df: pd.DataFrame, strategy_spec: dict) -> pd.DataFrame:
    """
    Universal strategy builder. Takes a spec dict and returns df with 'position' column.
    
    strategy_spec = {
        'dir_factor': 'bias_q',
        'conf_factor': 'liquidity_premium_z',
        'logic': 'reverse',
        'n': 60,
        'dir_thresh_long': 0.85,
        'dir_thresh_short': 0.15,
        'conf_thresh': -1.5,
        'conf_mode': 'lesser'
    }
    """
    work_df = df.copy()
    
    n = int(strategy_spec['n'])
    dir_factor_name = strategy_spec['dir_factor']
    conf_factor_name = strategy_spec['conf_factor']
    
    # Compute factors dynamically
    dir_col = f"{dir_factor_name}_{n}"
    conf_col = f"{conf_factor_name}_{n}"
    
    work_df = compute_factor(work_df, dir_factor_name, n, dir_col)
    work_df = compute_factor(work_df, conf_factor_name, n, conf_col)
    
    # Apply entry/exit logic
    logic_func = LOGIC_REGISTRY[strategy_spec['logic']]
    work_df['position'] = logic_func(
        df=work_df,
        dir_factor=dir_col,
        dir_long_thresh=strategy_spec['dir_thresh_long'],
        dir_short_thresh=strategy_spec['dir_thresh_short'],
        conf_factor=conf_col,
        conf_thresh=strategy_spec['conf_thresh'],
        conf_mode=strategy_spec['conf_mode']
    )
    
    return work_df
