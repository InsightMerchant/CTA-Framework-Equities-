# entry_exit_logic/two_factor_reverse.py

import numpy as np
import pandas as pd

def two_factor_reverse(
        df: pd.DataFrame,
        dir_factor: str, dir_long_thresh: float, dir_short_thresh: float,
        conf_factor: str, conf_thresh: float, conf_mode: str = 'lesser',
        exit_mode: str = 'conf_cross_zero',  # NEW: 'flip', 'conf_cross_zero', 'fixed_bars'
        exit_bars: int = 4,                   # NEW: only used if exit_mode='fixed_bars'
) -> pd.Series:
    """
    Mean-reversion two-factor logic with configurable exit.
    
    Exit modes:
        'flip'            : Original behavior — hold until opposite signal (ffill)
        'conf_cross_zero' : Exit when confirmation factor crosses back to zero
        'fixed_bars'      : Exit after N bars regardless
    """
    
    # 1. Entry conditions (same as before)
    long_dir_cond = df[dir_factor] >= dir_long_thresh
    short_dir_cond = df[dir_factor] <= dir_short_thresh

    if conf_mode == "greater":
        conf_cond = df[conf_factor] >= conf_thresh
    elif conf_mode == "lesser":
        conf_cond = df[conf_factor] <= conf_thresh
    else:
        raise ValueError("conf_mode must be 'greater' or 'lesser'")

    long_signal_trigger = long_dir_cond & conf_cond   # → go SHORT (fade the high)
    short_signal_trigger = short_dir_cond & conf_cond  # → go LONG (buy the low)

    # 2. Build position array based on exit mode
    if exit_mode == 'flip':
        # Original: forward-fill until opposite signal
        signals = pd.Series(np.nan, index=df.index)
        signals = np.where(long_signal_trigger, -1, signals)
        signals = np.where(short_signal_trigger, 1, signals)
        position = pd.Series(signals, index=df.index).ffill().fillna(0)

    elif exit_mode == 'conf_cross_zero':
        # Exit when confirmation factor crosses back through zero
        position = np.zeros(len(df))
        current_pos = 0
        conf_values = df[conf_factor].values

        for i in range(len(df)):
            # Check for new entry signals
            if current_pos == 0:
                if long_signal_trigger.iloc[i]:
                    current_pos = -1
                elif short_signal_trigger.iloc[i]:
                    current_pos = 1 

            # Check when conf_threshold cross 0
            if current_pos != 0:
                if conf_mode == 'lesser' and conf_values[i] >= 0:
                    current_pos = 0
                elif conf_mode == 'greater' and conf_values[i] <= 0:
                    current_pos = 0

            position[i] = current_pos

        position = pd.Series(position, index=df.index)

    elif exit_mode == 'fixed_bars':
        # Exit after N bars
        position = np.zeros(len(df))
        bars_in_trade = 0
        current_pos = 0

        for i in range(len(df)):
            # Check for new entry signals
            if current_pos == 0:
                if long_signal_trigger.iloc[i]:
                    current_pos = -1
                    bars_in_trade = 0
                elif short_signal_trigger.iloc[i]:
                    current_pos = 1
                    bars_in_trade = 0

            if current_pos != 0:
                # Count bars and exit after N interval
                if current_pos != 0:
                    bars_in_trade += 1
                    if bars_in_trade > exit_bars:
                        current_pos = 0
                        bars_in_trade = 0

            position[i] = current_pos

        position = pd.Series(position, index=df.index)

    else:
        raise ValueError("exit_mode must be 'flip', 'conf_cross_zero', or 'fixed_bars'")

    return position