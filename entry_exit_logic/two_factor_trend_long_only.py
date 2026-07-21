# entry_exit_logic/two_factor_trend_long_only.py
"""
Long-only trend: only takes the LONG side (follow breakout up).
When dir_factor >= dir_long_thresh AND conf condition met → go LONG.
Never shorts.
"""
import numpy as np
import pandas as pd


def two_factor_trend_long_only(
        df: pd.DataFrame,
        dir_factor: str, dir_long_thresh: float, dir_short_thresh: float,
        conf_factor: str, conf_thresh: float, conf_mode: str = 'greater',
        exit_mode: str = 'conf_cross_zero',
        exit_bars: int = 4,
        exit_threshold: float = 0,
) -> pd.Series:

    long_dir_cond = df[dir_factor] >= dir_long_thresh

    if conf_mode == 'greater':
        conf_cond = df[conf_factor] >= conf_thresh
    elif conf_mode == 'lesser':
        conf_cond = df[conf_factor] <= conf_thresh
    else:
        raise ValueError("conf_mode must be 'greater' or 'lesser'")

    long_entry = long_dir_cond & conf_cond

    long_arr = long_entry.values
    conf_values = df[conf_factor].values
    n_rows = len(df)
    position = np.zeros(n_rows)

    if exit_mode == 'flip':
        signals = np.full(n_rows, np.nan)
        signals[long_arr] = 1
        return pd.Series(signals, index=df.index).ffill().fillna(0)

    elif exit_mode == 'conf_cross_zero':
        current_pos = 0
        for i in range(n_rows):
            if current_pos == 0:
                if long_arr[i]:
                    current_pos = 1

            if current_pos != 0:
                if conf_mode == 'greater' and conf_values[i] <= exit_threshold:
                    current_pos = 0
                elif conf_mode == 'lesser' and conf_values[i] >= exit_threshold:
                    current_pos = 0

            position[i] = current_pos

    elif exit_mode == 'fixed_bars':
        current_pos = 0
        bars_in_trade = 0
        for i in range(n_rows):
            if current_pos == 0:
                if long_arr[i]:
                    current_pos = 1
                    bars_in_trade = 0

            if current_pos != 0:
                bars_in_trade += 1
                if bars_in_trade > exit_bars:
                    current_pos = 0
                    bars_in_trade = 0

            position[i] = current_pos

    else:
        raise ValueError("exit_mode must be 'flip', 'conf_cross_zero', or 'fixed_bars'")

    return pd.Series(position, index=df.index)