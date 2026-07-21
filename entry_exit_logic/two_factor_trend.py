# entry_exit_logic/two_factor_trend.py

import numpy as np
import pandas as pd


def two_factor_trend(
        df: pd.DataFrame,
        dir_factor: str, dir_long_thresh: float, dir_short_thresh: float,
        conf_factor: str, conf_thresh: float, conf_mode: str = 'greater',
        exit_mode: str = 'conf_cross_zero',
        exit_bars: int = 4,
        exit_threshold: float = 0,
) -> pd.Series:
    """
    Trend-following two-factor logic with configurable exit.

    Entry:
        Long  when dir_factor >= dir_long_thresh AND conf condition met
        Short when dir_factor <= dir_short_thresh AND conf condition met

    Exit modes:
        'flip'            : Hold until opposite signal (ffill)
        'conf_cross_zero' : Exit when confirmation factor crosses back to zero
        'fixed_bars'      : Exit after N bars, only enter when flat
    """

    # 1. Directional Conditions
    long_dir_cond = df[dir_factor] >= dir_long_thresh
    short_dir_cond = df[dir_factor] <= dir_short_thresh

    # 2. Confirmation Conditions
    if conf_mode == 'greater':
        conf_cond = df[conf_factor] >= conf_thresh
    elif conf_mode == 'lesser':
        conf_cond = df[conf_factor] <= conf_thresh
    else:
        raise ValueError("conf_mode must be 'greater' or 'lesser'")

    # 3. Combined entry signals (trend-following: go WITH direction)
    long_signal = long_dir_cond & conf_cond
    short_signal = short_dir_cond & conf_cond

    long_arr = long_signal.values
    short_arr = short_signal.values
    conf_values = df[conf_factor].values
    n_rows = len(df)
    position = np.zeros(n_rows)

    # 4. Build position array based on exit mode
    if exit_mode == 'flip':
        signals = np.fullI(n_rows, np.nan)
        signals[long_arr] = long_signal.values
        signals[short_arr] = short_signal.values
        return pd.Series(signals, index=df.index).ffill().fillna()

    elif exit_mode == 'conf_cross_zero':
        current_pos = 0
        for i in range(n_rows):
            # Only enter when flat
            if current_pos == 0:
                if long_arr[i]:
                    current_pos = 1
                elif short_arr[i]:
                    current_pos = -1

            # Exit when conf crosses zero
            if current_pos != 0:
                if conf_mode == 'greater' and conf_values[i] <= exit_threshold:
                    current_pos = 0
                elif conf_mode == 'lesser' and conf_values[i] >= exit_threshold:
                    current_pos = 0

            position[i] = current_pos

    elif exit_mode == 'fixed_bars':
        bars_in_trade = 0
        current_pos = 0

        for i in range(n_rows):
            # Only enter when flat
            if current_pos == 0:
                if long_arr[i]:
                    current_pos = 1   # Trend: go long
                    bars_in_trade = 0
                elif short_arr[i]:
                    current_pos = -1  # Trend: go short
                    bars_in_trade = 0

            # Count bars and exit after N
            if current_pos != 0:
                bars_in_trade += 1
                if bars_in_trade > exit_bars:
                    current_pos = 0
                    bars_in_trade = 0

            position[i] = current_pos

    else:
        raise ValueError("exit_mode must be 'flip', 'conf_cross_zero', or 'fixed_bars'")

    return pd.Series(position, index=df.index)
