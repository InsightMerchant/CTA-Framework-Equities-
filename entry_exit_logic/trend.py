# entry_exit_logic/trend.py
import pandas as pd
import numpy as np

def trend_logic(df:pd.DataFrame, factor_col:str, long_thresh: float, short_thresh: float) -> pd.Series:
    
    signals = pd.Series(np.nan, index=df.index)
    signals = np.where(df[factor_col] >= long_thresh, 1, signals)
    signals = np.where(df[factor_col] <= short_thresh, -1, signals)
    
    position = pd.Series(signals, index=df.index).ffill().fillna(0)
    
    return position