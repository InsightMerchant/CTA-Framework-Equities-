# factors/parkinson_volatility.py
# instead of close to close, it uses each bar high-low ranges
import numpy as np
import pandas as pd

def signal(*args):

    df = args[0]
    n = int(args[1])
    factor_name = args[2]

    parkinson_vol = (np.log(df['high'] / df['low'].replace(0, 1e-8)))**2 / (4 * np.log(2))
    rolling_mean = parkinson_vol.rolling(n, min_periods=n).mean()
    rolling_std = parkinson_vol.rolling(n, min_periods=n).std()

    df[factor_name] = (parkinson_vol - rolling_mean) / (rolling_std + 1e-8)

    return df