# factors/volatility_friction_ratio.py
# using close to close log return
import pandas as pd
import numpy as np

def signal(*args):
     
    df = args[0]
    n = int(args[1])
    factor_name = args[2]

    log_returns = np.log(df['close']  /df['close'].shift(1).replace(0, 1e-8))
    returns_vol = log_returns.rolling(n).std()

    parkinson_raw = (np.log(df['high'] / df['low'].replace(0, 1e-8)))**2 / (4 * np.log(2))
    parkinson_vol = np.sqrt(parkinson_raw.rolling(n).mean())
    raw_ratio = parkinson_vol / returns_vol.replace(0, 1e-8)
    
    df[factor_name] = raw_ratio.rolling(n).rank(ascending=True, pct=True)
    return df