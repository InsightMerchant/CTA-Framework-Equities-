# factors/returns_volatility.py
# using close to close log return
import pandas as pd
import numpy as np

def signal(*args):
     
    df = args[0]
    n = int(args[1])
    factor_name = args[2]

    log_returns = np.log(df['close']  /df['close'].shift(1).replace(0, 1e-8))
    abs_returns = np.abs(log_returns)
    rolling_mean = abs_returns.rolling(n).mean()
    rolling_std = abs_returns.rolling(n).std()

    df[factor_name] = (abs_returns - rolling_mean) / (rolling_std + 1e-8)

    return df


