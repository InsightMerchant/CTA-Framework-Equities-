# factors/order_imbalance.py

import numpy as np
import pandas as pd

def signal(*args):

    df = args[0]
    n = args[1]
    factor_name = args[2]

    denom = (df['high'] - df['low']).replace(0, 1e-8)
    buy_vol = df['turnover'] * ((df['close'] - df['low'])/ denom)
    sell_vol = df['turnover'] * ((df['high'] - df['close'])/ denom)
    
    net_vol = buy_vol - sell_vol
    mean = net_vol.rolling(n).mean()
    std = net_vol.rolling(n).std()
    df[factor_name] = (net_vol - mean) / (std + 1e-8)

    return df