# factors/ars.py
# asymmetric shadow ratio
"""
    Calculates the Asymmetric Shadow Ratio (ASR) and its rolling Z-score.
    
    Microstructural Edge: Tracks spatial imbalances between the upper wick 
    (supply wall) and lower shadow (demand defense) to catch institutional absorption.
"""
import numpy as np

def signal(*args):
    df = args[0]
    n = int(args[1])
    factor_name = args[2]

    upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
    lower_shadow = np.minimum(df['open'], df['close']) - df['low']
    total_range = df['high'] - df['low']

    asr = (upper_shadow - lower_shadow) / (total_range - 1e-8)
    asr_mean = asr.rolling(n).mean()
    asr_std = asr.rolling(n).std()

    df[factor_name] = (asr - asr_mean) / (asr_std + 1e-8)
    return df