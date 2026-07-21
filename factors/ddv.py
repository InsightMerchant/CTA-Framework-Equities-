# factors/ddv.py
# dealer distress vector
# Microstructural Edge: Quantifies the velocity of directional price displacement amplified by relative turnover intensity to isolate active dealer inventory flushes.

import numpy as np

def signal(*args):
    df = args[0]
    n = int(args[1])
    factor_name = args[2]

    displacement = df['close'] - df['open']
    total_range = df['high'] - df['low']
    price_efficiency = displacement / (total_range + 1e-8)

    rolling_turnover_mean = df['turnover'].rolling(n).mean()
    turnover_intensity = df['turnover'] / (rolling_turnover_mean + 1e-8)

    ddv = price_efficiency * turnover_intensity

    ddv_mean = ddv.rolling(n).mean()
    ddv_std = ddv.rolling(n).std()

    df[factor_name] = (ddv - ddv_mean) / (ddv_std + 1e-8)
    return df