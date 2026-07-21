# factors/liquidity_premium_z.py
# Rolling z of liquidity premium
import numpy as np

def signal(*args):
    df = args[0]
    n = int(args[1])
    factor_name = args[2]

    route_1 = 2 * (df['high'] - df['low']) + (df['open'] - df['close'])
    route_2 = 2 * (df['high'] - df['low']) + (df['close'] - df['open'])

    shortest = np.minimum(route_1, route_2)
    norm_shortest_path = (shortest / df['open']).clip(lower=1e-8)
    liquidity_premium = df['turnover'] / norm_shortest_path

    roll_mean = liquidity_premium.rolling(n).mean()
    roll_std = liquidity_premium.rolling(n).std()
    df[factor_name] = (liquidity_premium - roll_mean) / (roll_std + 1e-8)

    return df