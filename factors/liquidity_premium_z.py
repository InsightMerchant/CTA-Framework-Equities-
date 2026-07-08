# factors/liquidity_premium_z.py
# Rolling z of liquidity premium

def signal(*args):
    df = args[0]
    n = int(args[1])
    factor_name = args[2]

    route_1 = 2 * (df['high'] - df['low']) + (df['open'] - df['close'])
    route_2 = 2 * (df['high'] - df['low']) + (df['close'] - df['open'])

    shortest = route_2.where(route_1 > route_2, route_1)
    norm_shortest_path = (shortest / df['open']).clip(lower=1e-8)
    liquidity_premium = df['turnover'] / norm_shortest_path
    
    raw = liquidity_premium.rolling(n).mean()

    roll_mean = raw.rolling(n).mean()
    roll_std = raw.rolling(n).std()
    df[factor_name] = (raw - roll_mean) / (roll_std + 1e-8)

    return df