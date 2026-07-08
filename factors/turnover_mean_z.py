# factors/turnover_mean_z.py
# Rolling mean of turnover

def signal(*args):
    df = args[0]
    n = int(args[1])
    factor_name = args[2]

    raw = df['turnover']
    ma = df['turnover'].rolling(n, min_periods=n).mean()
    std = df['turnover'].rolling(n, min_periods=n).std()
    
    df[factor_name] = (raw - ma) / (std + 1e-8)

    return df