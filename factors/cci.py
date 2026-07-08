# factors/cci.py
# Commodity Channel Index

def signal(*args):
    df = args[0]
    n = int(args[1])
    factor_name = args[2]

    tp = (df['high'] + df['low'] + df['close']) / 3
    ma = tp.rolling(n, min_periods=n).mean()
    md = abs(tp - ma).rolling(n).mean()

    df[factor_name] = (tp - ma) / (md * 0.015 + 1e-8)

    return df