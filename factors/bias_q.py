# factors/bias_q.py
# Percentile rank of bias (close/MA - 1)
# Output: 0 to 1 (bounded)
# Done

def signal(*args):
    df = args[0]
    n = int(args[1])
    factor_name = args[2]

    bias = df['close'] / df['close'].rolling(n).mean() - 1
    df[factor_name] = bias.rolling(n).rank(ascending=True, pct=True)

    return df