# factors/vwap_bias.py
def signal(*args):

    df = args[0]
    n = args[1]
    factor_name = args[2]

    vwap = df['turnover'].rolling(n).sum() / df['volume'].rolling(n).sum()
    df[factor_name] = df['close'] / vwap - 1

    return df