# factors/ma_cross.py
# MA crossover: fast > slow = +1, fast < slow = -1 (conditional)
# Output: {-1, 0, 1}. No threshold needed.

import numpy as np

def signal(*args):
    df = args[0]
    n = int(args[1])
    factor_name = args[2]

    fast = df['close'].rolling(n).mean()
    slow = df['close'].rolling(2 * n).mean()
    diff = fast - slow
    df[factor_name] = np.sign(diff)

    return df