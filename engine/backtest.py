# engine/backtest.py

import numpy as np
import pandas as pd

class Backtest:

    def __init__(self, df: pd.DataFrame, position_col: str, delay_periods: int = 0):
        self.df = df.copy()
        self.position_col = position_col
        self.delay_periods = delay_periods

        # Calculate BOTH Point Differences and Percentage Changes dynamically
        self.df['close_diff'] = self.df['close'].diff()
        self.df['change_rate'] = self.df['close'].pct_change()

    def run(self, fee_rate: float = 0.0002):
        """
        Executes vectorized backtest math with accurate trade timelines,
        and applies transaction friction on turnover shifts to eliminate
        look-ahead fill errors.
        """
        # 1. Shift position array to represent execution delay (prevent lookahead bias)
        total_shift = 1 + self.delay_periods
        self.df['executable_position'] = self.df[self.position_col].shift(total_shift).fillna(0)

        # 2. Track when position changes size or flips directions (Turnover units)
        # e.g., 0 to 1 = 1 unit; 1 to -1 = 2 units (closing long, opening short)
        self.df['position_change'] = self.df['executable_position'].diff().abs().fillna(0)

        # ==========================================
        # SECTION A: Percentage Returns & Compounding
        # ==========================================
        self.df['gross_strategy_returns'] = self.df['executable_position'] * self.df['change_rate']
        self.df['percentage_fees'] = self.df['position_change'] * fee_rate
        
        # Net returns are gross returns minus transaction friction
        self.df['strategy_returns'] = self.df['gross_strategy_returns'] - self.df['percentage_fees']
        self.df['cum_returns'] = (1 + self.df['strategy_returns']).cumprod()

        # ==========================================
        # SECTION B: Absolute Point PnL Matrix
        # ==========================================
        self.df['gross_point_pnl'] = self.df['executable_position'] * self.df['close_diff']
        # Point fees approximate the cash friction based on asset price at execution
        self.df['point_fees'] = self.df['position_change'] * (fee_rate * self.df['close'].shift(1))
        
        self.df['point_pnl'] = self.df['gross_point_pnl'] - self.df['point_fees']
        self.df['cum_point_pnl'] = self.df['point_pnl'].cumsum()

        # Just return the enriched dataframe. PerformanceMetrics class handles KPI outputs!
        return self.df