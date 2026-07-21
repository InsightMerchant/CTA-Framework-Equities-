# engine/backtest.py

import numpy as np
import pandas as pd

class Backtest:

    def __init__(self, df: pd.DataFrame, position_col: str, delay_periods: int = 0):
        self.df = df.copy()
        self.position_col = position_col
        self.delay_periods = delay_periods

    def run(self, fee_rate: float = 0.0002):
        """
        Executes backtest with proper entry-bar vs holding-bar return logic.
        
        Entry bar: return = (close[i] - open[i]) / open[i]
            → You entered at the open of this bar, captured intrabar movement
        
        Holding bar: return = (close[i] - close[i-1]) / close[i-1]
            → You were already in from previous close, standard close-to-close
        
        This prevents weekend/overnight gaps from being incorrectly attributed
        to the entry bar's PnL.
        """
        # 1. Shift position array to represent execution delay (prevent lookahead bias)
        total_shift = 1 + self.delay_periods
        self.df['executable_position'] = self.df[self.position_col].shift(total_shift).fillna(0)

        # 2. Track position changes
        self.df['position_change'] = self.df['executable_position'].diff().abs().fillna(0)
        
        # 3. Identify entry bars vs holding bars
        prev_pos = self.df['executable_position'].shift(1).fillna(0)
        is_entry_bar = (self.df['executable_position'] != 0) & (prev_pos != self.df['executable_position'])
        is_holding_bar = (self.df['executable_position'] != 0) & (prev_pos == self.df['executable_position'])

        # 4. Calculate returns differently for entry vs holding
        # Entry bar: entered at open, PnL = (close - open) / open
        entry_return = (self.df['close'] - self.df['open']) / (self.df['open'] + 1e-8)
        
        # Holding bar: was in from last close, PnL = (close - prev_close) / prev_close
        holding_return = self.df['close'].pct_change()

        # 5. Combine: use entry_return on entry bars, holding_return on holding bars
        bar_return = np.where(is_entry_bar, entry_return, 
                     np.where(is_holding_bar, holding_return, 0.0))
        
        # 6. Apply position direction
        self.df['gross_strategy_returns'] = self.df['executable_position'] * bar_return
        self.df['percentage_fees'] = self.df['position_change'] * fee_rate
        self.df['strategy_returns'] = self.df['gross_strategy_returns'] - self.df['percentage_fees']
        self.df['cum_returns'] = (1 + self.df['strategy_returns']).cumprod()

        # 7. Point PnL (absolute)
        # Entry bar points: close - open
        entry_points = self.df['close'] - self.df['open']
        # Holding bar points: close - prev_close
        holding_points = self.df['close'].diff()
        
        bar_points = np.where(is_entry_bar, entry_points,
                    np.where(is_holding_bar, holding_points, 0.0))
        
        self.df['gross_point_pnl'] = self.df['executable_position'] * bar_points
        self.df['point_fees'] = self.df['position_change'] * (fee_rate * self.df['open'])
        self.df['point_pnl'] = self.df['gross_point_pnl'] - self.df['point_fees']
        self.df['cum_point_pnl'] = self.df['point_pnl'].cumsum()

        # Keep these for compatibility with metrics
        self.df['close_diff'] = self.df['close'].diff()
        self.df['change_rate'] = self.df['close'].pct_change()

        return self.df
