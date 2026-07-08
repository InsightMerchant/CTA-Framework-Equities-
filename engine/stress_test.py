# engine/stress_test.py
"""
Stress testing: evaluates whether an alpha remains profitable 
under adversarial execution conditions (delayed fills).
"""
import pandas as pd
from engine.backtest import Backtest
from engine.metrics import PerformanceMetrics


class StressTest:
    
    def __init__(self, df: pd.DataFrame, position_col: str = 'position', 
                 interval: str = '1hr', market_type: str = 'equity'):
        self.df = df.copy()
        self.position_col = position_col
        self.interval = interval
        self.market_type = market_type
    
    def run_delay_stress(self, max_delay: int = 2, fee_rate: float = 0.0002) -> pd.DataFrame:
        """
        Runs the backtest with increasing execution delays (0, 1, 2 intervals).
        If the alpha survives delay=1, it's robust to late execution.
        
        Returns a DataFrame comparing metrics at each delay level.
        """
        results = []
        
        for delay in range(0, max_delay + 1):
            bt = Backtest(df=self.df, position_col=self.position_col, delay_periods=delay)
            bt_df = bt.run(fee_rate=fee_rate)
            
            metrics_engine = PerformanceMetrics(df=bt_df, interval=self.interval, market_type=self.market_type)
            perf = metrics_engine.calculate()
            perf['delay_periods'] = delay
            perf['total_shift'] = 1 + delay  # actual shift applied in backtest
            results.append(perf)
        
        results_df = pd.DataFrame(results)
        results_df['sharpe_decay_%'] = (
            (results_df['Sharpe Ratio'] - results_df['Sharpe Ratio'].iloc[0]) 
            / abs(results_df['Sharpe Ratio'].iloc[0]) * 100
        ).round(2)
        
        return results_df
    
    def passes_stress(self, min_sharpe: float = 0.3, fee_rate: float = 0.0002) -> bool:
        """Quick check: is the alpha still profitable with 1 interval delay?"""
        bt = Backtest(df=self.df, position_col=self.position_col, delay_periods=1)
        bt_df = bt.run(fee_rate=fee_rate)
        metrics = PerformanceMetrics(df=bt_df, interval=self.interval, market_type=self.market_type)
        perf = metrics.calculate()
        return perf['Sharpe Ratio'] >= min_sharpe