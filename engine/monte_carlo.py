# engine/monte_carlo.py
"""
Monte Carlo simulation: perturbs historical price paths to test 
whether the alpha's performance is robust or path-dependent.

The key insight: because factors are computed FROM price data (OHLCV + turnover),
different price paths → different factor values → different signals → different PnL.
"""
import numpy as np
import pandas as pd
from engine.backtest import Backtest
from engine.metrics import PerformanceMetrics
from engine.strategy_combinator import build_strategy


class MonteCarloSimulator:
    
    def __init__(self, df: pd.DataFrame, strategy_spec: dict,
                 interval: str = '1hr', market_type: str = 'equity'):
        """
        Args:
            df: Raw OHLCV DataFrame (BEFORE factor computation)
            strategy_spec: Full strategy specification dict
        """
        self.df = df.copy()
        self.strategy_spec = strategy_spec
        self.interval = interval
        self.market_type = market_type
    
    def _perturb_prices(self, df: pd.DataFrame, noise_pct: float = 0.001, seed: int = None) -> pd.DataFrame:
        """
        Adds random noise to OHLCV prices while preserving:
        - High >= Open, Close, Low
        - Low <= Open, Close, Low
        - Volume/Turnover relationships
        
        noise_pct: standard deviation of noise as fraction of price (0.001 = 0.1%)
        """
        rng = np.random.default_rng(seed)
        perturbed = df.copy()
        
        n_rows = len(df)
        
        # Generate correlated noise for OHLC (same base noise + small independent component)
        base_noise = rng.normal(0, noise_pct, n_rows)
        
        for col in ['open', 'high', 'low', 'close']:
            if col in perturbed.columns:
                independent_noise = rng.normal(0, noise_pct * 0.3, n_rows)
                total_noise = 1 + base_noise + independent_noise
                perturbed[col] = perturbed[col] * total_noise
        
        # Enforce OHLC constraints
        if all(c in perturbed.columns for c in ['open', 'high', 'low', 'close']):
            perturbed['high'] = perturbed[['open', 'high', 'low', 'close']].max(axis=1)
            perturbed['low'] = perturbed[['open', 'high', 'low', 'close']].min(axis=1)
        
        # Perturb volume/turnover slightly (independent)
        if 'volume' in perturbed.columns:
            vol_noise = 1 + rng.normal(0, noise_pct * 2, n_rows)
            perturbed['volume'] = (perturbed['volume'] * vol_noise).clip(lower=0)
        
        if 'turnover' in perturbed.columns:
            turn_noise = 1 + rng.normal(0, noise_pct * 2, n_rows)
            perturbed['turnover'] = (perturbed['turnover'] * turn_noise).clip(lower=0)
        
        return perturbed
    
    def run(self, n_simulations: int = 100, noise_pct: float = 0.001, 
            fee_rate: float = 0.0002) -> dict:
        """
        Runs N Monte Carlo simulations with perturbed price paths.
        
        Returns:
            dict with:
                - 'sharpe_distribution': array of Sharpe ratios across simulations
                - 'summary': dict with mean, std, 5th/95th percentile, % profitable
                - 'all_results': DataFrame of all simulation metrics
        """
        all_results = []
        
        for i in range(n_simulations):
            # 1. Perturb the raw price data
            perturbed_df = self._perturb_prices(self.df, noise_pct=noise_pct, seed=i)
            
            # 2. Recompute factors and signals on perturbed data
            try:
                strategy_df = build_strategy(perturbed_df, self.strategy_spec)
            except Exception:
                continue
            
            # 3. Run backtest
            bt = Backtest(df=strategy_df, position_col='position', delay_periods=0)
            bt_df = bt.run(fee_rate=fee_rate)
            
            # 4. Calculate metrics
            metrics = PerformanceMetrics(df=bt_df, interval=self.interval, market_type=self.market_type)
            perf = metrics.calculate()
            perf['simulation_id'] = i
            all_results.append(perf)
        
        results_df = pd.DataFrame(all_results)
        sharpe_dist = results_df['Sharpe Ratio'].values
        
        summary = {
            'mean_sharpe': round(np.mean(sharpe_dist), 4),
            'std_sharpe': round(np.std(sharpe_dist), 4),
            'median_sharpe': round(np.median(sharpe_dist), 4),
            'sharpe_5th_pct': round(np.percentile(sharpe_dist, 5), 4),
            'sharpe_95th_pct': round(np.percentile(sharpe_dist, 95), 4),
            'pct_profitable': round((sharpe_dist > 0).mean() * 100, 2),
            'pct_above_0.5': round((sharpe_dist > 0.5).mean() * 100, 2),
            'n_simulations': len(results_df),
            'noise_pct': noise_pct
        }
        
        return {
            'sharpe_distribution': sharpe_dist,
            'summary': summary,
            'all_results': results_df
        }
