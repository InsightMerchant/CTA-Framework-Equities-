# engine/metrics.py

import numpy as np
import pandas as pd

class PerformanceMetrics:
    def __init__(self, df: pd.DataFrame, interval: str = '1h', market_type: str = 'equity'):
        self.df = df
        self.interval = interval
        self.market_type = market_type
        self.annualization_factor = self._get_annualization_factor(interval, market_type)

    def _get_annualization_factor(self, interval: str, market_type: str) -> float:
        days_per_year = 365 if market_type.lower() == 'crypto' else 252
        
        # Define trading hours per day (Equities 6.5 hrs, Crypto 24 hrs)
        hours_per_day = 24 if market_type.lower() == 'crypto' else 6.5
        minutes_per_day = hours_per_day * 60
        
        # Map intervals to their daily frequency
        interval_mapping = {
            '1d': 1,
            '4hr': hours_per_day / 4,
            '1hr': hours_per_day,
            '30min': minutes_per_day / 30,
            '15min': minutes_per_day / 15,
            '5min': minutes_per_day / 5,
            '1min': minutes_per_day
        }
        daily_freq = interval_mapping.get(interval.lower(), 1)
        return days_per_year * daily_freq

    def calculate(self) -> dict:
        strat_returns = self.df['strategy_returns'].dropna()
        total_return = self.df['cum_returns'].iloc[-1] - 1
        total_points = self.df['cum_point_pnl'].iloc[-1]

        # -------------------------------------------------------------
        # Calculate Calendar Time Elapsed Dynamically (No hardcoded years!)
        # -------------------------------------------------------------
        datetime_series = None
        
        # Check if the index is a DatetimeIndex
        if isinstance(self.df.index, pd.DatetimeIndex):
            datetime_series = self.df.index
        else:
            # Try to convert the index to Datetime if it contains string dates
            try:
                converted_index = pd.to_datetime(self.df.index)
                if not converted_index.isna().all():
                    datetime_series = converted_index
            except Exception:
                pass

        # If the index is not a datetime representation, search key date columns
        if datetime_series is None or datetime_series.isna().all():
            for col in ['date', 'timestamp', 'datetime', 'time']:
                if col in self.df.columns:
                    try:
                        converted_col = pd.to_datetime(self.df[col])
                        if not converted_col.isna().all():
                            datetime_series = converted_col
                            break
                    except Exception:
                        continue

        # Compute fractional calendar years
        if datetime_series is not None and len(datetime_series) > 1:
            try:
                time_delta = datetime_series[-1] - datetime_series[0]
                # Standard calendar year calculation (including leap-year adjustments)
                years = time_delta.total_seconds() / (365.25 * 24 * 3600)
            except Exception:
                years = len(strat_returns) / self.annualization_factor
        else:
            # Dynamic fallback to bar count method if no datetimes are present
            years = len(strat_returns) / self.annualization_factor

        # Safeguard against zero or negative values
        if years <= 0:
            years = len(strat_returns) / self.annualization_factor

        # -------------------------------------------------------------
        # Compounded vs Simple Annualized Return Calculations
        # -------------------------------------------------------------
        # 1) CAGR (Compounded Annual Growth Rate) - Standard institutional metric
        if years > 0 and (total_return + 1) > 0:
            cagr_return = (1 + total_return) ** (1 / years) - 1
        else:
            cagr_return = 0.0
            
        # 2) Simple (Arithmetic) Annualized Return - Represents linear average per year
        if years > 0:
            simple_annualized_return = total_return / years
        else:
            simple_annualized_return = 0.0
            
        # Sharpe Ratio (uses bar frequency standard deviation scaling)
        if strat_returns.std() != 0:
            sharpe_ratio = (strat_returns.mean() / strat_returns.std()) * np.sqrt(self.annualization_factor)
        else:
            sharpe_ratio = 0.0
            
        # Sortino Ratio
        negative_returns = strat_returns[strat_returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() != 0:
            sortino_ratio = (strat_returns.mean() / negative_returns.std()) * np.sqrt(self.annualization_factor)
        else:
            sortino_ratio = 0.0
            
        # Maximum Drawdown (MDD)
        rolling_max = self.df['cum_returns'].cummax()
        drawdowns = (self.df['cum_returns'] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calculate Trade Count
        position_changes = self.df['executable_position'].diff().abs()
        total_trades = (position_changes > 0).sum()
        
        return {
            "Total Return (%)": round(total_return * 100, 2),
            "Total Points Captured": round(total_points, 2),
            "CAGR (Compounded Annual Return %)": round(cagr_return * 100, 2),
            "Simple Average Annual Return (%)": round(simple_annualized_return * 100, 2),
            "Sharpe Ratio": round(sharpe_ratio, 4),
            "Sortino Ratio": round(sortino_ratio, 4),
            "Max Drawdown (%)": round(max_drawdown * 100, 2),
            "Total Trades": int(total_trades)
        }