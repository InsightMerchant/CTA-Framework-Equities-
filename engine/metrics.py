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
        hours_per_day = 24 if market_type.lower() == 'crypto' else 6.5
        minutes_per_day = hours_per_day * 60

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

        datetime_series = None

        if isinstance(self.df.index, pd.DatetimeIndex):
            datetime_series = self.df.index
        else:
            try:
                converted_index = pd.to_datetime(self.df.index)
                if not converted_index.isna().all():
                    datetime_series = converted_index
            except Exception:
                pass

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

        if datetime_series is not None and len(datetime_series) > 1:
            try:
                time_delta = datetime_series[-1] - datetime_series[0]
                years = time_delta.total_seconds() / (365.25 * 24 * 3600)
            except Exception:
                years = len(strat_returns) / self.annualization_factor
        else:
            years = len(strat_returns) / self.annualization_factor

        if years <= 0:
            years = len(strat_returns) / self.annualization_factor

        if years > 0 and (total_return + 1) > 0:
            cagr_return = (1 + total_return) ** (1 / years) - 1
        else:
            cagr_return = 0.0

        if years > 0:
            simple_annualized_return = total_return / years
        else:
            simple_annualized_return = 0.0

        if strat_returns.std() != 0:
            sharpe_ratio = (strat_returns.mean() / strat_returns.std()) * np.sqrt(self.annualization_factor)
        else:
            sharpe_ratio = 0.0

        negative_returns = strat_returns[strat_returns < 0]
        if len(negative_returns) > 0 and negative_returns.std() != 0:
            sortino_ratio = (strat_returns.mean() / negative_returns.std()) * np.sqrt(self.annualization_factor)
        else:
            sortino_ratio = 0.0

        rolling_max = self.df['cum_returns'].cummax()
        drawdowns = (self.df['cum_returns'] - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

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

    def calculate_directional(self) -> dict:
        """
        Calculate performance metrics by splitting long and short trades.
        Show if the alpha works on both sides or just rides market beta.
        """
        df = self.df.copy()

        # Count actual entries into each direction
        pos = df['executable_position']
        prev_pos = pos.shift(1).fillna(0)

        long_entries = int(((pos == 1) & (prev_pos != 1)).sum())
        short_entries = int(((pos == -1) & (prev_pos != -1)).sum())

        # Split returns by direction
        long_mask = df['executable_position'] == 1
        short_mask = df['executable_position'] == -1

        long_returns = df.loc[long_mask, 'strategy_returns']
        short_returns = df.loc[short_mask, 'strategy_returns']

        long_metrics = self._compute_side_metrics(long_returns, 'long', long_entries)
        short_metrics = self._compute_side_metrics(short_returns, 'short', short_entries)

        # Combined summary
        summary = {
            **long_metrics,
            **short_metrics,
            'long_short_return_ratio': round(
                long_metrics['long_total_return_%'] / (abs(short_metrics['short_total_return_%']) + 1e-8), 2
            ),
            'both_sides_profitable': long_metrics['long_total_return_%'] > 0 and short_metrics['short_total_return_%'] > 0,
        }

        return summary

    def _compute_side_metrics(self, returns: pd.Series, label: str, trade_count: int) -> dict:
        """Compute metrics for one side (long or short)."""

        if len(returns) == 0 or returns.sum() == 0:
            return {
                f'{label}_bars': 0,
                f'{label}_total_return_%': 0.0,
                f'{label}_sharpe': 0.0,
                f'{label}_win_rate_%': 0.0,
                f'{label}_avg_win_%': 0.0,
                f'{label}_avg_loss_%': 0.0,
                f'{label}_profit_factor': 0.0,
                f'{label}_max_drawdown_%': 0.0,
                f'{label}_max_consecutive_loss': 0,
                f'{label}_total_trades': 0,
            }

        # Basic metrics
        total_ret = (1 + returns).prod() - 1
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret / (std_ret + 1e-8)) * np.sqrt(self.annualization_factor)

        # Win/Loss analysis
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / len(returns) * 100 if len(returns) > 0 else 0
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        # Profit factor (total wins / total losses)
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 1e-8
        profit_factor = gross_profit / gross_loss

        # Max consecutive losses
        is_loss = (returns < 0).astype(int)
        consecutive = is_loss * (is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumcount() + 1)
        max_consec_loss = int(consecutive.max()) if len(consecutive) > 0 else 0

        # Max drawdown for this side only
        cum = (1 + returns).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max
        max_dd = drawdown.min() if len(drawdown) > 0 else 0

        return {
            f'{label}_bars': len(returns),
            f'{label}_total_return_%': round(total_ret * 100, 2),
            f'{label}_sharpe': round(sharpe, 4),
            f'{label}_win_rate_%': round(win_rate, 2),
            f'{label}_avg_win_%': round(avg_win * 100, 4),
            f'{label}_avg_loss_%': round(avg_loss * 100, 4),
            f'{label}_profit_factor': round(profit_factor, 4),
            f'{label}_max_drawdown_%': round(max_dd * 100, 2),
            f'{label}_max_consecutive_loss': max_consec_loss,
            f'{label}_total_trades': trade_count,
        }
