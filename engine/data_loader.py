# engine/data_loader.py
import numpy as np
import pandas as pd
from pathlib import Path


class DataLoader:

    COLUMN_MAP = {
        'name': 'name',
        'time_key': 'time_key',
        'open': 'open',
        'close': 'close',
        'high': 'high',
        'low': 'low',
        'volume': 'volume',
        'turnover': 'turnover',
    }

    VALID_INTERVALS = {
        '1min': '1min',
        '5min': '5min',
        '15min': '15min',
        '30min': '30min',
        '1hr': '1h',
        '4hr': '4h',
        '1d': '1D'
    }

    def __init__(self, data_dir: str = "data/raw"):
        # Resolve relative to this file's parent's parent (project root)
        # so the path works regardless of where you run the script from
        base_dir = Path(__file__).resolve().parent.parent
        self.data_dir = base_dir / data_dir

    def load(self, symbol: str, interval: str = '1hr') -> pd.DataFrame:
        """Loads and resamples data for a given symbol and interval."""

        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Choose from: {list(self.VALID_INTERVALS.keys())}")

        # Load parquet data (filename matches lowercase symbol e.g., 'spy.parquet')
        filepath = self.data_dir / f"{symbol.lower()}.parquet"
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_parquet(filepath)

        # Keep only the columns we want to map and work with
        cols_to_keep = [col for col in df.columns if col in self.COLUMN_MAP]
        df = df[cols_to_keep].rename(columns=self.COLUMN_MAP)

        df['time_key'] = pd.to_datetime(df['time_key'])
        df = df.set_index('time_key').sort_index()

        if interval == '1min':
            return self._finalize(df)

        pandas_interval = self.VALID_INTERVALS[interval]
        df = self._resample_with_rth_clean(df, pandas_interval)

        return self._finalize(df)

    def _resample_with_rth_clean(self, df: pd.DataFrame, pandas_interval: str) -> pd.DataFrame:
        """
        Resamples OHLCV data to a higher timeframe dynamically, with specialized
        intraday logic to absorb the 16:00:00 closing cross tick and prevent
        empty post-market phantom bars across any timeframe (5m, 15m, 1h, 4h, etc.)

        Uses offset-based resampling so that:
          - The bar labeled 09:00 covers minute ticks 09:01 through 10:00 inclusive
          - 'close' = the 10:00:00 tick (last tick of the hour)
          - 'open'  = the 09:01:00 tick (first tick after the boundary)

        This matches market convention where a bar's close price is the price
        printed at the END of the period, not 1 minute before.
        """
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'turnover': 'sum'
        }
        agg_dict = {k: v for k, v in agg_dict.items() if k in df.columns}

        # 1. Determine if we are resampling to an intraday timeframe
        is_intraday = pandas_interval not in ['1D', 'D']

        if is_intraday:
            # Rule: Any tick printed exactly at 16:00:00 (closing cross) is rolled back 1 second
            # to merge into the final active session bar (whether it's 15m, 1h, or 4h)
            closing_cross_mask = (
                (df.index.hour == 16) &
                (df.index.minute == 0) &
                (df.index.second == 0)
            )

            if closing_cross_mask.any():
                new_timestamps = np.where(
                    closing_cross_mask,
                    df.index - pd.Timedelta(seconds=1),
                    df.index
                )
                df.index = pd.to_datetime(new_timestamps)

        # 2. Resample with offset to shift bin edges by 1 minute
        #    Default bins: [09:00, 10:00), [10:00, 11:00) → close = 09:59 tick ❌
        #    With offset='1min': [09:01, 10:01), [10:01, 11:01) → close = 10:00 tick ✅
        #    Then we shift the labels back by 1 min so bars are labeled 09:00, 10:00, etc.
        if is_intraday:
            df_resampled = (
                df
                .resample(pandas_interval, offset='1min')
                .agg(agg_dict)
                .dropna(subset=['close'])
            )
            # Shift labels back so 09:01 label becomes 09:00
            df_resampled.index = df_resampled.index - pd.Timedelta(minutes=1)
        else:
            # Daily resampling — no offset needed
            df_resampled = (
                df
                .resample(pandas_interval)
                .agg(agg_dict)
                .dropna(subset=['close'])
            )

        # 3. Clean up empty post-close rows for intraday
        if is_intraday:
            df_resampled = df_resampled[df_resampled.index.hour < 16]

        return df_resampled

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=['close'])
        return df