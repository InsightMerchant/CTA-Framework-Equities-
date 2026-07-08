# engine/walk_forward.py

import pandas as pd
import numpy as np
from typing import Generator, Dict, Tuple

class WalkForwardSplitter:

    def __init__(self, df: pd.DataFrame, is_months: int = 12, oos_months: int = 3, step_months:int = None):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex for calendar-based walk-forward splits.")
        
        self.df = df.sort_index()
        self.is_months = is_months
        self.oos_months = oos_months
        self.step_months = step_months if step_months is not None else oos_months

    def get_slices(self) -> Generator[Dict[str, Tuple[pd.DataFrame, pd.DataFrame]], None, None]:
        
        start_date = self.df.index.min()
        end_date = self.df.index.max()
        is_start = start_date
        fold_idx = 1
        
        while True:
            # Calculate boundaries using exact month offsets
            is_end = is_start + pd.DateOffset(months=self.is_months)
            oos_end = is_end + pd.DateOffset(months=self.oos_months)
            
            # If the Out-of-Sample period doesn't have at least 1 week of data left, we stop
            if oos_end > end_date + pd.DateOffset(days=7):
                break
                
            # Slice with [is_start:is_end] (inclusive of start, exclusive of end to prevent double-counting boundaries)
            is_df = self.df.loc[is_start:is_end - pd.Timedelta(seconds=1)]
            oos_df = self.df.loc[is_end:oos_end - pd.Timedelta(seconds=1)]
            
            # Double check that we actually have data inside both slices
            if len(is_df) > 0 and len(oos_df) > 0:
                yield {
                    "fold": fold_idx,
                    "is_start": is_start.strftime('%Y-%m-%d'),
                    "is_end": (is_end - pd.Timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M'),
                    "oos_start": is_end.strftime('%Y-%m-%d %H:%M'),
                    "oos_end": (oos_end - pd.Timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M'),
                    "train_set": is_df,
                    "test_set": oos_df
                }
            
            is_start = is_start + pd.DateOffset(months=self.step_months)
            fold_idx += 1

if __name__ == "__main__":
    print("--- Testing Walk-Forward Splitter Slicing Period ---")
    dates = pd.date_range(start="2020-01-01", end="2026-06-01", freq="h")
    mock_df = pd.DataFrame(np.random.randn(len(dates)), index=dates, columns=["close"])
    
    # 12 months In-Sample, 3 months Out-of-Sample
    splitter = WalkForwardSplitter(mock_df, is_months=12, oos_months=3, step_months=1)
    
    for fold in splitter.get_slices():
        print(f"Fold {fold['fold']}:")
        print(f"  IS:  {fold['is_start']} to {fold['is_end']} (Bars: {len(fold['train_set'])})")
        print(f"  OOS: {fold['oos_start']} to {fold['oos_end']} (Bars: {len(fold['test_set'])})")
        if fold['fold'] >= 3:
            print("  ... testing logic")
            break