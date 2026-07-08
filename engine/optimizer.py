# engine/optimizer.py
import pandas as pd
import numpy as np
import itertools
import time
import os
import concurrent.futures
from functools import partial

from engine.data_loader import DataLoader
from factors import cci, turnover_mean_z, ma_cross, bias_q, liquidity_premium_z
from engine.backtest import Backtest
from engine.metrics import PerformanceMetrics
from entry_exit_logic.two_factor_reverse import two_factor_reverse
from entry_exit_logic.two_factor_trend import two_factor_trend

def _worker_backtest(params, df, symbol, fee_rate, interval, strategy_func):
    work_df = df.copy()

    work_df = strategy_func(work_df, params)
    bt = Backtest(df=work_df, position_col="position", delay_periods=0)
    bt_df = bt.run(fee_rate=fee_rate)
    metrics_engine = PerformanceMetrics(df=bt_df, interval=interval, market_type="equity")
    performance = metrics_engine.calculate()

    result = {'Symbol':symbol.upper()}
    result.update(params)
    result.update(performance)
    return result

class GridSearchOptimizer:
    def __init__(self, data_dir="data/raw", interval="1hr"):
        self.loader = DataLoader(data_dir=data_dir)
        self.interval = interval
        self.data_cache = {}

    def _get_data(self, symbol):
        if symbol not in self.data_cache:
            self.data_cache[symbol] = self.loader.load(symbol=symbol, interval=self.interval)
        return self.data_cache[symbol]
    
    def run_grid_parallel(self, symbols: list, param_grid: dict, strategy_func, fee_rate: float = 0.0002) -> pd.DataFrame:
        
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        cores = os.cpu_count()
        print(f"🚀 Starting Parallel Grid Search: {len(symbols) * len(combinations)} total runs.")
        
        results_list = []
        start_time = time.time()

        for symbol in symbols:
            print(f"\nLoading {symbol.upper()}...")
            try:
                base_df = self._get_data(symbol)
            except FileNotFoundError:
                continue

            # Pass the custom strategy_func into the worker
            worker_func = partial(
                _worker_backtest, 
                df=base_df, 
                symbol=symbol, 
                fee_rate=fee_rate, 
                interval=self.interval,
                strategy_func=strategy_func 
            )

            completed = 0
            with concurrent.futures.ProcessPoolExecutor(max_workers=cores) as executor:
                futures = {executor.submit(worker_func, params): params for params in combinations}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        results_list.append(future.result())
                        completed += 1
                        if completed % max(1, (len(combinations) // 10)) == 0:
                            print(f"   Progress: {completed}/{len(combinations)} ({(completed/len(combinations))*100:.0f}%)")
                    except Exception as exc:
                        import traceback
                        print(f"❌ Error: {exc}")
                        traceback.print_exc()

        print(f"\n✅ Complete in {round(time.time() - start_time, 2)} seconds!")
        return pd.DataFrame(results_list)
    