# test.py
# Full-period validation: runs one specific param set across ALL data (2020-2026)
# Purpose: visually inspect the CSV output to confirm entry/exit logic is correct
import pandas as pd
import numpy as np

from engine.data_loader import DataLoader
from factors import bias_q, liquidity_premium_z, ma_cross, turnover_mean_z
from entry_exit_logic.two_factor_reverse import two_factor_reverse
from entry_exit_logic.two_factor_trend import two_factor_trend
from engine.backtest import Backtest
from engine.metrics import PerformanceMetrics


def test():
    try:
        # ─── CONFIG (match what you tested in run_pipeline.py) ────────
        SYMBOL = "spy"
        INTERVAL = "1hr"
        N = 150
        # DIR_THRESH_LONG = 0.7
        # DIR_THRESH_SHORT = 1.0 - DIR_THRESH_LONG  # = 0.1
        DIR_THRESH_LONG = 1.0
        DIR_THRESH_SHORT = -1.0
        CONF_THRESH = 2.0
        CONF_MODE = "greater"
        EXIT_MODE = "conf_cross_zero"  # or 'fixed_bars' or 'flip' or 'conf_cross_zero'
        EXIT_BARS = 4                # only matters if EXIT_MODE='fixed_bars'
        FEE_RATE = 0.0002

        print("=" * 60)
        print("🧪 FULL-PERIOD VALIDATION TEST")
        print("=" * 60)
        print(f"   Symbol: {SYMBOL.upper()} | Interval: {INTERVAL}")
        print(f"   Params: n={N}, dir_long={DIR_THRESH_LONG}, dir_short={DIR_THRESH_SHORT}")
        print(f"           conf_thresh={CONF_THRESH}, conf_mode={CONF_MODE}")
        print(f"           exit_mode={EXIT_MODE}, exit_bars={EXIT_BARS}")
        print()

        # ─── 1. LOAD DATA ─────────────────────────────────────────────
        print("1. Loading Data...")
        loader = DataLoader(data_dir="data/raw")
        df = loader.load(symbol=SYMBOL, interval=INTERVAL)
        print(f"   ✅ Loaded {len(df)} bars ({df.index.min()} → {df.index.max()})")

        # ─── 2. COMPUTE FACTORS ───────────────────────────────────────
        print("\n2. Calculating Factors...")
        df = ma_cross.signal(df, N, 'dir_factor')
        df = turnover_mean_z.signal(df, N, 'conf_factor')
        print(f"   ✅ dir_factor (bias_q) range: [{df['dir_factor'].min():.4f}, {df['dir_factor'].max():.4f}]")
        print(f"   ✅ conf_factor (liq_prem_z) range: [{df['conf_factor'].min():.4f}, {df['conf_factor'].max():.4f}]")

        # ─── 3. APPLY ENTRY/EXIT LOGIC ────────────────────────────────
        print(f"\n3. Applying entry/exit logic (exit_mode='{EXIT_MODE}')...")
        # Change to two_factor_trend/two_factor_reverse
        df['position'] = two_factor_trend(
            df=df,
            dir_factor='dir_factor',
            dir_long_thresh=DIR_THRESH_LONG,
            dir_short_thresh=DIR_THRESH_SHORT,
            conf_factor='conf_factor',
            conf_thresh=CONF_THRESH,
            conf_mode=CONF_MODE,
            exit_mode=EXIT_MODE,
            exit_bars=EXIT_BARS,
        )

        # Position stats
        total_bars = len(df)
        bars_long = (df['position'] == 1).sum()
        bars_short = (df['position'] == -1).sum()
        bars_flat = (df['position'] == 0).sum()
        print(f"   Total bars: {total_bars}")
        print(f"   Long:  {bars_long} bars ({bars_long/total_bars*100:.1f}%)")
        print(f"   Short: {bars_short} bars ({bars_short/total_bars*100:.1f}%)")
        print(f"   Flat:  {bars_flat} bars ({bars_flat/total_bars*100:.1f}%)")

        # ─── 4. RUN BACKTEST ──────────────────────────────────────────
        print("\n4. Running Backtest...")
        bt = Backtest(df=df, position_col='position', delay_periods=0)
        bt_df = bt.run(fee_rate=FEE_RATE)

        # ─── 5. CALCULATE METRICS ─────────────────────────────────────
        print("5. Calculating Metrics...")
        metrics_engine = PerformanceMetrics(df=bt_df, interval=INTERVAL, market_type="equity")
        results = metrics_engine.calculate()

        # ─── 6. EXPORT CSV ────────────────────────────────────────────
        csv_filename = f"test_full_period_{EXIT_MODE}.csv"
        bt_df.to_csv(csv_filename)
        print(f"\n📁 Saved bar-by-bar data to '{csv_filename}'")
        print("   Columns to inspect in CSV:")
        print("   - dir_factor: the percentile rank (0-1)")
        print("   - conf_factor: the z-score")
        print("   - position: raw signal (+1, -1, 0)")
        print("   - executable_position: shifted position (what actually trades)")
        print("   - strategy_returns: per-bar net returns")
        print("   - cum_returns: compounded equity curve")

        # ─── 7. PRINT RESULTS ─────────────────────────────────────────
        print("\n" + "=" * 60)
        print("📈 BACKTEST RESULTS (Full Period 2020-2026)")
        print("=" * 60)
        for key, value in results.items():
            print(f"   {key}: {value}")

        # ─── 8. QUICK SANITY CHECKS ──────────────────────────────────
        print("\n" + "=" * 60)
        print("🔍 SANITY CHECKS")
        print("=" * 60)

        # Check: does position actually go flat?
        position_changes = df['position'].diff().abs()
        entries = (position_changes > 0).sum()
        print(f"   Position changes (entries+exits): {entries}")

        # Check: with conf_cross_zero, does it exit when conf >= 0?
        if EXIT_MODE == 'conf_cross_zero':
            # Find bars where position goes from non-zero to zero
            exits = (df['position'] == 0) & (df['position'].shift(1) != 0)
            if exits.sum() > 0:
                exit_conf_values = df.loc[exits, 'conf_factor']
                print(f"   Exits found: {exits.sum()}")
                print(f"   conf_factor at exit bars — mean: {exit_conf_values.mean():.4f}, min: {exit_conf_values.min():.4f}")
                print(f"   (Should be >= 0 for 'lesser' mode exits)")
            else:
                print("   ⚠️ No exits found! Position never goes flat.")

        elif EXIT_MODE == 'fixed_bars':
            # Check average hold time
            # Count consecutive non-zero blocks
            is_in_trade = df['position'] != 0
            trade_blocks = (is_in_trade != is_in_trade.shift()).cumsum()
            trade_lengths = is_in_trade.groupby(trade_blocks).sum()
            trade_lengths = trade_lengths[trade_lengths > 0]
            if len(trade_lengths) > 0:
                print(f"   Avg hold time: {trade_lengths.mean():.1f} bars")
                print(f"   Max hold time: {trade_lengths.max()} bars (should be <= {EXIT_BARS})")
                print(f"   (If max > {EXIT_BARS}, there's a bug in exit logic)")

        print("\n✅ Test complete. Open the CSV to manually verify entry/exit bars.")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure your 'spy.parquet' file is in the 'data/raw/' directory.")
    except Exception as e:
        import traceback
        print(f"\n❌ An unexpected error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test()