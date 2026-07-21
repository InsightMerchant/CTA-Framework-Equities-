# run_pipeline.py
"""
Master Pipeline — Config-driven alpha research workflow.
All you set here: symbol, interval, dir_factor, conf_factor, logic.
Everything else reads from config.py.

Phases:
  1. 3D Grid Search on Sandbox (2020-2022) → CSV + Heatmap
  2. Manual Selection (you pick params from heatmap + CSV)
  2.5. Full-Period Backtest (2020-2026 baseline)
  3. Walk-Forward Validation (12m IS + 6m OOS from 2024-01 → end)
  4. Stress Test (delay +1 interval)
  5. Monte Carlo (perturbed price paths)
"""
import pandas as pd
import numpy as np
import json
import config
import itertools
import time
from pathlib import Path
from engine.data_loader import DataLoader
from engine.backtest import Backtest
from engine.metrics import PerformanceMetrics
from engine.walk_forward import WalkForwardSplitter

# import factors
from factors import bias_q, liquidity_premium_z, cci, turnover_mean_z, vwap_bias, ma_cross, order_imbalance, volatility_friction_ratio, parkinson_volatility, returns_volatility, asr, ddv

# import entry exit logic
from entry_exit_logic.two_factor_reverse import two_factor_reverse
from entry_exit_logic.two_factor_trend import two_factor_trend
from entry_exit_logic.two_factor_reverse_long_only import two_factor_reverse_long_only
from entry_exit_logic.two_factor_reverse_short_only import two_factor_reverse_short_only
from entry_exit_logic.two_factor_trend_long_only import two_factor_trend_long_only
from entry_exit_logic.two_factor_trend_short_only import two_factor_trend_short_only

_DIR_FACTORS = {
    'BIAS_Q': bias_q.signal,
    'CCI': cci.signal,
    'VWAP_BIAS': vwap_bias.signal,
    'MA_CROSS': ma_cross.signal,
    'ORDER_IMBALANCE': order_imbalance.signal,
}

_CONF_FACTORS = {
    'LIQUIDITY_PREMIUM_Z': liquidity_premium_z.signal,
    'TURNOVER_MEAN_Z': turnover_mean_z.signal,
    'ORDER_IMBALANCE': order_imbalance.signal,
    'VOLATILITY_FRICTION_RATIO': volatility_friction_ratio.signal,
    'PARKINSON_VOLATILITY': parkinson_volatility.signal,
    'RETURN_VOLATILITY': returns_volatility.signal,
}

_LOGIC_FUNCS = {
    'reverse': two_factor_reverse,
    'trend': two_factor_trend,
    'reverse_long_only': two_factor_reverse_long_only,
    'reverse_short_only': two_factor_reverse_short_only,
    'trend_long_only': two_factor_trend_long_only,
    'trend_short_only': two_factor_trend_short_only,
}

# ─────────────────────────────────────────────────────────────────────
# RESULTS OUTPUT MANAGER
# ─────────────────────────────────────────────────────────────────────

def get_results_dir(symbol, dir_factor, conf_factor, logic):

    base_dir = Path(__file__).resolve().parent

    # Build exit mode suffix
    exit_mode = config.EXIT_CONFIG['exit_mode']
    if exit_mode == 'fixed_bars':
        exit_suffix = f"fixed_bars_{config.EXIT_CONFIG['exit_bars']}"
    else:
        exit_suffix = exit_mode

    # Asset folder
    asset_folder = symbol.upper()

    # Strategy folder
    strategy_folder = f"{dir_factor}_x_{conf_factor}_{logic}_{exit_suffix}"

    results_path = base_dir / "results" / asset_folder / strategy_folder
    results_path.mkdir(parents=True, exist_ok=True)
    return results_path

# ─────────────────────────────────────────────────────────────────────
# APPLY STRATEGY
# ─────────────────────────────────────────────────────────────────────

def apply_strategy(df, params, dir_factor, conf_factor, logic, slice_to_sandbox=True):
    """
    Universal strategy function. Works for ANY combo of dir × conf × logic.
    No pickling needed — this runs in-process.

    Args:
        df: OHLCV DataFrame
        params: dict with 'n', 'dir_thresh_symmetric'/'dir_thresh_long+short', 'conf_thresh', 'conf_mode'
        dir_factor: e.g. 'BIAS_Q', 'CCI'
        conf_factor: e.g. 'LIQUIDITY_PREMIUM_Z_LESSER', 'TURNOVER_MEAN_Z_GREATER'
        logic: 'reverse' or 'trend'
        slice_to_sandbox: if True, slice to OPTIMIZATION_START/END dates
    """
    n = int(params.get('n', params.get('n1', 20)))
    n2 = int(params.get('n2', n))

    # Resolve confirmation factor base name (strip _LESSER/_GREATER suffix)
    conf_base = conf_factor.replace('_LESSER', '').replace('_GREATER', '')

    # Compute factors
    df = _DIR_FACTORS[dir_factor](df, n, 'dir_factor')
    df = _CONF_FACTORS[conf_base](df, n2, 'conf_factor')

    # Slice to sandbox if requested (Phase 1 only)
    if slice_to_sandbox:
        df = df.loc[config.OPTIMIZATION_START:config.OPTIMIZATION_END].copy()

    # Handle symmetric threshold → expand to long/short
    if 'dir_thresh_symmetric' in params:
        dir_long = params['dir_thresh_symmetric']
        if dir_factor in ['BIAS_Q']:  # Rank-based (0 to 1)
            dir_short = 1.0 - dir_long
        elif dir_factor in ['MA_CROSS', 'CCI', 'VWAP_BIAS']:  # Binary/symmetric around 0
            dir_short = -dir_long
        else:
            dir_short = -dir_long
    else:
        dir_long = params.get('dir_thresh_long', 0.9)
        dir_short = params.get('dir_thresh_short', 0.1)

    exit_mode = params.get('exit_mode', config.EXIT_CONFIG['exit_mode'])
    exit_threshold = float(params.get('exit_threshold', 0))
    exit_bars = int(params.get('exit_bars', config.EXIT_CONFIG['exit_bars']))

    # Apply entry/exit logic
    logic_func = _LOGIC_FUNCS[logic]
    df['position'] = logic_func(
        df=df,
        dir_factor='dir_factor',
        dir_long_thresh=dir_long,
        dir_short_thresh=dir_short,
        conf_factor='conf_factor',
        conf_thresh=params.get('conf_thresh', -1.5),
        conf_mode=params.get('conf_mode', 'lesser'),
        exit_mode=exit_mode,
        exit_bars=exit_bars,
        exit_threshold=exit_threshold,
    )

    return df


# ─────────────────────────────────────────────────────────────────────
# GRID SEARCH (In-Process — No Pickling Issues)
# ─────────────────────────────────────────────────────────────────────

def run_grid_search(base_df, param_grid, dir_factor, conf_factor, logic,
                    fee_rate=0.0002, interval='1hr'):
    """Simple in-process grid search. No pickling, no multiprocessing."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"🚀 Starting Grid Search: {len(combinations)} total runs.")
    results_list = []
    start_time = time.time()

    for i, params in enumerate(combinations):
        work_df = base_df.copy()
        work_df = apply_strategy(work_df, params, dir_factor, conf_factor, logic, slice_to_sandbox=True)

        bt = Backtest(df=work_df, position_col='position', delay_periods=0)
        bt_df = bt.run(fee_rate=fee_rate)
        metrics = PerformanceMetrics(df=bt_df, interval=interval, market_type='equity')
        perf = metrics.calculate()

        result = {}
        result.update(params)
        result.update(perf)
        results_list.append(result)

        if (i + 1) % max(1, len(combinations) // 10) == 0:
            print(f"   Progress: {i+1}/{len(combinations)} ({(i+1)/len(combinations)*100:.0f}%)")

    elapsed = round(time.time() - start_time, 2)
    print(f"✅ Complete in {elapsed} seconds!")
    return pd.DataFrame(results_list)


# ─────────────────────────────────────────────────────────────────────
# STRESS TEST (Phase 4)
# ─────────────────────────────────────────────────────────────────────

def run_stress_test(df, interval='1hr', max_delay=2, fee_rate=0.0002):
    """Tests if alpha survives with delayed execution."""
    results = []
    for delay in range(0, max_delay + 1):
        bt = Backtest(df=df.copy(), position_col='position', delay_periods=delay)
        bt_df = bt.run(fee_rate=fee_rate)
        metrics = PerformanceMetrics(df=bt_df, interval=interval, market_type='equity')
        perf = metrics.calculate()
        perf['delay_periods'] = delay
        results.append(perf)

    results_df = pd.DataFrame(results)
    base_sharpe = results_df['Sharpe Ratio'].iloc[0]
    results_df['sharpe_decay_%'] = (
        (results_df['Sharpe Ratio'] - base_sharpe) / (abs(base_sharpe) + 1e-8) * 100
    ).round(2)
    return results_df


# ─────────────────────────────────────────────────────────────────────
# MONTE CARLO (Phase 5)
# ─────────────────────────────────────────────────────────────────────

def run_monte_carlo(base_df, params, dir_factor, conf_factor, logic,
                    interval='1hr', n_simulations=100, noise_pct=0.001, fee_rate=0.0002):
    """Perturbs price paths → recomputes factors → different signals → different PnL."""
    sharpe_results = []

    for i in range(n_simulations):
        rng = np.random.default_rng(seed=i)
        perturbed = base_df.copy()
        n_rows = len(perturbed)

        base_noise = rng.normal(0, noise_pct, n_rows)
        for col in ['open', 'high', 'low', 'close']:
            if col in perturbed.columns:
                ind_noise = rng.normal(0, noise_pct * 0.3, n_rows)
                perturbed[col] = perturbed[col] * (1 + base_noise + ind_noise)

        perturbed['high'] = perturbed[['open', 'high', 'low', 'close']].max(axis=1)
        perturbed['low'] = perturbed[['open', 'high', 'low', 'close']].min(axis=1)

        if 'volume' in perturbed.columns:
            perturbed['volume'] = (perturbed['volume'] * (1 + rng.normal(0, noise_pct * 2, n_rows))).clip(lower=0)
        if 'turnover' in perturbed.columns:
            perturbed['turnover'] = (perturbed['turnover'] * (1 + rng.normal(0, noise_pct * 2, n_rows))).clip(lower=0)

        try:
            work_df = apply_strategy(perturbed, params, dir_factor, conf_factor, logic, slice_to_sandbox=False)
            bt = Backtest(df=work_df, position_col='position', delay_periods=0)
            bt_df = bt.run(fee_rate=fee_rate)
            metrics = PerformanceMetrics(df=bt_df, interval=interval, market_type='equity')
            perf = metrics.calculate()
            sharpe_results.append(perf['Sharpe Ratio'])
        except Exception:
            continue

    sharpe_arr = np.array(sharpe_results)
    summary = {
        'mean_sharpe': round(np.mean(sharpe_arr), 4),
        'std_sharpe': round(np.std(sharpe_arr), 4),
        'median_sharpe': round(np.median(sharpe_arr), 4),
        'sharpe_5th_pct': round(np.percentile(sharpe_arr, 5), 4),
        'sharpe_95th_pct': round(np.percentile(sharpe_arr, 95), 4),
        'pct_profitable': round((sharpe_arr > 0).mean() * 100, 2),
        'n_simulations': len(sharpe_results),
    }
    return summary, sharpe_arr


# ─────────────────────────────────────────────────────────────────────
# HEATMAP GENERATION (saves to results folder)
# ─────────────────────────────────────────────────────────────────────

def generate_heatmap(results_df, dir_factor, conf_factor, logic, output_dir):
    """
    Generates heatmaps adapted to constrained vs unconstrained mode.
    
    Constrained: one heatmap per dir_thresh_symmetric, Y=n, X=conf_thresh
    Unconstrained: one heatmap per (exit_threshold × conf_thresh) pair, Y=n, X=(long/short pairs)
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    if 'dir_thresh_symmetric' in results_df.columns:
        # ─────────────────────────────────────────────────────────────
        # CONSTRAINED MODE: one heatmap per dir_thresh_symmetric value
        # ─────────────────────────────────────────────────────────────
        y_col = 'n'
        x_col = 'conf_thresh'
        slice_col = 'dir_thresh_symmetric'

        for thresh_val in sorted(results_df[slice_col].unique()):
            subset = results_df[results_df[slice_col] == thresh_val]
            pivot = subset.pivot_table(index=y_col, columns=x_col, values='Sharpe Ratio')

            plt.figure(figsize=(8, 10))
            sns.heatmap(pivot, annot=True, cmap="RdYlGn", center=0, fmt=".2f", linewidths=.5)
            plt.title(f'{dir_factor} × {conf_factor} ({logic})\ndir_thresh_symmetric = {thresh_val}')
            plt.xlabel('CONF_THRESH')
            plt.ylabel('LOOKBACK (N)')
            plt.tight_layout()

            filename = output_dir / f"heatmap_thresh{thresh_val}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved: {filename.name}")

    elif 'dir_thresh_long' in results_df.columns:
        # ─────────────────────────────────────────────────────────────
        # UNCONSTRAINED MODE: one heatmap per (exit_threshold, conf_thresh) pair
        # Y-axis = n, X-axis = "L=X / S=Y" threshold pairs
        # ─────────────────────────────────────────────────────────────
        y_col = 'n'

        # Get unique values for both slicing dimensions
        exit_thresholds = sorted(results_df['exit_threshold'].unique()) if 'exit_threshold' in results_df.columns else [None]
        conf_thresholds = sorted(results_df['conf_thresh'].unique()) if 'conf_thresh' in results_df.columns else [None]

        # Create combined X-axis label for all rows
        results_df = results_df.copy()
        results_df['threshold_pair'] = (
            'L=' + results_df['dir_thresh_long'].round(2).astype(str) +
            ' / S=' + results_df['dir_thresh_short'].round(2).astype(str)
        )

        for exit_val in exit_thresholds:
            for conf_val in conf_thresholds:
                # Filter to this specific (exit_threshold, conf_thresh) combination
                mask = pd.Series(True, index=results_df.index)
                if exit_val is not None:
                    mask &= results_df['exit_threshold'] == exit_val
                if conf_val is not None:
                    mask &= results_df['conf_thresh'] == conf_val

                subset = results_df[mask]
                if subset.empty:
                    continue

                # Pivot: Y=n, X=threshold_pair, values=Sharpe
                pivot = subset.pivot_table(
                    index=y_col,
                    columns='threshold_pair',
                    values='Sharpe Ratio',
                )

                if pivot.empty:
                    continue

                # Sort columns: grouped by long threshold, then short within each
                sorted_cols = sorted(pivot.columns, key=lambda x: (
                    float(x.split(' / ')[0].replace('L=', '')),
                    float(x.split(' / ')[1].replace('S=', ''))
                ))
                pivot = pivot[sorted_cols]

                # Dynamic figure width
                n_cols = len(pivot.columns)
                fig_width = max(12, n_cols * 1.2)

                plt.figure(figsize=(fig_width, 10))
                sns.heatmap(
                    pivot, annot=True, cmap="RdYlGn", center=0,
                    fmt=".2f", linewidths=.5, annot_kws={"size": 7}
                )

                # Build title showing both slice values
                title_parts = [f'{dir_factor} × {conf_factor} ({logic})']
                if exit_val is not None:
                    title_parts.append(f'exit_threshold = {exit_val}')
                if conf_val is not None:
                    title_parts.append(f'conf_thresh = {conf_val}')
                plt.title('\n'.join(title_parts), fontsize=11)

                plt.xlabel('LONG THRESHOLD / SHORT THRESHOLD', fontsize=9)
                plt.ylabel('LOOKBACK (N)', fontsize=10)
                plt.xticks(rotation=45, ha='right', fontsize=8)
                plt.tight_layout()

                # Filename includes both slice values
                fname_parts = []
                if exit_val is not None:
                    fname_parts.append(f"exit{exit_val}")
                if conf_val is not None:
                    fname_parts.append(f"conf{conf_val}")
                filename = output_dir / f"heatmap_{'_'.join(fname_parts)}.png"

                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Saved: {filename.name}")

    else:
        print("   ⚠️ Cannot generate heatmap — no recognized threshold column found.")

# ─────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    symbol: str = 'spy',
    interval: str = '1hr',
    dir_factor: str = 'BIAS_Q',
    conf_factor: str = 'LIQUIDITY_PREMIUM_Z_LESSER',
    logic: str = 'reverse',
    grid_mode: str = 'constrained',  # NEW: 'constrained' or 'unconstrained'
    mc_simulations: int = 100,
    mc_noise: float = 0.001,
    selected_params: dict = None,
):
    """
    Config-driven pipeline. Set your hypothesis above and run.

    Workflow:
      Phase 1: Grid search → CSV + Heatmap → YOU review & pick params
      Phase 2: (Manual) Review heatmap, pick stable params, re-run with selected_params
      Phase 2.5: Full-Period Backtest (2020-2026 baseline + exported CSV)
      Phase 3: Walk-Forward (12m IS + 6m OOS, rolling monthly from 2024-01)
      Phase 4: Stress Test (delay +1)
      Phase 5: Monte Carlo (perturbed paths)
    """
    # Create organized output directory
    output_dir = get_results_dir(symbol, dir_factor, conf_factor, logic)

    print("=" * 60)
    print(f"🚀 PIPELINE: {dir_factor} × {conf_factor} ({logic})")
    print(f"   Symbol: {symbol.upper()} | Interval: {interval}")
    print(f"   Output: {output_dir}")
    print("=" * 60)

    # Load full dataset once
    loader = DataLoader(data_dir="data/raw")
    full_df = loader.load(symbol=symbol, interval=interval)
    print(f"   Loaded {len(full_df)} bars ({full_df.index.min()} → {full_df.index.max()})")

    # Build grid from config
    if grid_mode == 'constrained':
        target_grid = config.CONSTRAINED_GRID
    elif grid_mode == 'unconstrained':
        target_grid = config.UNCONSTRAINED_GRID
    else:
        raise ValueError(f"grid_mode must be 'constrained' or 'unconstrained', got '{grid_mode}'")

    grid = {**target_grid['DIRECTIONAL'][dir_factor], **target_grid['CONFIRMATION'][conf_factor]}
    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Grid Search on Sandbox
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"📊 PHASE 1: 3D Grid Search")
    print(f"   Sandbox: {config.OPTIMIZATION_START} → {config.OPTIMIZATION_END}")
    print(f"{'='*60}")

    results_df = run_grid_search(
        base_df=full_df, param_grid=grid,
        dir_factor=dir_factor, conf_factor=conf_factor, logic=logic,
        fee_rate=0.0002, interval=interval
    )

    # Save grid CSV to results folder
    grid_csv = output_dir / "grid_search.csv"
    results_df.sort_values('Sharpe Ratio', ascending=False).to_csv(grid_csv, index=False)
    print(f"\n📁 Grid results saved: {grid_csv}")

    # Generate heatmaps into results folder
    generate_heatmap(results_df, dir_factor, conf_factor, logic, output_dir)

    # Show top results
    print(f"\n🏆 Top 10 (for reference — check heatmap for stability):")
    top_cols = [c for c in ['n', 'dir_thresh_symmetric', 'conf_thresh', 'conf_mode', 'exit_threshold',
                            'Sharpe Ratio', 'Total Return (%)', 'Max Drawdown (%)',
                            'Total Trades'] if c in results_df.columns]
    print(results_df.sort_values('Sharpe Ratio', ascending=False).head(10)[top_cols].to_string())

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: GATE
    # ═══════════════════════════════════════════════════════════════════
    if selected_params is None:
        print(f"\n{'='*60}")
        print("⏸️  PHASE 1 COMPLETE — Review the heatmap & CSV.")
        print(f"   📂 Results in: {output_dir}")
        print("   Once you've picked stable params, re-run with:")
        print(f"   run_full_pipeline(..., selected_params={{'n': X, 'dir_thresh_symmetric': Y, 'conf_thresh': Z, 'conf_mode': '...'}})")
        print(f"{'='*60}")
        return {'grid_results': results_df, 'output_dir': str(output_dir)}

    # ═══════════════════════════════════════════════════════════════════
    # AUTO-INJECT: Fill missing params from confirmation factor config
    # so user doesn't need to manually remember exit_threshold, conf_mode
    # ═══════════════════════════════════════════════════════════════════
    conf_config = target_grid['CONFIRMATION'].get(conf_factor, {})
    for key in ['exit_threshold', 'conf_mode']:
        if key not in selected_params and key in conf_config:
            val = conf_config[key]
            selected_params[key] = val[0] if isinstance(val, (list, np.ndarray)) else val
            print(f"   ℹ️ Auto-injected {key}={selected_params[key]} from config")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2.5: Full-Period Backtest (2019-2026)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"📊 PHASE 2.5: Full-Period Backtest")
    print(f"   Period: {full_df.index.min().strftime('%Y-%m-%d')} → {full_df.index.max().strftime('%Y-%m-%d')}")
    print(f"   Params: {selected_params}")
    print(f"{'='*60}")

    full_strategy_df = apply_strategy(full_df.copy(), selected_params, dir_factor, conf_factor, logic, slice_to_sandbox=False)

    bt_full = Backtest(df=full_strategy_df, position_col='position', delay_periods=0)
    bt_full_df = bt_full.run(fee_rate=0.0002)
    metrics_full = PerformanceMetrics(df=bt_full_df, interval=interval, market_type='equity')
    full_period_perf = metrics_full.calculate()
    directional_perf = metrics_full.calculate_directional()

    print(f"\n   📊 LONG/SHORT BREAKDOWN:")
    print(f"   {'─'*55}")
    print(f"   LONG  — Return: {directional_perf['long_total_return_%']}% | Sharpe: {directional_perf['long_sharpe']} | Win Rate: {directional_perf['long_win_rate_%']}% | PF: {directional_perf['long_profit_factor']} | Trades: {directional_perf['long_total_trades']}")
    print(f"   SHORT — Return: {directional_perf['short_total_return_%']}% | Sharpe: {directional_perf['short_sharpe']} | Win Rate: {directional_perf['short_win_rate_%']}% | PF: {directional_perf['short_profit_factor']} | Trades: {directional_perf['short_total_trades']}")
    print(f"   Both Sides Profitable: {'✅ YES' if directional_perf['both_sides_profitable'] else '❌ NO'}")
    print(f"   Long/Short Return Ratio: {directional_perf['long_short_return_ratio']}x")

    # Position breakdown
    total_bars = len(full_strategy_df)
    bars_long = (full_strategy_df['position'] == 1).sum()
    bars_short = (full_strategy_df['position'] == -1).sum()
    bars_flat = (full_strategy_df['position'] == 0).sum()

    print(f"   Total bars: {total_bars}")
    print(f"   Long:  {bars_long} ({bars_long/total_bars*100:.1f}%) | Short: {bars_short} ({bars_short/total_bars*100:.1f}%) | Flat: {bars_flat} ({bars_flat/total_bars*100:.1f}%)")
    print()
    for key, value in full_period_perf.items():
        print(f"   {key}: {value}")

    # Save full-period CSV for manual inspection
    full_period_csv = output_dir / "full_period_backtest.csv"
    bt_full_df.to_csv(full_period_csv)
    print(f"\n📁 Full-period bar-by-bar saved: {full_period_csv}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: Walk-Forward (2024-01 → end of data)
    # ═══════════════════════════════════════════════════════════════════
    wf_start = pd.Timestamp(config.OPTIMIZATION_END) + pd.Timedelta(days=1)
    print(f"\n{'='*60}")
    print(f"📈 PHASE 3: Walk-Forward Validation")
    print(f"   Selected params: {selected_params}")
    print(f"   12m IS + 6m OOS, step=1 month, starting {wf_start.strftime('%Y-%m-%d')}")
    print(f"{'='*60}")

    wf_df = full_df.loc[wf_start.strftime('%Y-%m-%d'):].copy()
    splitter = WalkForwardSplitter(wf_df, is_months=12, oos_months=6, step_months=1)

    wf_results = []
    for fold in splitter.get_slices():
        oos_data = fold['test_set'].copy()
        oos_data = apply_strategy(oos_data, selected_params, dir_factor, conf_factor, logic, slice_to_sandbox=False)

        bt = Backtest(df=oos_data, position_col='position', delay_periods=0)
        bt_df = bt.run(fee_rate=0.0002)
        metrics = PerformanceMetrics(df=bt_df, interval=interval, market_type='equity')
        perf = metrics.calculate()
        perf['fold'] = fold['fold']
        perf['oos_period'] = f"{fold['oos_start']} → {fold['oos_end']}"
        wf_results.append(perf)

    wf_results_df = pd.DataFrame(wf_results)
    print(wf_results_df[['fold', 'oos_period', 'Sharpe Ratio', 'Total Return (%)', 'Max Drawdown (%)', 'Total Trades']].to_string())

    # Save walk-forward results
    wf_csv = output_dir / "walk_forward.csv"
    wf_results_df.to_csv(wf_csv, index=False)
    print(f"\n📁 Walk-forward saved: {wf_csv}")

    avg_oos_sharpe = wf_results_df['Sharpe Ratio'].mean()
    pct_positive_folds = (wf_results_df['Sharpe Ratio'] > 0).mean() * 100
    print(f"\n   Avg OOS Sharpe: {avg_oos_sharpe:.4f}")
    print(f"   % Positive Folds: {pct_positive_folds:.1f}%")

    if avg_oos_sharpe < 0.2 or pct_positive_folds < 50:
        print("\n   ⚠️ Walk-forward results weak. Consider different params.")
        print("   Proceeding to stress test anyway for diagnostics...")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: Stress Test (Delay +1 Interval)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"💪 PHASE 4: Stress Test (Execution Delay)")
    print(f"{'='*60}")

    stress_results = run_stress_test(full_strategy_df, interval=interval, max_delay=2, fee_rate=0.0002)
    print(stress_results[['delay_periods', 'Sharpe Ratio', 'Total Return (%)', 'Max Drawdown (%)', 'sharpe_decay_%']].to_string())

    # Save stress test results
    stress_csv = output_dir / "stress_test.csv"
    stress_results.to_csv(stress_csv, index=False)

    stress_passed = stress_results.loc[stress_results['delay_periods'] == 1, 'Sharpe Ratio'].iloc[0] > 0.3
    print(f"\n   {'✅ PASSED' if stress_passed else '❌ FAILED'} (Sharpe > 0.3 with +1 delay)")

    if not stress_passed:
        print("   ⚠️ Alpha does not survive execution delay. Likely not tradeable.")
        print("   Running Monte Carlo anyway for completeness...")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 5: Monte Carlo
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"🎲 PHASE 5: Monte Carlo ({mc_simulations} sims, {mc_noise*100:.1f}% noise)")
    print(f"{'='*60}")

    mc_summary, mc_sharpes = run_monte_carlo(
        base_df=full_df.copy(), params=selected_params,
        dir_factor=dir_factor, conf_factor=conf_factor, logic=logic,
        interval=interval, n_simulations=mc_simulations,
        noise_pct=mc_noise, fee_rate=0.0002
    )

    print(f"   Mean Sharpe:    {mc_summary['mean_sharpe']}")
    print(f"   Std Sharpe:     {mc_summary['std_sharpe']}")
    print(f"   Median Sharpe:  {mc_summary['median_sharpe']}")
    print(f"   5th-95th:       [{mc_summary['sharpe_5th_pct']}, {mc_summary['sharpe_95th_pct']}]")
    print(f"   % Profitable:   {mc_summary['pct_profitable']}%")

    # Save Monte Carlo results
    mc_csv = output_dir / "monte_carlo.csv"
    pd.DataFrame({'sharpe': mc_sharpes}).to_csv(mc_csv, index=False)

    # ═══════════════════════════════════════════════════════════════════
    # FINAL REPORT (save to file + print)
    # ═══════════════════════════════════════════════════════════════════
    report = {
        'strategy': f"{dir_factor} × {conf_factor} ({logic})",
        'symbol': symbol.upper(),
        'interval': interval,
        'selected_params': selected_params,
        'exit_config': config.EXIT_CONFIG,
        'sandbox_period': f"{config.OPTIMIZATION_START} → {config.OPTIMIZATION_END}",
        'full_period_sharpe': full_period_perf['Sharpe Ratio'],
        'full_period_return': full_period_perf['Total Return (%)'],
        'full_period_trades': full_period_perf['Total Trades'],
        'full_period_max_dd': full_period_perf['Max Drawdown (%)'],
        'wf_avg_sharpe': round(avg_oos_sharpe, 4),
        'wf_pct_positive_folds': round(pct_positive_folds, 1),
        'stress_delay1_sharpe': round(stress_results.loc[stress_results['delay_periods'] == 1, 'Sharpe Ratio'].iloc[0], 4),
        'stress_passed': stress_passed,
        'mc_mean_sharpe': mc_summary['mean_sharpe'],
        'mc_5th_pct': mc_summary['sharpe_5th_pct'],
        'mc_95th_pct': mc_summary['sharpe_95th_pct'],
        'mc_pct_profitable': mc_summary['pct_profitable'],
        'long_return_%': directional_perf['long_total_return_%'],
        'long_sharpe': directional_perf['long_sharpe'],
        'long_win_rate_%': directional_perf['long_win_rate_%'],
        'long_profit_factor': directional_perf['long_profit_factor'],
        'long_trades': directional_perf['long_total_trades'],
        'short_return_%': directional_perf['short_total_return_%'],
        'short_sharpe': directional_perf['short_sharpe'],
        'short_win_rate_%': directional_perf['short_win_rate_%'],
        'short_profit_factor': directional_perf['short_profit_factor'],
        'short_trades': directional_perf['short_total_trades'],
        'both_sides_profitable': directional_perf['both_sides_profitable'],
    }

    # Save report as JSON
    report_json = output_dir / "final_report.json"
    with open(report_json, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Print final report
    print(f"\n{'='*60}")
    print("📋 FINAL REPORT")
    print(f"{'='*60}")
    print(f"Strategy:          {report['strategy']}")
    print(f"Selected Params:   {report['selected_params']}")
    print(f"Exit Config:       {report['exit_config']}")
    print(f"Full Period SR:    {report['full_period_sharpe']}")
    print(f"Full Period Ret:   {report['full_period_return']}%")
    print(f"Full Period Trades:{report['full_period_trades']}")
    print(f"Full Period MDD:   {report['full_period_max_dd']}%")
    print(f"WF Avg Sharpe:     {report['wf_avg_sharpe']}")
    print(f"WF % Positive:     {report['wf_pct_positive_folds']}%")
    print(f"Stress +1 SR:      {report['stress_delay1_sharpe']}")
    print(f"MC Mean Sharpe:    {report['mc_mean_sharpe']}")
    print(f"MC 5th Pct:        {report['mc_5th_pct']}")
    print(f"Stress Test:       {'✅ PASS' if stress_passed else '❌ FAIL'}")
    print(f"\n📂 All results saved in: {output_dir}")
    print(f"{'='*60}")

    return {
        'grid_results': results_df,
        'selected_params': selected_params,
        'full_period_perf': full_period_perf,
        'wf_results': wf_results_df,
        'stress_results': stress_results,
        'stress_passed': stress_passed,
        'mc_summary': mc_summary,
        'output_dir': str(output_dir),
        'report': report,
    }


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ┌─────────────────────────────────────────────────────────────┐
    # │ STEP 1: Run Phase 1 only (grid search + heatmap)            │
    # │ Review the CSV and heatmap, then proceed to Step 2          │
    # └─────────────────────────────────────────────────────────────┘
    
    # Run this to test your initial assumption, symmetric threshold, and theoretically sound exit threshold
    # run_full_pipeline(
    #     symbol='meta',
    #     interval='1hr',
    #     dir_factor='BIAS_Q',
    #     conf_factor='VOLATILITY_FRICTION_RATIO_GREATER',
    #     logic='reverse',
    #     grid_mode='constrained'
    # )

    # Run this to test out asymmetric threshold
    # run_full_pipeline(
    #     symbol='meta',
    #     interval='1hr',
    #     dir_factor='BIAS_Q',
    #     conf_factor='VOLATILITY_FRICTION_RATIO_GREATER',
    #     logic='reverse',
    #     grid_mode='unconstrained'
    # )

    # ┌─────────────────────────────────────────────────────────────┐
    # │ STEP 2: After reviewing heatmap, pick stable params and     │
    # │ re-run with selected_params to trigger Phases 2.5-5         │
    # └─────────────────────────────────────────────────────────────┘

    run_full_pipeline(
        symbol='meta',
        interval='1hr',
        dir_factor='BIAS_Q',
        conf_factor='VOLATILITY_FRICTION_RATIO_GREATER',
        logic='reverse',
        selected_params={
            'n': 100,
            # 'dir_thresh_symmetric': 0.7,
            'dir_thresh_long':0.8,
            'dir_thresh_short':0.4,
            'conf_thresh': 0.7,
            'exit_threshold': 0.5,       # ← pick the value from your grid search CSV
        },
        mc_simulations=200,
        mc_noise=0.001
    )
