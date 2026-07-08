# run_pipeline.py
"""
Master Pipeline — Config-driven alpha research workflow.
All you set here: symbol, interval, dir_factor, conf_factor, logic.
Everything else reads from config.py.

Phases:
  1. 3D Grid Search on Sandbox (2020-2022) → CSV + Heatmap
  2. Manual Selection (you pick params from heatmap + CSV)
  3. Walk-Forward Validation (12m IS + 3m OOS from 2023-01 → end)
  4. Stress Test (delay +1 interval)
  5. Monte Carlo (perturbed price paths)
"""
import pandas as pd
import numpy as np
import config

from engine.data_loader import DataLoader
from engine.backtest import Backtest
from engine.metrics import PerformanceMetrics
from engine.walk_forward import WalkForwardSplitter

from factors import bias_q, liquidity_premium_z, cci, turnover_mean_z, vwap_bias, ma_cross
from entry_exit_logic.two_factor_reverse import two_factor_reverse
from entry_exit_logic.two_factor_trend import two_factor_trend


# ─────────────────────────────────────────────────────────────────────
# SINGLE UNIVERSAL STRATEGY FUNCTION
# No dispatch tables. No per-combo functions. Just one function.
# ─────────────────────────────────────────────────────────────────────

# Simple name → signal function mapping
_DIR_FACTORS = {
    'BIAS_Q': bias_q.signal,
    'CCI': cci.signal,
    'VWAP_BIAS_Q': vwap_bias.signal,
    'MA_CROSS': ma_cross.signal,
}

_CONF_FACTORS = {
    'LIQUIDITY_PREMIUM_Z': liquidity_premium_z.signal,
    'TURNOVER_MEAN_Z': turnover_mean_z.signal,
}

_LOGIC_FUNCS = {
    'reverse': two_factor_reverse,
    'trend': two_factor_trend,
}


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
        if dir_factor in ['BIAS_Q', 'VWAP_BIAS_Q']:  # Rank-based (0 to 1)
            dir_short = 1.0 - dir_long
        elif dir_factor == 'MA_CROSS':  # Binary signal
            dir_short = -dir_long
        else:  # Z-score based (CCI etc.) — symmetric around 0
            dir_short = -dir_long
    else:
        dir_long = params.get('dir_thresh_long', 0.9)
        dir_short = params.get('dir_thresh_short', 0.1)

    exit_mode = params.get('exit_mode', config.EXIT_CONFIG['exit_mode'])
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
    )

    return df


# ─────────────────────────────────────────────────────────────────────
# GRID SEARCH (In-Process — No Pickling Issues)
# 135 runs takes ~8 seconds. No need for multiprocessing complexity.
# ─────────────────────────────────────────────────────────────────────

import itertools
import time


def run_grid_search(base_df, param_grid, dir_factor, conf_factor, logic, 
                    fee_rate=0.0002, interval='1hr'):
    """
    Simple in-process grid search. No pickling, no multiprocessing.
    135 runs at ~0.06s each = ~8 seconds total. Fast enough.
    """
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

        # Progress
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
    """
    Perturbs price paths → recomputes factors → different signals → different PnL.
    Tests if alpha is robust or path-dependent.
    """
    sharpe_results = []

    for i in range(n_simulations):
        rng = np.random.default_rng(seed=i)
        perturbed = base_df.copy()
        n_rows = len(perturbed)

        # Correlated noise to OHLC
        base_noise = rng.normal(0, noise_pct, n_rows)
        for col in ['open', 'high', 'low', 'close']:
            if col in perturbed.columns:
                ind_noise = rng.normal(0, noise_pct * 0.3, n_rows)
                perturbed[col] = perturbed[col] * (1 + base_noise + ind_noise)

        # Enforce OHLC constraints
        perturbed['high'] = perturbed[['open', 'high', 'low', 'close']].max(axis=1)
        perturbed['low'] = perturbed[['open', 'high', 'low', 'close']].min(axis=1)

        # Perturb volume/turnover
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
# HEATMAP GENERATION
# ─────────────────────────────────────────────────────────────────────

def generate_heatmap(results_df, dir_factor, conf_factor, logic):
    import matplotlib.pyplot as plt
    import seaborn as sns

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

        filename = f"heatmap_{dir_factor}_{conf_factor}_{logic}_thresh{thresh_val}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {filename}")


# ─────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────

def run_full_pipeline(
    symbol: str = 'spy',
    interval: str = '1hr',
    dir_factor: str = 'BIAS_Q',
    conf_factor: str = 'LIQUIDITY_PREMIUM_Z_LESSER',
    logic: str = 'reverse',
    mc_simulations: int = 100,
    mc_noise: float = 0.001,
    # Phase 3-5 params (set after you review Phase 1 heatmap)
    selected_params: dict = None,  # Manually chosen from heatmap
):
    """
    Config-driven pipeline. Set your hypothesis above and run.
    
    Workflow:
      Phase 1: Grid search → CSV + Heatmap → YOU review & pick params
      Phase 2: (Manual) Review heatmap, pick stable params, re-run with selected_params
      Phase 3: Walk-Forward (12m IS + 3m OOS, rolling monthly from 2023-01)
      Phase 4: Stress Test (delay +1)
      Phase 5: Monte Carlo (perturbed paths)
    """
    print("=" * 60)
    print(f"🚀 PIPELINE: {dir_factor} × {conf_factor} ({logic})")
    print(f"   Symbol: {symbol.upper()} | Interval: {interval}")
    print("=" * 60)

    # Load full dataset once
    loader = DataLoader(data_dir="data/raw")
    full_df = loader.load(symbol=symbol, interval=interval)
    print(f"   Loaded {len(full_df)} bars ({full_df.index.min()} → {full_df.index.max()})")

    # Build grid from config
    target_grid = config.CONSTRAINED_GRID
    grid = {**target_grid['DIRECTIONAL'][dir_factor], **target_grid['CONFIRMATION'][conf_factor]}

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: Grid Search on Sandbox (2020-2022)
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

    # Save results CSV (you inspect this manually)
    output_csv = f"grid_{dir_factor}_{conf_factor}_{logic}.csv"
    results_df.sort_values('Sharpe Ratio', ascending=False).to_csv(output_csv, index=False)
    print(f"\n📁 Grid results saved: {output_csv}")
    print(f"   Columns: {list(results_df.columns)}")

    # Generate heatmap
    generate_heatmap(results_df, dir_factor, conf_factor, logic)

    # Show top results for reference
    print(f"\n🏆 Top 10 (for reference — check heatmap for stability):")
    top_cols = [c for c in ['n', 'dir_thresh_symmetric', 'conf_thresh', 'conf_mode',
                            'Sharpe Ratio', 'Total Return (%)', 'Max Drawdown (%)', 
                            'Total Trades'] if c in results_df.columns]
    print(results_df.sort_values('Sharpe Ratio', ascending=False).head(10)[top_cols].to_string())

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: GATE — Do we have selected params to proceed?
    # ═══════════════════════════════════════════════════════════════════
    if selected_params is None:
        print(f"\n{'='*60}")
        print("⏸️  PHASE 1 COMPLETE — Review the heatmap & CSV.")
        print("   Once you've picked stable params, re-run with:")
        print(f"   run_full_pipeline(..., selected_params={{'n': X, 'dir_thresh_symmetric': Y, 'conf_thresh': Z, 'conf_mode': 'lesser'}})")
        print(f"{'='*60}")
        return {'grid_results': results_df}

    print(f"\n{'='*60}")
    print(f"📈 PHASE 3: Walk-Forward Validation")
    print(f"   Selected params: {selected_params}")
    print(f"   12m IS + 3m OOS, step=1 month, starting 2023-01-01")
    print(f"{'='*60}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: Walk-Forward (2023-01 → end of data)
    # ═══════════════════════════════════════════════════════════════════
    wf_df = full_df.loc['2023-01-01':].copy()
    splitter = WalkForwardSplitter(wf_df, is_months=12, oos_months=3, step_months=1)

    wf_results = []
    for fold in splitter.get_slices():
        oos_data = fold['test_set'].copy()

        # Apply strategy on OOS data (no sandbox slice)
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

    avg_oos_sharpe = wf_results_df['Sharpe Ratio'].mean()
    pct_positive_folds = (wf_results_df['Sharpe Ratio'] > 0).mean() * 100
    print(f"\n   Avg OOS Sharpe: {avg_oos_sharpe:.4f}")
    print(f"   % Positive Folds: {pct_positive_folds:.1f}%")

    # Gate check
    if avg_oos_sharpe < 0.2 or pct_positive_folds < 50:
        print("\n   ⚠️ Walk-forward results weak. Consider different params.")
        print("   Proceeding to stress test anyway for diagnostics...")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 4: Stress Test (Delay +1 Interval)
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"💪 PHASE 4: Stress Test (Execution Delay)")
    print(f"{'='*60}")

    # Apply strategy on full data for stress/MC
    full_strategy_df = apply_strategy(full_df.copy(), selected_params, dir_factor, conf_factor, logic, slice_to_sandbox=False)

    stress_results = run_stress_test(full_strategy_df, interval=interval, max_delay=2, fee_rate=0.0002)
    print(stress_results[['delay_periods', 'Sharpe Ratio', 'Total Return (%)', 'Max Drawdown (%)', 'sharpe_decay_%']].to_string())

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

    # ═══════════════════════════════════════════════════════════════════
    # FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("📋 FINAL REPORT")
    print(f"{'='*60}")
    print(f"Strategy:        {dir_factor} × {conf_factor} ({logic})")
    print(f"Selected Params: {selected_params}")
    print(f"Sandbox Sharpe:  {results_df[results_df['n']==selected_params.get('n')].sort_values('Sharpe Ratio', ascending=False).iloc[0]['Sharpe Ratio']:.4f}" if 'n' in selected_params else "N/A")
    print(f"WF Avg Sharpe:   {avg_oos_sharpe:.4f}")
    print(f"WF % Positive:   {pct_positive_folds:.1f}%")
    print(f"Stress +1 SR:    {stress_results.loc[stress_results['delay_periods']==1, 'Sharpe Ratio'].iloc[0]:.4f}")
    print(f"MC Mean Sharpe:  {mc_summary['mean_sharpe']}")
    print(f"MC 5th Pct:      {mc_summary['sharpe_5th_pct']}")
    print(f"Stress Test:     {'✅ PASS' if stress_passed else '❌ FAIL'}")
    print(f"{'='*60}")

    return {
        'grid_results': results_df,
        'selected_params': selected_params,
        'wf_results': wf_results_df,
        'stress_results': stress_results,
        'stress_passed': stress_passed,
        'mc_summary': mc_summary,
    }


# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    # ┌─────────────────────────────────────────────────────────────┐
    # │ STEP 1: Run Phase 1 only (grid search + heatmap)            │
    # │ Review the CSV and heatmap, then proceed to Step 2          │
    # └─────────────────────────────────────────────────────────────┘
    
    # run_full_pipeline(
    #     symbol='spy',
    #     interval='1hr',
    #     dir_factor='MA_CROSS',
    #     conf_factor='TURNOVER_MEAN_Z_GREATER',
    #     logic='trend',
    # )
    
    # ┌─────────────────────────────────────────────────────────────┐
    # │ STEP 2: After reviewing heatmap, pick stable params and     │
    # │ re-run with selected_params to trigger Phases 3-5           │
    # └─────────────────────────────────────────────────────────────┘
    
    run_full_pipeline(
        symbol='spy',
        interval='1hr',
        dir_factor='MA_CROSS',
        conf_factor='TURNOVER_MEAN_Z_GREATER',
        logic='trend',
        selected_params={'n': 150, 'dir_thresh_symmetric': 1, 'conf_thresh': 2, 'conf_mode': 'greater'},
        mc_simulations=200,
        mc_noise=0.001
    )