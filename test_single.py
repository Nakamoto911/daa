#!/usr/bin/env python3
"""Quick test: run JM-XGB for LargeCap to verify regime detection."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, time, sys
import matplotlib; matplotlib.use('Agg')

# Import building blocks (does NOT run the full pipeline thanks to __name__ guard)
from replication import (process_asset, exc_df, ret_df, rf_daily,
                         RF, MF, LG, LG_FILTERED, LAM_FLOOR, ts, te, ASSETS)

print(f"\nTest period: {ts} -> {te}")
if LAM_FLOOR > 0:
    print(f"Lambda grid: {LG_FILTERED.tolist()} (floor={LAM_FLOOR}, paper: {LG.tolist()})")
else:
    print(f"Lambda grid: {LG.tolist()}")

# --- Test 1: JM-XGB on LargeCap ---
print("\n" + "="*60)
print("TEST 1: JM-XGB on LargeCap")
print("="*60)
sys.stdout.flush()
t0 = time.time()
nm, res = process_asset('LargeCap', exc_df, ret_df, rf_daily, RF, MF, LG_FILTERED, ts, te)
print(f"  Time: {time.time()-t0:.0f}s")
print(f"  Best lambdas: {res['best_lambdas']}")
m = res['metrics']
print(f"  Sharpe={m['sharpe']:.3f}  MDD={m['mdd']*100:.1f}%  Bear={m['pct_bear']:.1f}%  Shifts={m['n_shifts']}")
print(f"\n  Paper: Bear=20.9%, Shifts=46, Sharpe=0.79")
print(f"  Delta: Bear={m['pct_bear']-20.9:+.1f}pp, Shifts={m['n_shifts']-46:+d}, Sharpe={m['sharpe']-0.79:+.2f}")

# Diagnostic: raw vs smoothed shifts
probs_raw = res.get('probs_raw', pd.Series())
if len(probs_raw) > 0:
    ar = ret_df['LargeCap'].dropna()
    fc_raw = (probs_raw >= 0.5).astype(int)
    ci_raw = fc_raw.index.intersection(ar.index)
    f_raw = fc_raw.reindex(ci_raw)
    shifts_raw = int(f_raw.diff().abs().sum())
    bear_raw = (f_raw == 1).mean() * 100
    print(f"\n  Raw (unsmoothed): Bear={bear_raw:.1f}%, Shifts={shifts_raw}")
    print(f"  Smoothed (HL=8):  Bear={m['pct_bear']:.1f}%, Shifts={m['n_shifts']}")
    print(f"  Smoothing effect: {shifts_raw - m['n_shifts']:+d} shifts removed")

# Quick pass/fail
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
ok = True
if abs(m['pct_bear'] - 20.9) > 15: print("  WARN: LargeCap bear% off by >15pp"); ok = False
if abs(m['n_shifts'] - 46) > 50: print("  WARN: LargeCap shifts off by >50"); ok = False
if m['n_shifts'] > 200: print("  WARN: LargeCap too many shifts (>200)"); ok = False
print(f"  LargeCap: Bear={m['pct_bear']:.1f}% (paper 20.9%), Shifts={m['n_shifts']} (paper 46), Sharpe={m['sharpe']:.3f} (paper 0.79)")
print(f"\n  Overall: {'PASS' if ok else 'NEEDS WORK'}")
