#!/usr/bin/env python3
"""Quick test: run JM-XGB and JM-only for LargeCap + AggBond to verify regime detection."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, time, sys
import matplotlib; matplotlib.use('Agg')

# Import building blocks (does NOT run the full pipeline thanks to __name__ guard)
from replication import (process_asset, run_jm_only, exc_df, ret_df, rf_daily,
                         RF, MF, LG, ts, te, ASSETS)

print(f"\nTest period: {ts} -> {te}")
print(f"Lambda grid: {LG}")
print(f"Assets available: {[a for a in ASSETS if a in exc_df]}")

# --- Test 1: JM-XGB on LargeCap ---
print("\n" + "="*60)
print("TEST 1: JM-XGB on LargeCap")
print("="*60)
sys.stdout.flush()
t0 = time.time()
nm, res = process_asset('LargeCap', exc_df, ret_df, rf_daily, RF, MF, LG, ts, te)
print(f"  Time: {time.time()-t0:.0f}s")
print(f"  Best lambdas: {res['best_lambdas']}")
m = res['metrics']
print(f"  Sharpe={m['sharpe']:.3f}  MDD={m['mdd']*100:.1f}%  Bear={m['pct_bear']:.1f}%  Shifts={m['n_shifts']}")
print(f"\n  Paper: Bear=20.9%, Shifts=46, Sharpe=0.79")
print(f"  Delta: Bear={m['pct_bear']-20.9:+.1f}pp, Shifts={m['n_shifts']-46:+d}, Sharpe={m['sharpe']-0.79:+.2f}")
sys.stdout.flush()

# --- Test 2: JM-only on LargeCap ---
print("\n" + "="*60)
print("TEST 2: JM-only on LargeCap")
print("="*60)
sys.stdout.flush()
t0 = time.time()
nm2, res2 = run_jm_only('LargeCap', exc_df, ret_df, rf_daily, RF, LG, ts, te)
print(f"  Time: {time.time()-t0:.0f}s")
print(f"  Best lambdas: {res2.get('best_lambdas', {})}")
m2 = res2.get('metrics', {})
print(f"  Sharpe={m2.get('sharpe',0):.3f}  Shifts={m2.get('n_shifts',0)}  Bear={m2.get('pct_bear',0):.1f}%")
print(f"\n  Paper JM-only: Sharpe=0.59")
sys.stdout.flush()

# --- Test 3: AggBond (another reference asset) ---
print("\n" + "="*60)
print("TEST 3: JM-XGB on AggBond")
print("="*60)
sys.stdout.flush()
t0 = time.time()
nm3, res3 = process_asset('AggBond', exc_df, ret_df, rf_daily, RF, MF, LG, ts, te)
print(f"  Time: {time.time()-t0:.0f}s")
m3 = res3['metrics']
print(f"  Sharpe={m3['sharpe']:.3f}  MDD={m3['mdd']*100:.1f}%  Bear={m3['pct_bear']:.1f}%  Shifts={m3['n_shifts']}")
print(f"\n  Paper: Bear=41.5%, Shifts=97, Sharpe=0.67")
print(f"  Delta: Bear={m3['pct_bear']-41.5:+.1f}pp, Shifts={m3['n_shifts']-97:+d}, Sharpe={m3['sharpe']-0.67:+.2f}")
sys.stdout.flush()

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"  LargeCap: Bear={m['pct_bear']:.1f}% (paper 20.9%), Shifts={m['n_shifts']} (paper 46)")
print(f"  AggBond:  Bear={m3['pct_bear']:.1f}% (paper 41.5%), Shifts={m3['n_shifts']} (paper 97)")
print(f"  JM-only LargeCap: Sharpe={m2.get('sharpe',0):.3f} (paper 0.59)")

# Quick pass/fail
ok = True
if abs(m['pct_bear'] - 20.9) > 15: print("  WARN: LargeCap bear% off by >15pp"); ok = False
if abs(m['n_shifts'] - 46) > 50: print("  WARN: LargeCap shifts off by >50"); ok = False
if abs(m3['pct_bear'] - 41.5) > 15: print("  WARN: AggBond bear% off by >15pp"); ok = False
if abs(m3['n_shifts'] - 97) > 50: print("  WARN: AggBond shifts off by >50"); ok = False
if m['n_shifts'] > 200: print("  WARN: LargeCap too many shifts (>200)"); ok = False
if m3['n_shifts'] > 200: print("  WARN: AggBond too many shifts (>200)"); ok = False
print(f"\n  Overall: {'PASS' if ok else 'NEEDS WORK'}")
