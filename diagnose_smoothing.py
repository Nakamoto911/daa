#!/usr/bin/env python3
"""Diagnostic: sweep EWM smoothing halflife values on cached raw probabilities.

Loads cached JM-XGB results (probs_raw), applies different halflife values
post-hoc, and reports Bear%, Shifts, Sharpe for each — no model retraining.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd

from replication import (ret_df, rf_daily, DATA_VERSION, ASSETS, SMOOTH,
                         _load_cache, _get_cache_suffix, ts, te)

PAPER = {
    'LargeCap': {'bear': 20.9, 'shifts': 46, 'sharpe': 0.79},
    'MidCap':   {'bear': 0, 'shifts': 0, 'sharpe': 0},
    'SmallCap': {'bear': 0, 'shifts': 0, 'sharpe': 0},
    'REIT':     {'bear': 18.4, 'shifts': 46, 'sharpe': 0.56},
    'AggBond':  {'bear': 41.5, 'shifts': 97, 'sharpe': 0.67},
    'Treasury': {'bear': 0, 'shifts': 0, 'sharpe': 0},
    'HighYield':{'bear': 0, 'shifts': 0, 'sharpe': 0},
    'Corporate':{'bear': 0, 'shifts': 0, 'sharpe': 0},
    'Commodity':{'bear': 0, 'shifts': 0, 'sharpe': 0},
    'Gold':     {'bear': 0, 'shifts': 0, 'sharpe': 0},
    'EAFE':     {'bear': 0, 'shifts': 0, 'sharpe': 0},
    'EM':       {'bear': 0, 'shifts': 0, 'sharpe': 0},
}

HALFLIFE_GRID = [0, 2, 4, 6, 8, 10, 12, 16, 21, 30]
TC = 5e-4

# Load cached JM-XGB results
jmxgb = {}
for nm in ASSETS:
    sfx = _get_cache_suffix(nm)
    key = f"jmxgb_{DATA_VERSION}_{nm}{sfx}_{ts}_{te}"
    c = _load_cache('models', key)
    if c is None and sfx:  # backward compat
        c = _load_cache('models', f"jmxgb_{DATA_VERSION}_{nm}_{ts}_{te}")
    if c is not None:
        jmxgb[nm] = c

if not jmxgb:
    print("ERROR: No cached results found. Run replication.py first.")
    exit(1)

print(f"Loaded {len(jmxgb)} assets from cache (DATA_VERSION={DATA_VERSION})")
print(f"Current SMOOTH: {SMOOTH}")

for nm in ['LargeCap', 'MidCap', 'SmallCap', 'REIT', 'AggBond', 'Treasury',
          'HighYield', 'Corporate', 'Commodity', 'Gold', 'EAFE', 'EM']:
    if nm not in jmxgb:
        print(f"\n  {nm}: not in cache, skipping")
        continue

    res = jmxgb[nm]
    probs_raw = res.get('probs_raw', pd.Series())
    if len(probs_raw) == 0:
        print(f"\n  {nm}: no raw probabilities available")
        continue

    ar = ret_df[nm].dropna()
    rf = rf_daily
    paper = PAPER[nm]
    cur_hl = SMOOTH.get(nm, 4)

    print(f"\n{'='*90}")
    print(f"  {nm}: Smoothing halflife sweep")
    print(f"  Paper: Bear={paper['bear']}%, Shifts={paper['shifts']}, Sharpe={paper['sharpe']}")
    print(f"  Current HL={cur_hl}")
    print(f"{'='*90}")
    print(f"  {'HL':>4s} {'Bear%':>7s} {'Shifts':>8s} {'Sharpe':>8s} {'SharpeT':>8s}"
          f" {'dBear':>8s} {'dShifts':>9s} {'dSharpe':>9s} {'Note':>8s}")
    print(f"  {'-'*75}")

    for hl in HALFLIFE_GRID:
        if hl > 0:
            smoothed = probs_raw.ewm(halflife=hl, min_periods=1).mean()
        else:
            smoothed = probs_raw.copy()

        fc = (smoothed >= 0.5).astype(int)
        ci = fc.index.intersection(ar.index)
        f_ = fc.reindex(ci)

        bear = (f_ == 1).mean() * 100
        shifts = int(f_.diff().abs().sum())

        pos = (f_ == 0).astype(float)
        r = ar.reindex(ci); rfv = rf.reindex(ci).fillna(0)
        sr = pos * r + (1 - pos) * rfv - pos.diff().abs().fillna(0) * TC
        # Excess-return Sharpe
        exc_sr = sr - rfv
        vol = exc_sr.std() * np.sqrt(252)
        sharpe = exc_sr.mean() * 252 / vol if vol > 0 else 0
        # Total-return Sharpe
        vol_t = sr.std() * np.sqrt(252)
        sharpe_t = sr.mean() * 252 / vol_t if vol_t > 0 else 0

        db = bear - paper['bear']
        ds = shifts - paper['shifts']
        dsh = sharpe - paper['sharpe']
        note = " <--cur" if hl == cur_hl else ""

        print(f"  {hl:4d} {bear:6.1f}% {shifts:8d} {sharpe:8.3f} {sharpe_t:8.3f}"
              f" {db:+7.1f}pp {ds:+8d} {dsh:+8.3f}{note}")
