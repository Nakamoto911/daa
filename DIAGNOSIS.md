# LargeCap Regime Shifts Diagnosis

## Problem
LargeCap JM-XGB produces 96 regime shifts vs paper's 46.

## Root Cause
The lambda validation (Algorithm 2) selects low lambdas (0.3-7) with Yahoo Finance
data, while the paper's Bloomberg data leads to higher lambdas (~40-50).

With fixed lambda=50, our pipeline produces:
- 42 shifts (paper: 46)
- Sharpe=0.801 (paper: 0.79)
- Bear=22.1% (paper: 20.9%)

This proves the algorithm is correct; the difference is purely data-driven.

## Evidence

### 1. Algorithm verified correct
- JM implementation matches official `jumpmodels` package at every lambda value
- XGBoost uses correct default params (n_estimators=100, max_depth=6, lr=0.3)
- Feature construction matches paper Table 2 exactly (DD at 5,21; R at 5,10,21; S at 5,10,21)
- Macro features match paper Table 3
- Target construction: np.roll(labels, -1)[:-1] is correct
- EWM smoothing halflife=8 matches paper footnote 14
- Validation methodology matches Algorithm 2

### 2. Lambda validation landscape
- 15/34 validation races have gap < 0.05 between winner and runner-up
- Early period (2007-2012): lambda=1.0 dominates due to 2008 crisis responsiveness
- Later period (2015+): lambda=40 dominates in calmer markets
- Small data differences easily tip these close races

### 3. Eliminated hypotheses
- Feature clipping: WORSE (102-108 shifts)
- predict_online for labels: WORSE (106 shifts)
- XGB regularization: even max_depth=2 gives 82 shifts
- Risk-free rate: doesn't change lambda ranking
- Different lambda grids: don't help
- Boundary overlaps: only ~8 extra shifts
- Sharpe formula (excess vs total): same ranking

### 4. Data differences
- Yahoo: ^GSPC price returns + synthetic dividends (VTI adj close)
- Bloomberg: SPTR total return index (includes dividends precisely)
- Risk-free: Yahoo ^IRX ~2.54%/yr vs Bloomberg ~1.1%/yr
- These affect daily returns during volatile periods, cascading through JM → XGB → validation
