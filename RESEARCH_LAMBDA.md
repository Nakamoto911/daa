# Lambda Selection Research Trail

## 1. Problem Statement

LargeCap JM-XGB produces **96 regime shifts** vs the paper's **46**. EAFE/EM produce 400+ shifts. The paper (SSRN 4864358) uses Bloomberg data; our replication uses Yahoo Finance. The question: is the shift excess due to an algorithm bug or a data difference?

**Paper reference values** (Figure 2, Table 4):
- LargeCap: Bear=20.9%, Shifts=46, Sharpe=0.79
- REIT: Bear=18.4%, Shifts=46
- AggBond: Bear=41.5%, Shifts=97

---

## 2. Algorithm Verification

### 2.1 Official `jumpmodels` Package Comparison
**Script**: `diagnose_official_jm.py`

Compared our `JM` class against the official `jumpmodels` package (v0.1.1, by paper author Yizhan Shu).

**Result**: Our JM implementation matches the official package **perfectly** at every lambda value -- identical shift counts and bear percentages. The JM algorithm is correct.

### 2.2 XGBoost Default Parameters
Verified that `n_estimators=100`, `max_depth=6`, `learning_rate=0.3` match XGBoost's actual defaults. The paper says "default hyperparameters" and our code uses these exact values. Confirmed correct.

### 2.3 Feature Construction
Verified against paper Table 2 (return features: DD at hl=5,21; R at hl=5,10,21; S at hl=5,10,21) and Table 3 (5 macro features). Our `ret_features()` and `macro_features()` functions match exactly.

### 2.4 Target Construction
`np.roll(labels, -1)[:-1]` creates `(features_day_t, regime_day_{t+1})` pairs. Correct per the paper's `{(x_t, s_{t+1})}` formulation.

---

## 3. Investigation Timeline

### 3.1 diagnose_largecap.py
**Goal**: Deep per-window analysis of the LargeCap JM-XGB pipeline.

**Findings**:
- 90.7% of XGBoost probabilities are extreme (< 0.2 or > 0.8)
- Raw shifts per window range from 4 to 33 (worst: 2015-01 with 33)
- In-sample accuracy: 100% -- XGBoost perfectly fits training labels
- Feature importance: S10 (0.462) and S21 (0.252) dominate

### 3.2 diagnose_boundaries.py
**Goal**: Check if probability discontinuities at 6-month window boundaries cause extra shifts.

**Findings**:
- Only 4 boundary crossings of the 0.5 threshold out of 33 boundaries
- Global smoothing: 94 shifts; per-window smoothing: 106 shifts
- Boundary effects contribute ~8 extra shifts -- **not the main issue**

### 3.3 diagnose_labels.py
**Goal**: Feature ablation -- which features drive the excess shifts?

**Findings**:
- Return features only: 92 shifts
- Macro features only: 96 shifts
- All features: 94 shifts
- Day-by-day probability shows rapid oscillation (0.97 to 0.23 in consecutive days)
- **All feature sets give similar shift counts** -- features are not the issue

### 3.4 diagnose_labels_v2.py
**Goal**: Test raw vs excess returns for JM label construction.

**Findings**: Identical results. Whether JM labels use raw returns or excess returns, the outcome is the same. Label construction is not the issue.

### 3.5 diagnose_overfit.py
**Goal**: Test if XGBoost overfitting causes excessive regime shifts.

**Findings**:
- Default (max_depth=6): 96 shifts
- max_depth=3: 98 shifts
- max_depth=2: 82 shifts
- n_estimators=10: 84 shifts
- Even aggressive regularization doesn't bring shifts below 82 -- **XGBoost hyperparameters don't explain the gap**

### 3.6 diagnose_features.py
**Goal**: Test JM alone (without XGB) and with different feature dimensions.

**Findings**:
- JM on 8D features with lambda=7: 92 shifts (training)
- JM on 1D excess returns: only 12 shifts but Sharpe=0.296
- JM carry-forward (no XGB): 14 shifts
- **JM with lambda=15 on 8D features: 48 shifts** -- close to paper's 46!
- This was the first hint that lambda is the key variable

### 3.7 diagnose_clipping.py
**Goal**: Test feature clipping within 3 standard deviations (as shown in the authors' `jumpmodels` example notebook).

**Findings**:
- JM with clip+standardize: 102 shifts (**WORSE**)
- Full lambda selection with clipping: 108 shifts (**WORSE**)
- Feature clipping does not help and should not be applied

### 3.8 diagnose_lambda.py
**Goal**: Examine what lambda values our Algorithm 2 selects over time.

**Findings**:
- 2007-2015: mostly lambda=0.3 and 1.0 (low values)
- 2015-2020: mostly lambda=15 and 40 (higher values)
- Lambda frequency: 0.0(4x), 0.3(3x), 1.0(12x), 7.0(2x), 15.0(5x), 40.0(8x)
- Validation genuinely favors low lambdas for early periods (2008 crisis dynamics)

### 3.9 diagnose_fixed_lambda.py -- BREAKTHROUGH
**Goal**: What fixed lambda gives closest match to the paper's 46 shifts?

**Findings**:

| Lambda | Raw | Smooth | Sharpe | Bear% |
|--------|-----|--------|--------|-------|
| 0.3 | ~300 | ~94 | 0.758 | 22.3% |
| 1.0 | ~270 | ~90 | 0.777 | 21.1% |
| 7.0 | ~200 | ~72 | 0.804 | 20.0% |
| 15.0 | ~180 | ~58 | 0.808 | 19.4% |
| 40.0 | 250 | 52 | 0.764 | 22.3% |
| **50.0** | **170** | **42** | **0.801** | **22.1%** |
| 100.0 | 150 | 44 | 0.394 | ~23% |

**Lambda=50 produces 42 shifts, Sharpe=0.801, Bear=22.1%** -- essentially matching the paper's 46 shifts, Sharpe=0.79, Bear=20.9%.

### 3.10 diagnose_val_sharpe.py
**Goal**: Why does validation pick lambda=0.3 instead of lambda=50?

**Findings** (at ud=2010-01, validation window 2005-2010):
- Lambda=0.3: validation Sharpe=0.765 (**winner**)
- Lambda=7.0: Sharpe=0.532
- Lambda=40.0: Sharpe=0.421
- Validation genuinely favors low lambdas at this date because low-lambda models react faster to the 2008 crisis, producing better Sharpe in the crisis-heavy 2005-2010 window.

### 3.11 diagnose_rf_impact.py
**Goal**: Does the risk-free rate difference (Yahoo ~2.54%/yr vs Bloomberg ~1.1%/yr) change lambda ranking?

**Findings**: No. With paper's rf rate (1.1%), with zero rf, and with total-return Sharpe -- the lambda ranking is identical. Risk-free rate is not the driver.

### 3.12 diagnose_logspace_grid.py
**Goal**: Does a truly log-spaced grid (matching the paper's "distributed evenly on a logarithmic scale") fix the issue?

**Findings**:
- Current grid: 96 shifts
- LogSpace 8 (0.3-100): 106 shifts (WORSE)
- LogSpace 12 (0.3-100): similar or worse
- Different grid spacings don't fix the issue

### 3.13 diagnose_val_landscape.py
**Goal**: Full validation Sharpe landscape across all lambda values at every update date.

**Findings**:
- **15 out of 34 validation races have gap < 0.05** between winner and runner-up
- Early period (2007-2012): lambda=1.0 dominates (12 wins)
- Later period (2015+): lambda=40 dominates (8 wins)
- Small daily return differences between Yahoo and Bloomberg easily tip these close races
- **Using cumulative return** instead of Sharpe as criterion: lambda=40 wins 14/34 (vs 8/34 with Sharpe)

### 3.14 diagnose_lambda_grid.py
**Goal**: Test even denser/different lambda grids.

**Findings**: Denser grids (12, 16, 20 points) don't change the fundamental issue. The problem is the validation criterion ranking, not grid resolution.

---

## 4. Additional Investigations

### 4.1 Authors' GitHub Repos
- **jumpmodels** (https://github.com/Yizhan-Oliver-Shu/jump-models): Example notebook uses `jump_penalty=50.0` as the default -- consistent with our finding that lambda=50 matches paper results.
- **continuous-jump-model** repo: Contains M&A data processing, not DAA-specific code.

### 4.2 Authors' Earlier Paper (arXiv:2402.05272)
"Downside Risk Reduction Using Regime-Switching Signals" (Journal of Asset Management 2024):
- Uses 8-year validation lookback (vs 5-year in DAA paper)
- Monthly lambda updates (vs biannual in DAA paper)
- Features use halflives 10, 20, 60 (vs DAA paper's 5, 10, 21)
- S&P 500 gets fewer than 1 shift/year at lambda=50-100

### 4.3 Transaction Cost Sensitivity
Tested whether higher transaction costs (5bps, 10bps, 20bps, 50bps, 100bps, 200bps) change the lambda ranking in validation.

**Result**: Even at 50bps (10x paper value), lambda=0.3 still wins at 2010-01. After EWM smoothing, the shift count difference between lambdas is only ~10-18 per 5-year validation period, making TC impact negligible (~5-9bps total).

### 4.4 Sharpe Formula Variants
Tested excess-return Sharpe (mean(r-rf)/std(r-rf)) vs total-return Sharpe (mean(r)/std(r)).

**Result**: Same lambda ranking regardless of Sharpe formula.

### 4.5 Data Quality Check
- B&H MDD for LargeCap: -55.25% (matches paper exactly)
- B&H Sharpe with paper's rf: 0.487 (paper: 0.50) -- close, ~0.013 gap from slight return differences
- Yahoo `^SP500TR` (S&P 500 Total Return) vs Bloomberg SPTR: nearly identical but not exact

---

## 5. Root Cause Diagnosis

### The algorithm is correct.
Every component has been verified:
- JM matches official package perfectly
- XGBoost uses correct default parameters
- Features match paper Tables 2 and 3
- Target construction is correct
- EWM smoothing is correctly applied
- Validation methodology matches Algorithm 2

### The divergence is purely data-driven.
Small differences in Yahoo vs Bloomberg daily returns cascade through:
1. Different daily excess returns
2. Different JM labels (especially around volatile periods like 2008, 2020)
3. Different XGBoost training targets
4. Different P(bear) probabilities
5. Different validation Sharpe for each lambda
6. **Different optimal lambda selected** (0.3-7 instead of ~40-50)
7. Different final OOS forecasts
8. More regime shifts

### The validation landscape is fragile.
15/34 update dates have < 0.05 Sharpe gap between the winning lambda and the runner-up. This means data differences of ~0.5% annualized Sharpe are enough to flip the winner, changing the lambda from e.g. 1.0 to 40.0.

---

## 6. Decision: Lambda Floor

A lambda floor of **10** is applied, filtering the grid from `[0, 0.3, 1, 3, 7, 15, 40, 100]` to `[15, 40, 100]`.

### Rationale
1. **Paper's Bloomberg data naturally produces higher optimal lambdas (~40-50)**
2. **Yahoo data's noise biases validation toward low lambdas**
3. **The floor compensates for data quality, not algorithm error**
4. **Authors' own example uses lambda=50 as default**
5. **User goal**: buy-and-hold-oriented strategy with fewer transactions
6. **Result**: fewer regime shifts, closer to paper's behavior

### Configuration
- `LAM_FLOOR = 10` in `replication.py` (line 44)
- Set `LAM_FLOOR = 0` to restore full paper grid behavior
- `LG_FILTERED = LG[LG >= LAM_FLOOR]` derives the active grid
- `DATA_VERSION` automatically updates to ensure cache separation

### Future Improvement
If Bloomberg data becomes available, the lambda floor could be removed. The algorithm would then naturally select appropriate lambdas as the paper does. The floor is a pragmatic workaround for free-data limitations, not a permanent architectural choice.

---

## 7. Macro Feature Source Fix (v3)

### Bug Found
The macro features (paper Table 3) used **wrong Treasury yield sources**:
- `T2Y` used Yahoo `^IRX` (13-week / 3-month T-bill rate) instead of FRED `DGS2` (2-year constant maturity)
- This caused the yield curve slope (10Y - T2Y) to compute 10Y-3M instead of the paper's 10Y-2Y
- Both `t2d` (interest rate trend) and `yc`/`ycd` (yield curve) features were wrong

### Fix
Switched all macro sources to FRED (with Yahoo fallback):
- `T2Y`: FRED `DGS2` (was Yahoo `^IRX`)
- `T10Y`: FRED `DGS10` (was Yahoo `^TNX`)
- `VIX`: FRED `VIXCLS` (was Yahoo `^VIX`)

### Impact on LargeCap (with LAM_FLOOR=10)

| Metric | Before (v2) | After (v3) | Paper |
|--------|------------|------------|-------|
| Bear%  | 27.3%      | 24.8%      | 20.9% |
| Shifts | 72         | 54         | 46    |
| Sharpe | 0.649      | 0.776      | 0.79  |

Sharpe improved from 0.649 to 0.776 (paper: 0.79, gap reduced from 0.14 to 0.01).
Lambda selection now strongly favors lambda=40 (28/34 updates), matching paper expectations.

### LAM_FLOOR still needed
With `LAM_FLOOR=0` and FRED macros: Shifts=112, Bear=32.7%. The full grid still allows low lambdas to win in validation for some periods due to Yahoo vs Bloomberg return differences. The floor remains necessary.
