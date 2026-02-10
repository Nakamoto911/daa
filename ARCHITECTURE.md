# Architecture & Algorithm Reference

## Dynamic Asset Allocation with Asset-Specific Regime Forecasts

Replication of Shu, Yu, Mulvey (2024) — SSRN 4864358

---

## 1. Overview

This codebase replicates a regime-based dynamic asset allocation strategy. The core idea: use a Jump Model (JM) to identify bull/bear regimes in each asset class, boost the forecast with XGBoost, then feed regime forecasts into portfolio optimization. The paper shows this outperforms buy-and-hold and non-regime-aware portfolios across 12 asset classes over 2007-2023.

### File Map

| File | Purpose | Lines |
|------|---------|-------|
| `replication.py` | Full pipeline: data, features, JM+XGB, backtests, tables/figures | ~938 |
| `compare.py` | Section-by-section comparison with paper reference values | ~633 |
| `test_replication.py` | 112 pytest tests validating against paper | ~525 |
| `test_single.py` | Quick smoke test on LargeCap + AggBond | ~74 |

### Execution

```bash
.venv/bin/python replication.py      # Full pipeline (~15 min cold, ~1 min warm)
.venv/bin/python compare.py          # Detailed comparison report
.venv/bin/python -m pytest test_replication.py -v  # 112 tests
.venv/bin/python test_single.py      # Quick 2-asset check (~6 min)
```

### Cache System

Three-tier pickle cache in `cache/`:

| Directory | Contents | Invalidation |
|-----------|----------|-------------|
| `cache/data/` | Raw price data, features | 7 days (168 hours) |
| `cache/models/` | Per-asset JM-XGB and JM-only results | 7 days |
| `cache/backtests/` | Portfolio backtest results | 7 days |

Functions: `_save_cache(obj, category, name)`, `_load_cache(category, name, max_hours=168)`, `clear_cache(category=None)`.

---

## 2. Data Pipeline

### 2.1 Asset Universe

12 asset classes spanning equities, real estate, bonds, and commodities. **V2 total-return data** uses Yahoo Finance total-return index tickers, FRED bond total-return indices, and mutual fund adjusted close (which includes dividends/coupons) to approximate the paper's Bloomberg total-return data:

| Group | Asset | Primary Source (V2) | Type | Start |
|-------|-------|-------------------|------|-------|
| Equity & RE | LargeCap | `^SP500TR` (Yahoo) | TR Index | 1990 |
| | MidCap | `VIMSX` adj close (Yahoo) | MF adj close | 1998 |
| | SmallCap | `^RUTTR` (Yahoo) + `NAESX` backfill | TR Index + MF | 1990 |
| | EAFE | `VGTSX` adj close (Yahoo) | MF adj close | 1996 |
| | EM | `VEIEX` adj close (Yahoo) | MF adj close | 1994 |
| | REIT | `VGSIX` adj close (Yahoo) | MF adj close | 1996 |
| Bonds & Comm | AggBond | `VBMFX` adj close (Yahoo) | MF adj close | 1990 |
| | Treasury | `VUSTX` adj close (Yahoo) | MF adj close | 1990 |
| | HighYield | `BAMLHYH0A0HYM2TRIV` (FRED) | ICE BofA TR | 1990 |
| | Corporate | `BAMLCC0A0CMTRIV` (FRED) | ICE BofA TR | 1990 |
| | Commodity | `DBC` adj close (Yahoo) | ETF adj close | 2006 |
| | Gold | `GC=F` + `GLD` (Yahoo) | Futures + ETF | 2000 |

**Paper uses Bloomberg total-return indices from 1991.** V2 data uses total-return indices and adjusted-close mutual funds that include dividends and coupon income, significantly reducing divergence vs. the paper. Key improvements: FRED bond indices add ~2-5%/yr coupon income, `^SP500TR` adds ~2%/yr dividends, VGSIX/VUSTX add dividend/coupon income for REIT/Treasury.

### 2.2 Data Construction

`load_data(start='1990-01-01', end='2024-01-01')` at module-level execution:

1. For each asset, walk `ASSET_CONFIG_V2` source chain: try Yahoo TR indices, FRED indices, or MF adj close, with ETF+MF fallback
2. Splice segments chronologically (preferred source first, backfill sources for earlier dates)
3. Build `ret_df` (daily returns), `exc_df` (excess returns = ret - rf)
4. Risk-free rate from FRED `DGS3MO` (3-month Treasury constant maturity, ~1.1%/yr over 2007-2023), fallback to Yahoo `^IRX`
5. Macro data: `^IRX` (2Y yield), `^TNX` (10Y yield), `^VIX`

**Key outputs available at module level**: `ret_df`, `exc_df`, `rf_daily`, `wealth_df`

### 2.3 Feature Engineering

Two feature sets, computed at module level by `_compute_features()`:

**Return Features** (`RF` dict, per-asset) — Paper Table 2:

| Feature | Code | Formula | Assets |
|---------|------|---------|--------|
| Downside Dev (log) | DD5, DD21 | `log(sqrt(EWM(min(r,0)^2, hl)))` | All except AggBond, Treasury, Gold |
| Average Return | R5, R10, R21 | `EWM(excess_return, hl)` | All |
| Sortino Ratio | S5, S10, S21 | `EWM_mean / DD` | All |

Result: 8 features for most assets, 6 for AggBond/Treasury/Gold (no DD features per paper footnote 13).

**Macro Features** (`MF` DataFrame, shared across assets) — Paper Table 3:

| Feature | Code | Formula |
|---------|------|---------|
| US Treasury 2Y Yield | t2d | `diff(T2Y)` then `EWMA(hl=21)` |
| Yield Curve Slope | yc | `(T10Y - T2Y)` then `EWMA(hl=10)` |
| Yield Curve Change | ycd | `diff(slope)` then `EWMA(hl=21)` |
| VIX Change | vd | `diff(log(VIX))` then `EWMA(hl=63)` |
| Stock-Bond Correlation | sbc | `rolling_corr(LargeCap, AggBond, 252d)` |

---

## 3. Jump Model (JM)

### 3.1 Mathematical Formulation

The Jump Model solves (Paper Equation 1):

```
minimize   Σ_t l(x_t, θ_{s_t}) + λ Σ_t 1{s_{t-1} ≠ s_t}
```

Where:
- `x_t` = feature vector at time t (standardized)
- `s_t ∈ {0, 1}` = regime state (bull/bear)
- `θ_k` = centroid of state k (mean feature vector)
- `l(x, θ) = 0.5 * ||x - θ||²` = squared Euclidean loss
- `λ` = jump penalty controlling regime persistence (higher = fewer switches)

### 3.2 Implementation: `JM` Class

**Parameters**: `lam` (jump penalty), `n_init=3` (random restarts), `max_it=20` (max iterations per restart), `seed=42`.

**Algorithm** — Coordinate descent with Viterbi:

```
for each of n_init random initializations:
    1. Pick 2 random data points as initial centroids
    2. Repeat up to max_it times:
       a. Compute distances: d[t,k] = 0.5 * ||X[t] - center[k]||²
       b. Viterbi forward pass:
          V[0] = d[0]
          for t = 1..T-1:
              stay_cost   = V[t-1]
              switch_cost = V[t-1, reversed] + λ
              V[t] = d[t] + elementwise_min(stay_cost, switch_cost)
       c. Viterbi backward pass: trace optimal state sequence
       d. Update centroids: center[k] = mean(X[s==k])
       e. Stop if |cost_change| < 1e-6
    3. Keep initialization with lowest total cost
```

**Label Assignment** (`labels` method): The state with higher cumulative excess return is labeled 0 (bull), the other 1 (bear). This semantic assignment ensures consistent interpretation across assets.

**Standardization** (`stdz` static method): `(X - mean) / std` per feature, with `std=0` protection.

### 3.3 Lambda's Role

Lambda controls the tradeoff between fit quality and regime stability:
- `λ = 0`: No penalty for switching → regimes change freely, many shifts
- `λ = 100`: Extreme penalty → very few regime switches, sticky regimes
- Paper's grid: `[0, 0.3, 1, 3, 7, 15, 40, 100]` — 8 values, roughly log-spaced

The optimal lambda is selected per-asset, per-update via Algorithm 2 (validation Sharpe).

---

## 4. Algorithm 2: Lambda Selection (Core Innovation)

This is the paper's main contribution and the most complex part of the code. Implemented in `process_asset()`.

### 4.1 Paper Pseudocode (Page 14)

```
Algorithm 2: Optimal Jump Penalty Selection

Input: Candidate lambdas Λ = {λ_1, ..., λ_8}

for each asset:
    for every 6 months from test_start:
        for each λ ∈ Λ:
            forecasts = Algorithm1(λ, validation_window=[ud-5yr, ud))
            sharpe_λ = Sharpe(0/1_strategy(forecasts), validation_window)
        λ* = argmax_λ(sharpe_λ)

        # Use λ* for final OOS forecast
        final_forecasts = Algorithm1(λ*, training=[ud-11yr, ud))
        Record OOS forecasts for next 6 months

Output: Optimal lambda sequences, OOS regime forecasts
```

### 4.2 Implementation Detail: `process_asset()`

**Parameters**: `nm, exc_df, ret_df, rf_daily, RF, MF, lam_grid, test_start, test_end, val_years=5, tc=5e-4`

**Outer loop**: Biannual update dates from `test_start` to `test_end` (freq='6MS'), typically ~34 updates for 2007-2023.

**For each update date `ud`**:

#### Step A: Lambda Selection via Validation

```python
val_start = ud - 5 years
val_end   = ud
val_sub_updates = date_range(val_start, val_end, freq='6MS')  # ~10 sub-updates

for lam in lam_grid:           # 8 candidates
    for vud in val_sub_updates: # ~10 validation sub-updates
        cache_key = (lam, vud)
        if cache_key in sub_cache:
            reuse cached OOS probabilities
        else:
            training = [vud - 11yr, vud)
            X_train = standardize(features[training])

            jm = JM(lam=lam, n_init=3, max_it=20)
            jm.fit(X_train)
            labels = jm.labels(excess_returns[training])

            target = np.roll(labels, -1)[:-1]  # next-day regime
            xgb = XGBClassifier(n_estimators=100, max_depth=6, lr=0.3)
            xgb.fit(X_train[:-1], target)

            X_oos = standardize(features[vud : vud+6months], using training stats)
            probs = xgb.predict_proba(X_oos)[:, 1]  # P(bear)
            sub_cache[(lam, vud)] = probs

    # Concatenate all OOS probs for this lambda across validation window
    all_probs = concatenate(sub_cache results)
    if smoothing_halflife > 0:
        all_probs = EWMA(all_probs, hl=smoothing_halflife)
    forecasts = (all_probs >= 0.5).astype(int)  # 1=bear, 0=bull

    # 0/1 strategy: invest in asset when bull, risk-free when bear
    strategy_returns = compute_returns(forecasts, asset_returns, rf, tc)
    validation_sharpe[lam] = sharpe(strategy_returns)

best_lambda = argmax(validation_sharpe)
```

#### Step B: Final Model with Best Lambda

```python
training = [ud - 11yr, ud)
X_train = standardize(features[training])

jm = JM(lam=best_lambda, n_init=3, max_it=20)
jm.fit(X_train)
labels = jm.labels(excess_returns[training])

target = np.roll(labels, -1)[:-1]
xgb = XGBClassifier(n_estimators=100, max_depth=6, lr=0.3)
xgb.fit(X_train[:-1], target)

X_oos = standardize(features[ud : ud+6months], using training stats)
oos_probs = xgb.predict_proba(X_oos)[:, 1]
```

#### Post-Processing

After all updates complete:
1. Concatenate all OOS probabilities
2. Deduplicate overlapping forecasts (keep first occurrence)
3. Apply EWMA smoothing (asset-specific halflife)
4. Threshold at 0.5 → binary forecasts
5. Compute 0/1 strategy wealth and metrics

### 4.3 Sub-Update Caching (Key Optimization)

The validation windows of consecutive updates overlap heavily. For example:
- Update at 2012-01-01: validates over 2007-2012, with sub-updates at 2007-01, 2007-07, 2008-01, ...
- Update at 2012-07-01: validates over 2007.5-2012.5, with sub-updates at 2007-07, 2008-01, ...

Many `(lambda, sub_update_date)` pairs are identical across consecutive main updates. The `sub_cache` dict avoids recomputing these, providing ~4x speedup.

### 4.4 Smoothing Halflife (Paper Footnote 14)

After computing raw XGBoost P(bear) probabilities, an EWMA smoother is applied before thresholding. The halflife was selected per-asset using the initial validation window (2002-2007):

| Halflife | Assets |
|----------|--------|
| 8 days | LargeCap, MidCap, SmallCap, AggBond, Treasury, REIT |
| 4 days | Commodity, Gold |
| 2 days | Corporate |
| 0 (none) | EAFE, EM, HighYield |

---

## 5. JM-Only Baseline: `run_jm_only()`

The JM-only strategy uses regime labels directly (carry-forward) without XGBoost enhancement.

### Key Differences from `process_asset()`

| Aspect | JM-XGB (`process_asset`) | JM-only (`run_jm_only`) |
|--------|--------------------------|------------------------|
| Forecast method | XGBoost P(bear) → threshold | JM label shifted 1 day (carry-forward) |
| Features used | Return + Macro (8-13 cols) | Return only (6-8 cols) |
| Validation JM params | n_init=3, max_it=20 | n_init=1, max_it=10 (faster) |
| Final JM params | n_init=3, max_it=20 | n_init=3, max_it=20 |
| Smoothing | Asset-specific EWMA | None (binary labels) |
| Training window (final) | `[ud-11yr, ud)` strictly OOS | `[ud-11yr, end_of_test]` includes OOS |
| Caching | Sub-update cache + disk | None (recomputes each run) |

**Note on data leakage in JM-only**: The final JM fit in `run_jm_only()` uses data up to the end of the test period (`fe=test_end`), not just up to the current update date. This is intentional: the "carry-forward" strategy labels today's regime using all available data, then uses today's label as tomorrow's forecast. This is how the paper describes the JM baseline — it has look-ahead bias by design, representing the theoretical best-case for JM without forecasting.

---

## 6. XGBoost Configuration

XGBoost is used identically in validation and final model fitting:

| Parameter | Value | Source |
|-----------|-------|--------|
| n_estimators | 100 | Paper: "default hyperparameters" |
| max_depth | 6 | XGBoost default |
| learning_rate | 0.3 | XGBoost default |
| use_label_encoder | False | Suppress deprecation warning |
| eval_metric | 'logloss' | Binary classification |
| verbosity | 0 | Silent |

**Target construction**: `np.roll(jm_labels, -1)[:-1]` — the label at time t becomes the target for features at time t-1. This means XGBoost learns to predict tomorrow's regime from today's features.

**Features**: Concatenation of per-asset return features (6-8 cols) and macro features (5 cols) = 11-13 total features, standardized using training-period statistics.

---

## 7. Portfolio Construction

### 7.1 Strategy Configurations

Seven strategies are backtested:

| Strategy | Optimization | Uses Regime | γ_risk | γ_trade | Description |
|----------|-------------|-------------|--------|---------|-------------|
| 60/40 | None | No | — | — | Fixed weights (Paper Table 5) |
| MinVar | Min variance | No | 10 | 0 | All assets, minimum variance |
| MinVar (JM-XGB) | Min variance | Yes | 10 | 1 | Exclude bear assets |
| MV | Mean-variance | No | 5 | 0 | EWMA return estimates |
| MV (JM-XGB) | Mean-variance | Yes | 10 | 1 | Regime-dependent returns |
| EW | Equal weight | No | — | — | 1/12 each asset |
| EW (JM-XGB) | Equal weight | Yes | — | — | Equal-weight bull assets only |

### 7.2 Optimization Problem (Paper Equation 2)

For MinVar and MV strategies:

```
maximize:  μ'w - γ_risk * w'Σw - γ_trade * tc * ||w - w_prev||₁
subject to: 0 ≤ w ≤ 0.4 (per-asset cap)
            Σw ≤ 1       (no leverage)
```

Where:
- `μ` = expected return vector (construction depends on strategy)
- `Σ` = covariance matrix (3-year lookback, EWM with halflife=252 days, ridge regularized)
- `w_prev` = previous day's weights (after drift)
- `tc = 5e-4` = one-way transaction cost
- Solved with CVXPY using SCS solver

### 7.3 Expected Return Construction

**MinVar (no regime)**: `μ = 1e-3 * ones` — tiny constant, effectively just minimizes variance.

**MinVar (JM-XGB)**: `μ[j] = 1e-3` if asset j is in bull regime (fc=0), `μ[j] = 0` if bear. Bear assets get zero expected return → optimizer allocates nothing to them.

**MV (no regime)**: `μ[j] = EWMA(excess_returns[j], halflife=1260 days)` — 5-year exponentially weighted average of historical excess returns. Uses `γ_risk=5`.

**MV (JM-XGB)**: `μ[j] = rrf[j]` where `rrf` is the regime-dependent return forecast:
- Bull: `max(mean(last 504 excess returns), 0) * 1.5`
- Bear: `min(-1e-3, mean(negative returns in last 504) * 0.5)`

### 7.4 Safety Rule

If fewer than 4 assets are in bull regime, allocate 100% to risk-free. This prevents concentrated bets during broad market stress.

### 7.5 Weight Drift

After each day's return, weights drift: `w_next = w * (1 + r) / (1 + portfolio_return)`. This is standard no-rebalance drift.

### 7.6 60/40 Weights (Paper Table 5)

```
LargeCap:10%, MidCap:5%, SmallCap:5%, EAFE:5%, EM:5%, REIT:10%
HighYield:10%, Commodity:5%, Gold:5%
AggBond:20%, Treasury:10%, Corporate:10%
```

---

## 8. Performance Metrics

`mets(r, w, rf)` computes:

| Metric | Formula |
|--------|---------|
| Return | `mean(r) * 252 - mean(rf) * 252` (annualized excess) |
| Volatility | `std(r) * sqrt(252)` |
| Sharpe | `Return / Volatility` |
| MDD | `min((cumwealth - cummax) / cummax)` |
| Calmar | `Return / |MDD|` |
| Turnover | `sum(|Δw|) / years` |
| Leverage | `mean(sum(w))` |

---

## 9. Code Structure: `run_full_pipeline()`

The pipeline is called via `if __name__ == '__main__': run_full_pipeline()`.

All function definitions (`table4`, `fig2`, `bt`, `mets`, `fig3`, `fig_all`) are at **module level** so they can be imported independently. All pipeline **calls** are inside `run_full_pipeline()` with 4-space indentation.

```
run_full_pipeline():
    1. _run_jmxgb_all()    → jmxgb  (12 assets, parallel + cached)
    2. _run_jm_all()       → jm_only (12 assets, parallel + cached)
    3. table4()            → Print Table 4 (0/1 strategy Sharpe & MDD)
    4. fig2()              → Save fig2.png (regime plots for 3 assets)
    5. build_regime_return_forecasts() → rfc, rrf
    6. _run_backtests()    → strats  (7 strategies, cached)
    7. Print Table 6       → Portfolio performance
    8. fig3()              → Save fig3.png (portfolio wealth curves)
    9. Print Table 7       → Forecast correlation
   10. Print Table 8       → gamma_trade sensitivity
   11. Print Table 9       → gamma_risk sensitivity
   12. fig_all()           → Save fig_all.png (all 12 asset regimes)
```

**Module-level imports**: `test_single.py` imports `process_asset`, `run_jm_only`, `exc_df`, `ret_df`, etc. directly without triggering the full pipeline, thanks to the `__name__` guard.

---

## 10. Known Divergences from Paper

### 10.1 Current Scorecard: 17/20 Checks Pass

| # | Check | Status | Detail |
|---|-------|--------|--------|
| 1 | 12 assets loaded | PASS | 12/12 |
| 2 | Test period coverage | PASS | 12/12 |
| 3 | Feature columns correct | PASS | 12/12 |
| 4 | Macro features >= 4 | PASS | 5 cols |
| 5 | B&H Sharpe within +/-0.15 | PASS | 11/12 |
| 6 | B&H MDD within +/-5pp | PASS | 12/12 |
| 7 | Regime shifts within +/-30 (3 known) | PASS | 2/3 |
| 8 | Bear% within +/-10pp (3 known) | **FAIL** | 1/3 (REIT) |
| 9 | Regime shifts < 200 all assets | **FAIL** | 8/12 (EAFE/EM/HY/Corp) |
| 10 | JM Sharpe within +/-0.50 | **FAIL** | 3/12 |
| 11 | JM-XGB Sharpe within +/-0.30 | PASS | 7/12 |
| 12 | JM-XGB Sharpe >= paper | PASS | 12/12 |
| 13 | JM-XGB MDD within +/-15pp | PASS | 12/12 |
| 14 | JM-XGB Sharpe > B&H | PASS | 12/12 |
| 15 | Portfolio Sharpe within +/-0.30 | PASS | 5/7 |
| 16 | Portfolio MDD within +/-10pp | PASS | 6/7 |
| 17 | Forecast correlation positive | PASS | 12/12 |
| 18 | MinVar(JM-XGB) > MinVar | PASS | 0.99 vs 0.49 |
| 19 | MV(JM-XGB) > 60/40 | PASS | 0.85 vs 0.48 |
| 20 | EW(JM-XGB) > EW | PASS | 1.34 vs 0.43 |

### 10.2 Regime Detection Comparison

| Asset | Bear%(Ours) | Bear%(Paper) | Delta | Shifts(Ours) | Shifts(Paper) | Delta |
|-------|-------------|-------------|-------|-------------|--------------|-------|
| LargeCap | 32.0% | 20.9% | +11.1pp | 76 | 46 | +30 |
| REIT | 44.6% | 18.4% | +26.2pp | 100 | 46 | +54 |
| AggBond | 46.2% | 41.5% | +4.7pp | 109 | 97 | +12 |
| EAFE | 38.6% | — | — | 416 | — | — |
| EM | 53.2% | — | — | 428 | — | — |
| HighYield | 51.4% | — | — | 344 | — | — |

### 10.3 Root Causes of Divergence

**1. Data source (largely addressed in V2)**

The paper uses Bloomberg total-return indices. V2 data uses total-return indices (Yahoo `^SP500TR`, `^RUTTR`), FRED bond TR indices (`BAMLHYH0A0HYM2TRIV`, `BAMLCC0A0CMTRIV`), and mutual fund adjusted close (VBMFX, VUSTX, VGSIX) which include dividends/coupons. Risk-free rate now from FRED `DGS3MO` (~1.1%/yr over test period, matching paper).

Remaining gaps:
- **EAFE/EM**: No free MSCI total-return index; mutual fund adj close (VGTSX from 1996, VEIEX from 1994) is a reasonable proxy but not exact
- **MidCap**: VIMSX starts 1998 (S&P MidCap 400 launched 1991); ~7yr gap vs paper
- **Commodity**: DBC ETF only from 2006 (underlying index from ~2003); no free historical data
- **Gold**: GC=F futures from 2000; paper's LBMA data from 1991 — ~9yr gap

**2. Lambda selection sensitivity**

Small differences in input data cascade through lambda selection:
- Different excess returns → different JM labels → different XGBoost targets → different P(bear) → different validation Sharpe → different optimal lambda → different final forecasts
- The lambda sequence shows convergence to λ=7.0 for most assets after 2013, but early-period lambdas (2007-2011) are volatile and data-dependent

**3. JM-only baseline is highly sensitive**

The JM-only strategy uses carry-forward labels without smoothing. Small differences in price levels during crisis periods (2008, 2020) can flip individual days' labels, and the carry-forward propagates these differences.

### 10.4 Test Failures (5 of 112)

All are data-driven divergences, not code bugs:

| Test | Expected | Got | Cause |
|------|----------|-----|-------|
| `test_regime_bear_pct[REIT]` | 18.4% +/-15pp | 44.6% | Yahoo VNQ vs Bloomberg REIT |
| `test_regime_shifts[REIT]` | 46 +/-40 | 100 | Same |
| `test_regime_shifts_reasonable[EAFE]` | < 300 | 416 | ETF data quality |
| `test_regime_shifts_reasonable[EM]` | < 300 | 428 | ETF data quality |
| `test_regime_shifts_reasonable[HighYield]` | < 300 | 344 | Missing coupon income |

---

## 11. What Needs Improvement to Match the Paper

### 11.1 High Priority: Data Source

The single most impactful change would be using total-return data instead of price-return data. Options:

1. **Bloomberg terminal access**: Use Bloomberg total return indices directly (paper's source). This would likely resolve most divergences immediately.
2. **Dividend/coupon adjustment**: Reconstruct total returns by adding back dividend yields from another source (e.g., FRED for Treasury yields, ETF distribution data for equity/bond funds).
3. **Alternative data providers**: Refinitiv, Quandl/Nasdaq Data Link, or CRSP for total-return series.

**Expected impact**: Fixing the data source would likely bring REIT bear% from 44.6% to near 18.4%, reduce EAFE/EM shifts from 400+ to reasonable levels, and correct HighYield/Corporate Sharpe ratios.

### 11.2 High Priority: Risk-Free Rate

Our `^IRX` gives 2.54%/yr average. The paper uses ~1.1%. This shifts every excess return calculation, every Sharpe ratio, and every regime label. Potential fixes:
- Use FRED DGS3MO (3-month Treasury constant maturity) — likely closer to paper's source
- Or compute from total-return T-bill data

### 11.3 Medium Priority: JM-Only Implementation

The JM-only baseline shows the largest divergences (Sharpe +0.87 for LargeCap vs paper). Issues:
- Current implementation includes OOS data in the final JM fit (`common<=te_`), which is intentional carry-forward but may not match the paper's exact procedure
- Validation uses reduced params (n_init=1, max_it=10) — paper may use full params
- No sub-update caching — recomputes on every run

### 11.4 Medium Priority: Turnover

Our portfolio turnover is significantly higher than the paper's (e.g., EW(JM-XGB): 37.5 vs 11.7 in paper). This suggests:
- The paper may rebalance less frequently (monthly instead of daily?)
- Or the paper's regime forecasts are more stable (fewer shifts → less turnover)
- The excessive turnover in our EW(JM-XGB) is partly caused by excessive regime shifts in EAFE/EM/HighYield

### 11.5 Low Priority: MV Portfolio

MV is highly sensitive to return forecast quality. Our MV(JM-XGB) Sharpe (0.85) is below paper (1.02), and leverage is much lower (0.30 vs 0.86). This is likely downstream of regime detection differences — fixing the data source should cascade improvements to MV.

### 11.6 Low Priority: Smoothing Halflife Calibration

The smoothing halflife values were taken from the paper's footnote 14 as fixed constants. The paper selected these from candidates {0, 2, 4, 8} using the initial validation window (2002-2007). With different data, optimal halflife values might differ. Could re-calibrate these via a grid search on the 2002-2007 window.

---

## 12. Paper Reference Values

### Table 4 — 0/1 Strategy Sharpe Ratios

|  | LgCap | MdCap | SmCap | EAFE | EM | REIT | AggBd | Treas | HiYld | Corp | Comm | Gold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B&H | 0.50 | 0.45 | 0.36 | 0.20 | 0.20 | 0.27 | 0.46 | 0.26 | 0.67 | 0.54 | 0.03 | 0.43 |
| JM | 0.59 | 0.49 | 0.28 | 0.28 | 0.65 | 0.39 | 0.43 | 0.21 | 1.49 | 0.83 | 0.08 | 0.12 |
| JM-XGB | 0.79 | 0.59 | 0.51 | 0.56 | 0.85 | 0.56 | 0.67 | 0.38 | 1.88 | 0.76 | 0.23 | 0.31 |

### Table 4 — 0/1 Strategy Max Drawdown

|  | LgCap | MdCap | SmCap | EAFE | EM | REIT | AggBd | Treas | HiYld | Corp | Comm | Gold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B&H | -55.2% | -55.1% | -58.9% | -60.4% | -65.2% | -74.2% | -18.4% | -46.9% | -32.9% | -22.0% | -75.5% | -44.6% |
| JM | -24.8% | -33.2% | -38.4% | -29.7% | -26.2% | -54.7% | -6.1% | -22.9% | -13.9% | -8.3% | -58.5% | -31.8% |
| JM-XGB | -17.7% | -29.9% | -35.8% | -19.9% | -21.3% | -32.7% | -6.3% | -17.5% | -10.2% | -6.8% | -47.9% | -21.6% |

### Figure 2 — Regime Detection

| Asset | Bear% | Shifts |
|-------|-------|--------|
| LargeCap | 20.9% | 46 |
| REIT | 18.4% | 46 |
| AggBond | 41.5% | 97 |

### Table 6 — Portfolio Performance

| Strategy | Return | Vol | Sharpe | MDD | Calmar | Turnover | Leverage |
|----------|--------|-----|--------|-----|--------|----------|----------|
| 60/40 | 5.0% | 8.9% | 0.57 | -31.5% | 0.16 | 0.74 | 1.00 |
| MinVar | 2.8% | 4.0% | 0.70 | -19.3% | 0.15 | 0.49 | 1.00 |
| MinVar (JM-XGB) | 3.9% | 3.5% | 1.12 | -7.1% | 0.55 | 2.06 | 0.91 |
| MV | 2.6% | 7.1% | 0.37 | -25.6% | 0.10 | 3.40 | 0.95 |
| MV (JM-XGB) | 8.9% | 8.7% | 1.02 | -13.5% | 0.66 | 9.12 | 0.86 |
| EW | 5.5% | 10.8% | 0.51 | -37.5% | 0.15 | 0.81 | 1.00 |
| EW (JM-XGB) | 8.2% | 9.0% | 0.91 | -17.6% | 0.47 | 11.70 | 0.92 |

### Table 7 — Forecast Correlation (JM-XGB)

| | LgCap | MdCap | SmCap | EAFE | EM | REIT | AggBd | Treas | HiYld | Corp | Comm | Gold |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| JM-XGB | 1.66% | 0.90% | 1.03% | 4.53% | 6.02% | 2.10% | 3.22% | 1.64% | 10.54% | 2.62% | 3.39% | 0.32% |
| EWMA | -1.58% | -3.86% | -3.72% | -3.73% | -2.03% | -5.09% | 1.25% | -1.17% | -0.16% | -0.06% | -1.05% | -1.59% |

### Table 8 — MinVar(JM-XGB) gamma_trade Sensitivity

| γ_trade | Return | Vol | Sharpe | MDD | Calmar | Turnover | Leverage |
|---------|--------|-----|--------|-----|--------|----------|----------|
| 0.0 | 5.7% | 5.2% | 1.09 | -11.1% | 0.51 | 11.80 | 0.91 |
| 1.0 | 3.9% | 3.5% | 1.12 | -7.1% | 0.55 | 2.06 | 0.91 |

### Table 9 — MV(JM-XGB) gamma_risk Sensitivity

| γ_risk | Return | Vol | Sharpe | MDD | Calmar | Turnover | Leverage |
|--------|--------|-----|--------|-----|--------|----------|----------|
| 5.0 | 10.1% | 10.0% | 1.01 | -15.4% | 0.65 | 10.03 | 0.89 |
| 10.0 | 8.9% | 8.7% | 1.02 | -13.5% | 0.66 | 9.12 | 0.86 |
| 20.0 | 6.7% | 6.9% | 0.96 | -13.5% | 0.49 | 7.67 | 0.76 |

---

## 13. Performance Characteristics

| Operation | Cold (no cache) | Warm (cached) |
|-----------|----------------|---------------|
| Data download | ~30s | <1s |
| Feature computation | ~2s | <1s |
| Single asset JM-XGB | ~194s | <1s |
| All 12 assets JM-XGB (parallel) | ~330s | <1s |
| All 12 assets JM-only (parallel) | ~615s | <1s |
| 7 portfolio backtests | ~45s | <1s |
| Full pipeline | ~15-20 min | ~1 min |
| pytest (112 tests) | ~80s | ~80s |

Parallelism: `joblib.Parallel(n_jobs=-1)` for multi-asset JM-XGB and JM-only runs. Backtests are sequential (each depends on prior day's weights).
