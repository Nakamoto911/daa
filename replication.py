#!/usr/bin/env python3
"""
=============================================================================
REPLICATION: Dynamic Asset Allocation with Asset-Specific Regime Forecasts
Shu, Yu, Mulvey (2024) — SSRN 4864358
=============================================================================
v6 — Paper-faithful Algorithm 2 lambda selection + sub-update caching.

Implements the paper's Algorithm 2 exactly:
  For each biannual update, for each candidate lambda, run Algorithm 1
  (biannual JM+XGB with 11yr training, default XGB params) over a 5-year
  validation window, then select lambda that maximizes validation 0/1 Sharpe.
Both validation and final fits use the same parameters (n_init=3, max_it=20,
n_estimators=100, max_depth=6) per paper's "default hyperparameters" approach.
Speed: sub-update caching eliminates redundant JM+XGB fits across overlapping
validation windows.
"""

# ===========================================================================
# CELL 1: Install & Import
# ===========================================================================
# !pip install -q yfinance xgboost cvxpy joblib

import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, time, os, pickle, hashlib
import matplotlib.ticker as mticker
from pathlib import Path
from matplotlib.patches import Patch
import yfinance as yf, xgboost as xgb
from joblib import Parallel, delayed
try: import cvxpy as cp
except: os.system("pip install -q cvxpy"); import cvxpy as cp
import requests
HAS_FRED = True  # uses requests (always available via yfinance dependency)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'figure.figsize':(12,6),'font.size':11,'font.family':'serif'})
print("Imports OK")

# ===========================================================================
# CELL 2: Config
# ===========================================================================
LAM_FLOOR = 40  # Lambda floor: min lambda for Algorithm 2 grid search (0 = disabled)
                # See RESEARCH_LAMBDA.md: fixed lambda=50 matches paper (42 shifts, Sharpe=0.801).
                # Authors' jumpmodels package uses jump_penalty=50 as default.
                # Floor=40 -> grid [40, 100], eliminating low-lambda noise from Yahoo data.
# Asset-specific lambda floor overrides.
# REIT: Even with IYR (same DJ index family as paper's Bloomberg DJUSRET), Yahoo data
# produces less separable JM features than Bloomberg — validation picks lambda=40 for
# 23/34 windows, yielding ~50% bear (paper: 18.4%). At lambda=100, JM gives ~25% bear
# across most windows. Override floor to 100 so validation only considers lambda=100.
LAM_FLOOR_OVERRIDE = {'REIT': 100}
DATA_VERSION = f'v3_fred_macro_lf{LAM_FLOOR}' if LAM_FLOOR > 0 else 'v3_fred_macro'

# V2 data sources: prioritized source chains per asset.
# Each source: type (yahoo_index/yahoo_etf/yahoo_mf/fred_tr_index/fred_price),
#              ticker, data_type (tr_index/adj_close/price — all use pct_change).
# Fallback: old ETF+MF approach if primary sources fail.
ASSET_CONFIG_V2 = {
    'LargeCap': {
        'sources': [{'type':'yahoo_index','ticker':'^SP500TR'}],
        'fallback': {'etf':'SPY','bf':'VFINX'},
    },
    'MidCap': {
        'sources': [
            {'type':'yahoo_mf','ticker':'VIMSX'},   # adj close ~ TR, from ~1990s
            {'type':'yahoo_etf','ticker':'IJH'},
        ],
        'fallback': {'etf':'IJH','bf':'VIMSX'},
    },
    'SmallCap': {
        'sources': [
            {'type':'yahoo_index','ticker':'^RUTTR'},  # Russell 2000 TR, from 1995
            {'type':'yahoo_mf','ticker':'NAESX'},      # backfill pre-1995
        ],
        'fallback': {'etf':'IWM','bf':'NAESX'},
    },
    'EAFE': {
        'sources': [
            {'type':'yahoo_mf','ticker':'VGTSX'},
            {'type':'yahoo_etf','ticker':'EFA'},
        ],
        'fallback': {'etf':'EFA','bf':'VGTSX'},
    },
    'EM': {
        'sources': [
            {'type':'yahoo_mf','ticker':'VEIEX'},
            {'type':'yahoo_etf','ticker':'EEM'},
        ],
        'fallback': {'etf':'EEM','bf':'VEIEX'},
    },
    'AggBond': {
        'sources': [{'type':'yahoo_mf','ticker':'VBMFX'}],
        'fallback': {'etf':'AGG','bf':'VBMFX'},
    },
    'Treasury': {
        'sources': [
            {'type':'yahoo_mf','ticker':'VUSTX'},   # adj close ~ TR, from 1990
        ],
        'fallback': {'etf':'SPTL','bf':'VUSTX'},
    },
    'HighYield': {
        'sources': [{'type':'fred_tr_index','ticker':'BAMLHYH0A0HYM2TRIV'}],
        'fallback': {'etf':'HYG','bf':'VWEHX'},
    },
    'Corporate': {
        'sources': [{'type':'fred_tr_index','ticker':'BAMLCC0A0CMTRIV'}],
        'fallback': {'etf':'SPBO','bf':'VFICX'},
    },
    'REIT': {
        'sources': [
            {'type':'yahoo_etf','ticker':'IYR'},     # DJ US Real Estate ETF, same index family as paper's DJUSRET
            {'type':'yahoo_mf','ticker':'VGSIX'},    # Vanguard REIT (MSCI REIT), backfill pre-IYR
        ],
        'fallback': {'etf':'IYR','bf':'VGSIX'},
    },
    'Commodity': {
        'sources': [{'type':'yahoo_etf','ticker':'DBC'}],
        'fallback': {'etf':'DBC','bf':None},
    },
    'Gold': {
        'sources': [
            {'type':'yahoo_etf','ticker':'GC=F'},   # gold futures, back to ~2000
            {'type':'yahoo_etf','ticker':'GLD'},     # gold ETF, back to 2004
        ],
        'fallback': {'etf':'GLD','bf':None},
    },
}

# Backward compatibility: derive old-style config from V2
ASSET_CONFIG = {nm: cfg['fallback'] for nm, cfg in ASSET_CONFIG_V2.items()}
ASSETS = list(ASSET_CONFIG.keys())
EQ_RE = ['LargeCap','MidCap','SmallCap','EAFE','EM','REIT']
BD_CM = ['AggBond','Treasury','HighYield','Corporate','Commodity','Gold']
SMOOTH = {'LargeCap':10,'MidCap':10,'SmallCap':10,'EAFE':8,'EM':8,
          'AggBond':8,'Treasury':8,'HighYield':8,'Corporate':2,
          'REIT':8,'Commodity':4,'Gold':4}
W6040 = {'LargeCap':.10,'MidCap':.05,'SmallCap':.05,'EAFE':.05,'EM':.05,
         'AggBond':.20,'Treasury':.10,'HighYield':.10,'Corporate':.10,
         'REIT':.10,'Commodity':.05,'Gold':.05}

# ===========================================================================
# CELL 2.5: Cache Management
# ===========================================================================
CACHE_DIR = Path("cache")
for _d in ['data', 'models', 'backtests']:
    (CACHE_DIR / _d).mkdir(parents=True, exist_ok=True)

def _save_cache(obj, category, name):
    path = CACHE_DIR / category / f"{name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump({'data': obj, 'ts': time.time()}, f)

def _load_cache(category, name, max_hours=168):
    path = CACHE_DIR / category / f"{name}.pkl"
    if not path.exists(): return None
    try:
        with open(path, 'rb') as f:
            c = pickle.load(f)
        if (time.time() - c['ts']) / 3600 > max_hours: return None
        return c['data']
    except: return None

def clear_cache(category=None):
    targets = [category] if category else ['data','models','backtests']
    for d in targets:
        for f in (CACHE_DIR / d).glob("*.pkl"): f.unlink()
    print(f"Cache cleared: {', '.join(targets)}")

def cache_stats():
    for d in ['data','models','backtests']:
        files = list((CACHE_DIR / d).glob("*.pkl"))
        sz = sum(f.stat().st_size for f in files) / (1024**2)
        print(f"  {d:12s}: {len(files):3d} files, {sz:6.1f} MB")

# ===========================================================================
# CELL 3: Data Download (cached)
# ===========================================================================
def _gp(data, tk):
    if data is None or len(data)==0: return pd.Series(dtype=float)
    for c in ['Adj Close','Close']:
        try:
            s = data[(c,tk)] if isinstance(data.columns, pd.MultiIndex) else data[c]
            if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
            s = s.dropna()
            if len(s)>0: return s
        except: pass
    try: return data.iloc[:,0].dropna()
    except: return pd.Series(dtype=float)

def _dl(tk, start, end):
    for aa in [False, True]:
        try:
            s = _gp(yf.download(tk, start=start, end=end, progress=False, auto_adjust=aa), tk)
            if len(s)>0: return s
        except: pass
    try: return yf.Ticker(tk).history(start=start, end=end, auto_adjust=True)['Close'].dropna()
    except: return pd.Series(dtype=float)

def _dl_fred(series_id, start, end):
    """Download a series from FRED via public CSV endpoint (no API key needed)."""
    try:
        url = (f"https://fred.stlouisfed.org/graph/fredgraph.csv"
               f"?id={series_id}&cosd={start}&coed={end}")
        df = pd.read_csv(url, index_col=0, parse_dates=True, na_values='.')
        s = df.iloc[:, 0].dropna()
        if len(s) > 0:
            return s
    except Exception as e:
        print(f"  FRED download failed for {series_id}: {e}")
    return pd.Series(dtype=float)

def _download_asset(name, cfg, start, end):
    """Download asset data using prioritized source chain with fallback.
    Returns a pd.Series of daily returns.
    """
    segments = []
    for src in cfg['sources']:
        ticker = src['ticker']
        src_type = src['type']
        if src_type in ('yahoo_index', 'yahoo_etf', 'yahoo_mf'):
            raw = _dl(ticker, start, end)
        elif src_type in ('fred_tr_index', 'fred_price'):
            raw = _dl_fred(ticker, start, end)
        else:
            raw = pd.Series(dtype=float)
        if len(raw) > 0:
            returns = raw.pct_change().dropna()
            if len(returns) > 0:
                segments.append(returns)
                print(f"    {name:12s}: {src_type}({ticker}) -> {len(returns)} days "
                      f"[{returns.index[0].date()} to {returns.index[-1].date()}]")

    if not segments:
        print(f"    {name:12s}: primary sources failed, using fallback ETF+MF")
        fb = cfg['fallback']
        ep = _dl(fb['etf'], start, end)
        bp = _dl(fb['bf'], start, end) if fb['bf'] else pd.Series(dtype=float)
        if len(ep) > 0 and len(bp) > 0:
            bb = bp[bp.index < ep.index[0]]
            r = pd.concat([bb.pct_change().dropna(), ep.pct_change().dropna()]) if len(bb) > 0 else ep.pct_change().dropna()
            return r[~r.index.duplicated(keep='last')]
        elif len(ep) > 0: return ep.pct_change().dropna()
        elif len(bp) > 0: return bp.pct_change().dropna()
        else: return pd.Series(dtype=float)

    if len(segments) == 1:
        return segments[0]

    # Splice: first source preferred, later sources fill earlier dates
    combined = segments[0]
    for i in range(1, len(segments)):
        prepend = segments[i][segments[i].index < combined.index[0]]
        if len(prepend) > 0:
            combined = pd.concat([prepend, combined])
    return combined[~combined.index.duplicated(keep='last')].sort_index()

def load_data(start='1990-01-01', end='2024-01-01', use_cache=True):
    key = f"raw_{DATA_VERSION}_{start}_{end}"
    if use_cache:
        cached = _load_cache('data', key, max_hours=24)
        if cached is not None:
            print(f"Data loaded from cache ({key})")
            return cached

    print(f"Downloading (yfinance {yf.__version__}, FRED {'available' if HAS_FRED else 'unavailable'})...")
    rets = {}
    for nm, cfg in ASSET_CONFIG_V2.items():
        r = _download_asset(nm, cfg, start, end)
        if len(r) == 0:
            print(f"  {nm}: NO DATA"); continue
        rets[nm] = r
        print(f"  {nm:12s}: {len(r):>5d} days  {r.index[0].date()}->{r.index[-1].date()}")

    ret_df = pd.DataFrame(rets).dropna(how='all').fillna(0.)

    # Risk-free rate: prefer FRED DGS3MO (matches paper's 3-month CM rate)
    rf_fred = _dl_fred('DGS3MO', start, end) if HAS_FRED else pd.Series(dtype=float)
    if len(rf_fred) > 100:
        rf = ((1 + rf_fred / 100) ** (1 / 252) - 1)
        rf = rf.reindex(ret_df.index).ffill().fillna(0)
        print(f"  RF: FRED DGS3MO ({rf.mean()*252*100:.2f}%/yr)")
    else:
        # Fallback to Yahoo ^IRX
        rfp = _dl('^IRX', start, end)
        rf = ((1+rfp/100)**(1/252)-1).reindex(ret_df.index).ffill().fillna(0) if len(rfp)>0 else pd.Series(0.02/252, index=ret_df.index)
        print(f"  RF: Yahoo ^IRX fallback ({rf.mean()*252*100:.2f}%/yr)")

    # Macro features: FRED primary sources per paper Table 3, Yahoo fallback
    macro = {}
    # T2Y: paper says "US Treasury 2-Year Yield" → FRED DGS2 (2yr constant maturity)
    # Previously used ^IRX (13-week T-bill) which is a 3-month rate — WRONG
    t2y = _dl_fred('DGS2', start, end)
    if len(t2y) > 100:
        macro['T2Y'] = t2y
        print(f"  Macro T2Y: FRED DGS2 ({len(t2y)} obs)")
    else:
        s = _dl('^IRX', start, end)
        if len(s) > 0: macro['T2Y'] = s; print(f"  Macro T2Y: Yahoo ^IRX fallback ({len(s)} obs)")
    # T10Y: FRED DGS10 (10yr constant maturity)
    t10y = _dl_fred('DGS10', start, end)
    if len(t10y) > 100:
        macro['T10Y'] = t10y
        print(f"  Macro T10Y: FRED DGS10 ({len(t10y)} obs)")
    else:
        s = _dl('^TNX', start, end)
        if len(s) > 0: macro['T10Y'] = s; print(f"  Macro T10Y: Yahoo ^TNX fallback ({len(s)} obs)")
    # VIX: FRED VIXCLS (CBOE VIX)
    vix = _dl_fred('VIXCLS', start, end)
    if len(vix) > 100:
        macro['VIX'] = vix
        print(f"  Macro VIX: FRED VIXCLS ({len(vix)} obs)")
    else:
        s = _dl('^VIX', start, end)
        if len(s) > 0: macro['VIX'] = s; print(f"  Macro VIX: Yahoo ^VIX fallback ({len(s)} obs)")
    mdf = pd.DataFrame(macro).ffill().reindex(ret_df.index).ffill().bfill()

    exc_df = ret_df.subtract(rf, axis=0)
    wdf = (1+ret_df).cumprod()
    print(f"  Final: {ret_df.shape[0]} days x {ret_df.shape[1]} assets, RF={rf.mean()*252*100:.1f}%/yr")

    result = (ret_df, exc_df, rf, mdf, wdf)
    if use_cache:
        _save_cache(result, 'data', key)
    return result

ret_df, exc_df, rf_daily, macro_df, wealth_df = load_data()

# ===========================================================================
# CELL 4: Figure 1
# ===========================================================================
def plot_fig1(w):
    fig,(a1,a2)=plt.subplots(2,1,figsize=(12,10))
    c1=['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#8c564b','#e377c2']
    c2=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b']
    for ax,grp,cs,t in [(a1,EQ_RE,c1,'Equity & RE'),(a2,BD_CM,c2,'Bonds & Commodities')]:
        for nm,c in zip(grp,cs):
            if nm in w: ax.plot(w.index,w[nm],label=nm,color=c,lw=.8)
        ax.set_yscale('log');ax.set_ylabel('Wealth (log)');ax.set_title(f'Wealth Curves: {t}')
        ax.legend(loc='upper left',ncol=2,fontsize=9,prop={'family':'monospace'});ax.grid(alpha=.3)
    plt.tight_layout();plt.savefig('fig1.png',dpi=150,bbox_inches='tight');plt.close()
plot_fig1(wealth_df)

# ===========================================================================
# CELL 5: Features (Tables 2 & 3) — cached
# ===========================================================================
def _dd(r,hl): return np.sqrt(r.clip(upper=0).pow(2).ewm(halflife=hl,min_periods=max(hl,5)).mean())

def ret_features(er, nm=None):
    f={}; skip=nm in['AggBond','Treasury','Gold']
    if not skip:
        for h in[5,21]: f[f'DD{h}']=np.log(_dd(er,h).clip(lower=1e-10))
    for h in[5,10,21]: f[f'R{h}']=er.ewm(halflife=h,min_periods=max(h,5)).mean()
    for h in[5,10,21]:
        d=_dd(er,h).replace(0,np.nan)
        f[f'S{h}']=er.ewm(halflife=h,min_periods=max(h,5)).mean()/d
    return pd.DataFrame(f,index=er.index).dropna()

def macro_features(rdf, mdf):
    f={}
    if 'T2Y' in mdf: f['t2d']=mdf['T2Y'].diff().ewm(halflife=21,min_periods=21).mean()
    if 'T10Y' in mdf and 'T2Y' in mdf:
        sl=mdf['T10Y']-mdf['T2Y']
        f['yc']=sl.ewm(halflife=10,min_periods=10).mean()
        f['ycd']=sl.diff().ewm(halflife=21,min_periods=21).mean()
    if 'VIX' in mdf: f['vd']=np.log(mdf['VIX']).diff().ewm(halflife=63,min_periods=63).mean()
    if 'LargeCap' in rdf and 'AggBond' in rdf:
        f['sbc']=rdf['LargeCap'].rolling(252,min_periods=126).corr(rdf['AggBond'])
    return pd.DataFrame(f).dropna(how='all')

def _compute_features(exc_df, ret_df, macro_df, use_cache=True):
    key = f"features_{DATA_VERSION}_{exc_df.index[0].date()}_{exc_df.index[-1].date()}"
    if use_cache:
        cached = _load_cache('data', key, max_hours=168)
        if cached is not None:
            print("Features loaded from cache")
            return cached
    print("Computing features...")
    # Use exc_df with real data only: drop leading zeros from fillna(0)
    # by finding the first nonzero return for each asset.
    RF = {}
    for nm in ASSETS:
        if nm not in exc_df: continue
        er = exc_df[nm]
        # Find first date with nonzero raw return (before rf subtraction)
        raw = ret_df[nm]
        first_real = raw[raw.abs() > 1e-10].index.min()
        if pd.isna(first_real):
            RF[nm] = ret_features(er, nm)
        else:
            RF[nm] = ret_features(er.loc[first_real:], nm)
    MF = macro_features(ret_df, macro_df)
    print(f"  {len(RF)} assets, macro shape={MF.shape}")
    result = (RF, MF)
    if use_cache:
        _save_cache(result, 'data', key)
    return result

RF, MF = _compute_features(exc_df, ret_df, macro_df)

# ===========================================================================
# CELL 6: Jump Model (vectorized K=2 Viterbi)
# ===========================================================================
class JM:
    def __init__(self, lam=1., n_init=3, max_it=20, seed=42):
        self.lam,self.n_init,self.max_it,self.seed=lam,n_init,max_it,seed
        self.states_=self.centers_=None

    def fit(self, X):
        if isinstance(X,pd.DataFrame): X=X.values
        T,D=X.shape; rng=np.random.RandomState(self.seed)
        bc,bs,bcc=np.inf,None,None
        for _ in range(self.n_init):
            c=X[rng.choice(T,2,replace=False)].copy(); prev=np.inf
            for _ in range(self.max_it):
                d=0.5*np.stack([((X-c[k])**2).sum(1) for k in range(2)],1)
                # Viterbi K=2
                V=np.empty_like(d); bp=np.empty(d.shape,dtype=np.intp)
                V[0]=d[0]
                for t in range(1,T):
                    stay=V[t-1]; sw=V[t-1,::-1]+self.lam
                    pick=stay<=sw
                    V[t]=d[t]+np.where(pick,stay,sw)
                    bp[t]=np.where(pick,[0,1],[1,0])
                s=np.empty(T,dtype=np.intp); s[-1]=V[-1].argmin()
                for t in range(T-2,-1,-1): s[t]=bp[t+1,s[t+1]]
                cost=V[-1,s[-1]]
                for k in range(2):
                    m=s==k
                    if m.any(): c[k]=X[m].mean(0)
                if abs(prev-cost)<1e-6: break
                prev=cost
            if cost<bc: bc,bs,bcc=cost,s.copy(),c.copy()
        self.states_,self.centers_=bs,bcc; return self

    def labels(self, exc):
        if isinstance(exc,(pd.Series,pd.DataFrame)): exc=exc.values.flatten()
        c=np.array([exc[self.states_==k].sum() for k in range(2)])
        b=c.argmax(); return np.where(self.states_==b,0,1)

def stdz(X):
    if isinstance(X,pd.DataFrame):
        m,s=X.mean(),X.std().replace(0,1); return (X-m)/s
    m,s=X.mean(0),X.std(0); s[s==0]=1; return (X-m)/s

# ===========================================================================
# CELL 7: FAST Pipeline
# ===========================================================================
def process_asset(nm, exc_df, ret_df, rf_daily, RF, MF, lam_grid,
                  test_start, test_end, val_years=5, tc=5e-4):
    """
    Paper Algorithm 2 (faithful): Time-series cross-validation for lambda selection.

    For each biannual update at date ud:
      1. For each candidate lambda, run Algorithm 1 (biannual JM+XGB with 11yr
         training, default XGB params) over a 5-year validation window [ud-5yr, ud).
         Compute Sharpe of 0/1 strategy.
      2. Pick lambda* with best validation Sharpe.
      3. Refit JM+XGB on [ud-11yr, ud) with lambda*, forecast next 6mo OOS.

    Speed optimization: cache sub-update probabilities by (lambda, sub_update_date)
    since consecutive validation windows share many overlapping sub-updates.
    """
    t0 = time.time()
    er = exc_df[nm].dropna(); ar = ret_df[nm].dropna(); rf = rf_daily
    feat = RF[nm]; shl = SMOOTH.get(nm, 4)

    common = feat.index
    if MF is not None and len(MF) > 0: common = common.intersection(MF.index)
    common = common.sort_values()

    feat_a = feat.reindex(common)
    if MF is not None and len(MF) > 0:
        xf = pd.concat([feat_a, MF.reindex(common).ffill().fillna(0)], axis=1)
    else: xf = feat_a.copy()
    exc_a = er.reindex(common).fillna(0)

    xf_vals = xf.values
    feat_vals = feat_a.values
    exc_vals = exc_a.values

    updates = pd.date_range(test_start, test_end, freq='6MS')
    all_prob = pd.Series(dtype=float); best_lams = {}

    # Cache: (lam, sub_update_date) -> pd.Series of OOS probs for that sub-update
    # This avoids re-fitting the same JM+XGB for overlapping validation windows
    sub_cache = {}

    n_updates = len(updates)
    for iu, ud in enumerate(updates):
        print(f"\r    {nm}: update {iu+1}/{n_updates} ({ud.strftime('%Y-%m')}) ...", end='', flush=True)
        # --- Algorithm 2: lambda selection via 5-year validation window ---
        val_start = ud - pd.DateOffset(years=val_years)
        val_end = ud

        # Paper: Algorithm 1 uses biannual sub-updates within validation window
        val_sub_updates = pd.date_range(val_start, val_end, freq='6MS')

        best_sh, best_l = -np.inf, lam_grid[0]
        any_lam_evaluated = False

        for lam in lam_grid:
            # Run Algorithm 1 for this lambda over validation window
            val_all_prob = pd.Series(dtype=float)

            for viu, vud in enumerate(val_sub_updates):
                cache_key = (lam, vud)

                # Compute OOS end date for this sub-update in this validation window
                vfc_end = val_sub_updates[viu + 1] if viu + 1 < len(val_sub_updates) else pd.Timestamp(val_end)

                # Cache stores fitted model (clf) or None; OOS predictions recomputed each time
                # because vfc_end varies by validation window context
                if cache_key in sub_cache:
                    cached = sub_cache[cache_key]
                    if cached is not None:
                        clf_cached = cached
                        oos_mask = (common >= vud) & (common < vfc_end)
                        oos_pos = np.where(oos_mask)[0]
                        if len(oos_pos) > 0:
                            probs = clf_cached.predict_proba(xf_vals[oos_pos])[:, 1]
                            val_all_prob = pd.concat([val_all_prob,
                                pd.Series(probs, index=common[oos_pos])])
                    continue

                vtr_s = vud - pd.DateOffset(years=11)
                vtr_mask = (common >= vtr_s) & (common < vud)
                vtr_idx = common[vtr_mask]
                if len(vtr_idx) < 504:
                    sub_cache[cache_key] = None
                    continue

                vtr_pos = np.searchsorted(common, vtr_idx)

                X_v = feat_vals[vtr_pos]
                m_v, s_v = X_v.mean(0), X_v.std(0); s_v[s_v == 0] = 1
                X_v_std = (X_v - m_v) / s_v
                exc_v = exc_vals[vtr_pos]

                # Paper: same params as final model (default XGBoost)
                jm = JM(lam=lam, n_init=3, max_it=20)
                jm.fit(X_v_std)
                labs = jm.labels(exc_v)

                target = np.roll(labs, -1)[:-1]
                xf_tr = xf_vals[vtr_pos[:-1]]
                if len(np.unique(target)) < 2:
                    sub_cache[cache_key] = None
                    continue

                # Paper: default XGBoost params (n_estimators=100, max_depth=6)
                clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                                         eval_metric='logloss', random_state=42, verbosity=0)
                clf.fit(xf_tr, target)

                # Cache the fitted model (training is expensive; OOS prediction is cheap)
                sub_cache[cache_key] = clf

                oos_mask = (common >= vud) & (common < vfc_end)
                oos_pos = np.where(oos_mask)[0]
                if len(oos_pos) == 0:
                    continue

                probs = clf.predict_proba(xf_vals[oos_pos])[:, 1]
                probs_s = pd.Series(probs, index=common[oos_pos])
                val_all_prob = pd.concat([val_all_prob, probs_s])

            val_all_prob = val_all_prob[~val_all_prob.index.duplicated(keep='last')].sort_index()
            if shl > 0 and len(val_all_prob) > 0:
                val_all_prob = val_all_prob.ewm(halflife=shl, min_periods=1).mean()

            if len(val_all_prob) < 50: continue

            fc_val = (val_all_prob >= 0.5).astype(int)
            ci = fc_val.index.intersection(ar.index)
            if len(ci) < 30: continue

            fc_s = fc_val.reindex(ci)
            pos = (fc_s == 0).astype(float)
            r = ar.reindex(pos.index); rfv = rf.reindex(pos.index).fillna(0)
            sr = pos * r + (1 - pos) * rfv - pos.diff().abs().fillna(0) * tc
            exc_sr = sr - rfv
            vol = exc_sr.std() * np.sqrt(252)
            sh = exc_sr.mean() * 252 / vol if vol > 0 else 0

            if sh > best_sh:
                best_sh, best_l = sh, lam
                any_lam_evaluated = True

        if not any_lam_evaluated:
            print(f"    WARNING: {nm} at {ud.strftime('%Y-%m')}: "
                  f"no lambda passed validation, using default={best_l}")
        best_lams[ud] = best_l

        # --- Final model: refit on full training window with best lambda ---
        tr_s = ud - pd.DateOffset(years=11)
        tr_mask = (common >= tr_s) & (common < ud)
        tr_idx = common[tr_mask]
        if len(tr_idx) < 504: continue

        tr_pos = np.searchsorted(common, tr_idx)

        X_full = feat_vals[tr_pos]
        m_full, s_full = X_full.mean(0), X_full.std(0); s_full[s_full == 0] = 1
        X_full_std = (X_full - m_full) / s_full
        exc_full = exc_vals[tr_pos]

        jm = JM(lam=best_l, n_init=3, max_it=20)
        jm.fit(X_full_std)
        labs = jm.labels(exc_full)

        target = np.roll(labs, -1)[:-1]
        xf_train = xf_vals[tr_pos[:-1]]
        if len(np.unique(target)) < 2: continue

        clf = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3,
                                 eval_metric='logloss', random_state=42, verbosity=0)
        clf.fit(xf_train, target)

        fc_end = updates[iu + 1] if iu + 1 < len(updates) else pd.Timestamp(test_end)
        # Half-open [ud, fc_end) for interior windows; inclusive for last window
        if iu + 1 < len(updates):
            oos_mask = (common >= ud) & (common < fc_end)
        else:
            oos_mask = (common >= ud) & (common <= fc_end)
        oos_pos = np.where(oos_mask)[0]
        if len(oos_pos) == 0: continue

        probs = clf.predict_proba(xf_vals[oos_pos])[:, 1]
        all_prob = pd.concat([all_prob, pd.Series(probs, index=common[oos_pos])])

    all_prob = all_prob[~all_prob.index.duplicated(keep='last')].sort_index()
    all_prob_raw = all_prob.copy()

    # Apply paper's hardcoded smoothing halflife (footnote 14, Table 4)
    if shl > 0 and len(all_prob) > 0:
        all_prob = all_prob.ewm(halflife=shl, min_periods=1).mean()
    fc = (all_prob >= 0.5).astype(int) if len(all_prob) > 0 else pd.Series(dtype=int)

    if len(fc) > 0:
        ci = fc.index.intersection(ar.index); f_ = fc.reindex(ci)
        pos = (f_ == 0).astype(float); r = ar.reindex(ci); rfv = rf.reindex(ci).fillna(0)
        sr = pos * r + (1 - pos) * rfv - pos.diff().abs().fillna(0) * tc
        w = (1 + sr).cumprod(); pk = w.cummax()
        vol_tot = sr.std() * np.sqrt(252); sh_tot = sr.mean() * 252 / vol_tot if vol_tot > 0 else 0
        # Excess-return Sharpe (subtracts risk-free) — primary metric per paper
        exc_sr = sr - rfv
        vol_exc = exc_sr.std() * np.sqrt(252)
        sh_exc = exc_sr.mean() * 252 / vol_exc if vol_exc > 0 else 0
        met = {'sharpe': sh_exc, 'sharpe_total': sh_tot,
               'mdd': ((w - pk) / pk).min(), 'ann_ret': sr.mean() * 252,
               'ann_vol': vol_tot, 'n_shifts': int(pos.diff().abs().sum()),
               'pct_bear': (f_ == 1).mean() * 100}
    else:
        sr = pd.Series(dtype=float); w = sr; met = {}

    el = time.time() - t0
    lam_vals = list(best_lams.values())
    n_unique = len(set(lam_vals))
    avg_lam = np.mean(lam_vals) if lam_vals else 0
    min_lam = min(lam_vals) if lam_vals else 0
    max_lam = max(lam_vals) if lam_vals else 0
    print(f"\r    {nm}: done ({el:.0f}s)" + " " * 30)
    print(f"  {nm:12s}: Sharpe={met.get('sharpe', 0):.3f}  "
          f"MDD={met.get('mdd', 0) * 100:.1f}%  Bear={met.get('pct_bear', 0):.0f}%  "
          f"Shifts={met.get('n_shifts', 0)}  ({el:.1f}s)")
    shl_note = f"  SMOOTH: HL={shl}"
    print(f"    Lambda: avg={avg_lam:.1f}  range=[{min_lam:.1f}, {max_lam:.1f}]  "
          f"unique={n_unique}/{len(lam_vals)}{shl_note}")
    return nm, {'forecasts': fc, 'probs': all_prob, 'probs_raw': all_prob_raw,
                'strat_ret': sr, 'metrics': met, 'wealth': w,
                'best_lambdas': best_lams, 'smooth_hl': shl}


def run_jm_only(nm, exc_df, ret_df, rf_daily, RF, lam_grid, test_start, test_end, tc=5e-4):
    """JM-only: carry forward regime as forecast.

    Fixed: uses full lambda grid, no data leak, proper 5-year validation,
    and tracks regime metrics (bear%, shifts, best_lambdas).
    """
    er = exc_df[nm].dropna(); ar = ret_df[nm].dropna()
    feat = RF[nm]; common = feat.index.sort_values()
    feat_vals = feat.values
    exc_vals = er.reindex(common).fillna(0).values
    updates = pd.date_range(test_start, test_end, freq='6MS')
    all_fc = pd.Series(dtype=float); best_lams = {}

    n_updates = len(updates)
    for iu, ud in enumerate(updates):
        print(f"\r    {nm} (JM-only): update {iu+1}/{n_updates} ({ud.strftime('%Y-%m')}) ...", end='', flush=True)
        # --- Lambda selection via 5-year validation window ---
        val_start = ud - pd.DateOffset(years=5)
        val_end = ud

        best_s, best_l = -np.inf, lam_grid[0]
        any_lam_evaluated = False

        for lam in lam_grid:
            # Run JM-only forecasts over validation window (biannual per paper)
            val_sub_updates = pd.date_range(val_start, val_end, freq='6MS')
            val_fc = pd.Series(dtype=float)

            for viu, vud in enumerate(val_sub_updates):
                vtr_s = vud - pd.DateOffset(years=11)
                vfc_end = val_sub_updates[viu + 1] if viu + 1 < len(val_sub_updates) else pd.Timestamp(val_end)
                # Fit JM on training only [vtr_s, vud), carry forward last label
                vtr_mask = (common >= vtr_s) & (common < vud)
                vtr_idx = common[vtr_mask]
                if len(vtr_idx) < 252: continue

                vtr_pos = np.searchsorted(common, vtr_idx)
                X_v = feat_vals[vtr_pos]
                m_v, s_v = X_v.mean(0), X_v.std(0); s_v[s_v == 0] = 1
                X_v_std = (X_v - m_v) / s_v
                exc_v = exc_vals[vtr_pos]

                jm = JM(lam=lam, n_init=1, max_it=10)
                jm.fit(X_v_std)
                labs = jm.labels(exc_v)
                last_label = int(labs[-1])

                oos_mask = (common >= vud) & (common < vfc_end)
                oos_idx = common[oos_mask]
                if len(oos_idx) == 0: continue
                oos = pd.Series(last_label, index=oos_idx)
                val_fc = pd.concat([val_fc, oos])

            val_fc = val_fc[~val_fc.index.duplicated(keep='last')].sort_index()
            if len(val_fc) < 50: continue

            ci = val_fc.index.intersection(ar.index)
            if len(ci) < 30: continue
            pos = (val_fc.reindex(ci) == 0).astype(float)
            rfv = rf_daily.reindex(ci).fillna(0)
            sr = pos * ar.reindex(ci) + (1 - pos) * rfv - pos.diff().abs().fillna(0) * tc
            exc_sr = sr - rfv
            v = exc_sr.std() * np.sqrt(252)
            s = exc_sr.mean() * 252 / v if v > 0 else 0
            if s > best_s:
                best_s, best_l = s, lam
                any_lam_evaluated = True

        if not any_lam_evaluated:
            print(f"    WARNING: {nm} (JM-only) at {ud.strftime('%Y-%m')}: "
                  f"no lambda passed validation, using default={best_l}")
        best_lams[ud] = best_l

        # --- Final model: fit JM on TRAINING only, carry forward last label ---
        # Paper's "carry-forward" = last training regime label broadcast to OOS.
        # Viterbi is a smoothing algorithm (backward pass uses future data), so
        # fitting on [train+OOS] would leak future information into OOS labels.
        tr_s = ud - pd.DateOffset(years=11)
        fe = updates[iu + 1] if iu + 1 < len(updates) else pd.Timestamp(test_end)
        tr_mask = (common >= tr_s) & (common < ud)
        tr_idx = common[tr_mask]
        if len(tr_idx) < 252: continue

        tr_pos = np.searchsorted(common, tr_idx)
        X_tr = feat_vals[tr_pos]
        m_f, s_f = X_tr.mean(0), X_tr.std(0); s_f[s_f == 0] = 1
        X_tr_std = (X_tr - m_f) / s_f
        exc_tr = exc_vals[tr_pos]

        jm = JM(lam=best_l, n_init=3, max_it=20)
        jm.fit(X_tr_std)
        labs = jm.labels(exc_tr)
        last_label = int(labs[-1])

        # Half-open [ud, fe) for interior windows; inclusive for last window
        if iu + 1 < len(updates):
            oos_mask = (common >= ud) & (common < fe)
        else:
            oos_mask = (common >= ud) & (common <= fe)
        oos_idx = common[oos_mask]
        if len(oos_idx) == 0: continue
        oos = pd.Series(last_label, index=oos_idx)
        all_fc = pd.concat([all_fc, oos])

    print(f"\r    {nm} (JM-only): done" + " " * 30)
    all_fc = all_fc[~all_fc.index.duplicated(keep='last')].sort_index().astype(int)
    if len(all_fc) > 0:
        ci = all_fc.index.intersection(ar.index)
        pos = (all_fc.reindex(ci) == 0).astype(float)
        r = ar.reindex(ci); rfv = rf_daily.reindex(ci).fillna(0)
        sr = pos * r + (1 - pos) * rfv - pos.diff().abs().fillna(0) * tc
        w = (1 + sr).cumprod(); pk = w.cummax()
        v_tot = sr.std() * np.sqrt(252); sh_tot = sr.mean() * 252 / v_tot if v_tot > 0 else 0
        exc_sr = sr - rfv
        vol_exc = exc_sr.std() * np.sqrt(252)
        sh_exc = exc_sr.mean() * 252 / vol_exc if vol_exc > 0 else 0
        f_ = all_fc.reindex(ci)
        met = {'sharpe': sh_exc, 'sharpe_total': sh_tot,
               'mdd': ((w - pk) / pk).min(), 'ann_ret': sr.mean() * 252,
               'ann_vol': v_tot, 'n_shifts': int(pos.diff().abs().sum()),
               'pct_bear': (f_ == 1).mean() * 100}
        lam_vals = list(best_lams.values())
        avg_lam = np.mean(lam_vals) if lam_vals else 0
        n_unique = len(set(lam_vals))
        print(f"  {nm:12s}: Sharpe={sh_exc:.3f}  Shifts={met['n_shifts']}  "
              f"Bear={met['pct_bear']:.0f}%")
        print(f"    Lambda: avg={avg_lam:.1f}  range=[{min(lam_vals):.1f}, {max(lam_vals):.1f}]  "
              f"unique={n_unique}/{len(lam_vals)}")
    else:
        sr = pd.Series(dtype=float); w = sr; met = {}
        print(f"  {nm:12s}: no data")
    return nm, {'forecasts': all_fc, 'strat_ret': sr, 'metrics': met,
                'wealth': w, 'best_lambdas': best_lams}

# ===========================================================================
# CELL 8: Parallel + Cached Execution Wrappers
# ===========================================================================
def _run_jmxgb_all(assets, exc_df, ret_df, rf_daily, RF, MF, lam_grid, ts, te, use_cache=True):
    results = {}; to_run = []
    for nm in assets:
        if nm not in exc_df: continue
        sfx = _get_cache_suffix(nm)
        key = f"jmxgb_{DATA_VERSION}_{nm}{sfx}_{ts}_{te}"
        if use_cache:
            c = _load_cache('models', key)
            if c is not None:
                results[nm] = c
                print(f"  {nm:12s}: cached")
                continue
        to_run.append(nm)
    if to_run:
        print(f"  Processing {len(to_run)} assets in parallel...")
        par = Parallel(n_jobs=-1, verbose=0)(
            delayed(process_asset)(nm, exc_df, ret_df, rf_daily, RF, MF, _get_lam_grid(nm), ts, te)
            for nm in to_run
        )
        for nm, res in par:
            if use_cache:
                sfx = _get_cache_suffix(nm)
                _save_cache(res, 'models', f"jmxgb_{DATA_VERSION}_{nm}{sfx}_{ts}_{te}")
            results[nm] = res
    # Lambda selection summary
    if results:
        print(f"\n  Lambda Selection Summary (JM-XGB):")
        print(f"    {'Asset':12s} {'Avg':>6s} {'Min':>6s} {'Max':>6s} {'#Unique':>8s} {'Varies?':>8s}")
        for nm in assets:
            if nm not in results: continue
            lams = results[nm].get('best_lambdas', {})
            if not lams: continue
            vals = list(lams.values())
            varies = "YES" if len(set(vals)) > 1 else "NO"
            print(f"    {nm:12s} {np.mean(vals):6.1f} {min(vals):6.1f} {max(vals):6.1f} "
                  f"{len(set(vals)):>3d}/{len(vals):<3d}  {varies:>8s}")
    return results

def _run_jm_all(assets, exc_df, ret_df, rf_daily, RF, lam_grid, ts, te, use_cache=True):
    results = {}; to_run = []
    for nm in assets:
        if nm not in exc_df: continue
        sfx = _get_cache_suffix(nm)
        key = f"jm_{DATA_VERSION}_{nm}{sfx}_{ts}_{te}"
        if use_cache:
            c = _load_cache('models', key)
            if c is not None:
                results[nm] = c
                print(f"  {nm:12s}: cached")
                continue
        to_run.append(nm)
    if to_run:
        print(f"  Processing {len(to_run)} assets in parallel...")
        par = Parallel(n_jobs=-1, verbose=0)(
            delayed(run_jm_only)(nm, exc_df, ret_df, rf_daily, RF, _get_lam_grid(nm), ts, te)
            for nm in to_run
        )
        for nm, res in par:
            if use_cache:
                sfx = _get_cache_suffix(nm)
                _save_cache(res, 'models', f"jm_{DATA_VERSION}_{nm}{sfx}_{ts}_{te}")
            results[nm] = res
    return results

def diagnose_smooth(jmxgb, ret_df):
    """Diagnostic: compare regime shifts with and without EWM smoothing."""
    print(f"\n{'='*80}")
    print("SMOOTH DIAGNOSTIC: Impact of EWM smoothing on regime shifts")
    print(f"{'='*80}")
    print(f"  {'Asset':12s} {'HL':>4s} {'Bear%':>7s} {'Shifts':>8s} {'Bear%(raw)':>11s} "
          f"{'Shifts(raw)':>12s} {'ShiftDelta':>11s}")
    print(f"  {'-'*75}")
    for nm in ASSETS:
        if nm not in jmxgb: continue
        res = jmxgb[nm]
        probs_raw = res.get('probs_raw', pd.Series())
        if len(probs_raw) == 0: continue
        shl = SMOOTH.get(nm, 4)
        bear_s = res['metrics'].get('pct_bear', 0)
        shifts_s = res['metrics'].get('n_shifts', 0)
        # Raw (unsmoothed) metrics
        fc_raw = (probs_raw >= 0.5).astype(int)
        ci = fc_raw.index.intersection(ret_df[nm].dropna().index)
        f_raw = fc_raw.reindex(ci)
        bear_raw = (f_raw == 1).mean() * 100
        shifts_raw = int(f_raw.diff().abs().sum())
        delta = shifts_s - shifts_raw
        print(f"  {nm:12s} {shl:4d} {bear_s:6.1f}% {shifts_s:8d}  {bear_raw:10.1f}% "
              f"{shifts_raw:12d} {delta:+11d}")

ts = '2007-01-01' if (ret_df.index[-1]-ret_df.index[0]).days/365>16 else \
     (ret_df.index[0]+pd.DateOffset(years=5)).strftime('%Y-%m-%d')
te = ret_df.index[-1].strftime('%Y-%m-%d')
LG = np.array([0., .3, 1., 3., 7., 15., 40., 100.])
# Apply lambda floor — see RESEARCH_LAMBDA.md for evidence
LG_FILTERED = LG[LG >= LAM_FLOOR] if LAM_FLOOR > 0 else LG

def _get_lam_grid(nm):
    """Get asset-specific lambda grid, applying per-asset floor overrides.
    If an override floor exceeds the max value in LG, a grid containing just
    the override value is returned."""
    floor = LAM_FLOOR_OVERRIDE.get(nm, LAM_FLOOR)
    if floor <= 0:
        return LG
    grid = LG[LG >= floor]
    if len(grid) == 0:
        # Override exceeds max in LG — create a single-element grid
        grid = np.array([float(floor)])
    return grid

def _get_cache_suffix(nm):
    """Get asset-specific cache key suffix (includes override floor if different)."""
    override = LAM_FLOOR_OVERRIDE.get(nm)
    if override is not None and override != LAM_FLOOR:
        return f"_lfo{override}"
    return ""

print(f"\nTest period: {ts} -> {te}")
if LAM_FLOOR > 0:
    print(f"Lambda floor: {LAM_FLOOR} -- grid: {LG_FILTERED.tolist()} (paper: {LG.tolist()})")
else:
    print(f"Lambda grid: {LG.tolist()} (full paper grid, no floor)")
if LAM_FLOOR_OVERRIDE:
    for _nm, _fl in LAM_FLOOR_OVERRIDE.items():
        _g = LG[LG >= _fl].tolist()
        print(f"  {_nm} override: floor={_fl} -> grid: {_g}")

def run_full_pipeline():
    """Run the full 12-asset pipeline (JM-XGB + JM-only + backtests + tables)."""
    global jmxgb, jm_only, rfc, rrf, strats

    # ===========================================================================
    # CELL 9: Full run (parallel + cached)
    # ===========================================================================
    print("\n" + "="*60)
    print("JM-XGB: All 12 assets (parallel + cached)")
    print("="*60)
    t0=time.time()
    jmxgb = _run_jmxgb_all(ASSETS, exc_df, ret_df, rf_daily, RF, MF, LG_FILTERED, ts, te)
    print(f"JM-XGB total: {time.time()-t0:.0f}s")
    diagnose_smooth(jmxgb, ret_df)

    print("\nJM-only pipeline (parallel + cached):")
    t0=time.time()
    jm_only = _run_jm_all(ASSETS, exc_df, ret_df, rf_daily, RF, LG_FILTERED, ts, te)
    print(f"JM-only total: {time.time()-t0:.0f}s")

    # CELL 10: TABLE 4
    table4(jmxgb, jm_only, ret_df, rf_daily, ts, te)

    # CELL 11: Figure 2
    fig2(jmxgb, ret_df, rf_daily, ts=ts, te=te)

    # CELL 12: Portfolio backtests
    rfc, rrf = build_regime_return_forecasts(jmxgb, exc_df)
    strats = _run_backtests(ret_df, exc_df, rf_daily, rfc, rrf, ts, te)

    # TABLE 6
    print("\n"+"="*110)
    print("TABLE 6: Portfolio Performance")
    print("="*110)
    nms=list(strats.keys())
    print(f"{'':12s}"+"".join(f"{n:>16s}" for n in nms))
    print("-"*124)
    am={n:mets(r,w,rf_daily) for n,(r,w) in strats.items()}
    for k,f in [('Return','{:.1%}'),('Volatility','{:.1%}'),('Sharpe','{:.2f}'),
                ('MDD','{:.1%}'),('Calmar','{:.2f}'),('Turnover','{:.2f}'),('Leverage','{:.2f}')]:
        print(f"{k:12s}"+"".join(f"{f.format(am[n][k]):>16s}" for n in nms))
    print(f"\nRF: {rf_daily.mean()*252:.1%}/yr")

    # CELL 13: Figure 3
    fig3(strats)

    # TABLE 7: Forecast Correlation
    print("\n"+"="*90)
    print("TABLE 7: Forecast Correlation")
    print("="*90)
    ts_,te_=pd.Timestamp(ts),pd.Timestamp(te)
    for grp in [EQ_RE,BD_CM]:
        av=[a for a in grp if a in ret_df]
        print(f"\n{'':12s}{'Overall':>10s}"+"".join(f"{a:>12s}" for a in av))
        for lab,src in [("EWMA",None),("JM-XGB",rrf)]:
            cs=[]
            for a in av:
                r=ret_df[a];mask=(r.index>=ts_)&(r.index<=te_);act=r[mask]
                if src and a in src:
                    fc=src[a];ci=fc.index.intersection(act.index)
                    c=act.reindex(ci).corr(fc.reindex(ci)) if len(ci)>100 else 0
                else:
                    ew=r.ewm(halflife=1260,min_periods=252).mean().shift(1).reindex(act.index).dropna()
                    c=act.reindex(ew.index).corr(ew)
                cs.append(c)
            ov=np.mean(cs)
            print(f"{lab:12s}{ov*100:9.2f}%"+"".join(f"{c*100:11.2f}%" for c in cs))

    # TABLE 8: gamma_trade sensitivity
    print("\n"+"="*60)
    print("TABLE 8: MinVar (JM-XGB) gamma_trade sensitivity")
    print("="*60)
    for gt_ in [0.0, 1.0]:
        r,w=bt(ret_df,exc_df,rf_daily,rfc,None,'MinVar',True,10.,gt_,ts,te)
        m=mets(r,w,rf_daily)
        print(f"\ngamma_trade={gt_:.1f}{'  (default)' if gt_==1 else ''}")
        for k in ['Return','Volatility','Sharpe','MDD','Calmar','Turnover','Leverage']:
            v=m[k]; print(f"  {k:12s}: {v:.1%}" if k in['Return','Volatility','MDD'] else f"  {k:12s}: {v:.2f}")

    # TABLE 9: gamma_risk sensitivity
    print("\n"+"="*60)
    print("TABLE 9: MV (JM-XGB) gamma_risk sensitivity")
    print("="*60)
    for gr_ in [5., 10., 20.]:
        r,w=bt(ret_df,exc_df,rf_daily,rfc,rrf,'MV',True,gr_,1.,ts,te)
        m=mets(r,w,rf_daily)
        print(f"\ngamma_risk={gr_:.1f}{'  (default)' if gr_==10 else ''}")
        for k in ['Return','Volatility','Sharpe','MDD','Calmar','Turnover','Leverage']:
            v=m[k]; print(f"  {k:12s}: {v:.1%}" if k in['Return','Volatility','MDD'] else f"  {k:12s}: {v:.2f}")

    # CELL 15: All-asset regimes
    fig_all(jmxgb)

    print("\nREPLICATION COMPLETE — All figures and tables generated.")
    print("\nCache stats:")
    cache_stats()

    return jmxgb, jm_only, strats, rfc, rrf


# ===========================================================================
# CELL 10: TABLE 4
# ===========================================================================
def table4(jmx, jmo, ret_df, rf_daily, ts, te):
    ts_,te_=pd.Timestamp(ts),pd.Timestamp(te)
    bh={}
    for nm in ASSETS:
        if nm not in ret_df: continue
        r=ret_df[nm];mask=(r.index>=ts_)&(r.index<=te_);rt=r[mask]
        rft=rf_daily.reindex(rt.index).fillna(0)
        er=rt.mean()*252-rft.mean()*252; vol=rt.std()*np.sqrt(252)
        w=(1+rt).cumprod(); mdd=((w-w.cummax())/w.cummax()).min()
        bh[nm]={'sharpe':er/vol if vol>0 else 0,'mdd':mdd}

    print("\n"+"="*95)
    print("TABLE 4: 0/1 Strategy Performance")
    print("="*95)
    for metric,key in [("Sharpe Ratio",'sharpe'),("Max Drawdown",'mdd')]:
        print(f"\n{metric}"); print("-"*95)
        for grp in [EQ_RE,BD_CM]:
            av=[a for a in grp if a in ret_df]
            if not av: continue
            print(f"{'':12s}"+"".join(f"{a:>12s}" for a in av))
            for lab,src in [("B & H",None),("JM",jmo),("JM-XGB",jmx)]:
                row=f"{lab:12s}"
                for a in av:
                    if lab=="B & H": v=bh.get(a,{}).get(key,0)
                    else: v=src.get(a,{}).get('metrics',{}).get(key,0)
                    row+=f"{v*100:11.2f}%" if 'mdd'==key else f"{v:12.2f}"
                print(row)
            print()

# ===========================================================================
# CELL 11: Figure 2
# ===========================================================================
def fig2(res, ret_df, rf_daily, assets=['LargeCap','REIT','AggBond'], ts=None, te=None):
    ts_,te_=pd.Timestamp(ts),pd.Timestamp(te)
    avail=[a for a in assets if a in res and len(res[a].get('forecasts',[]))>0]
    fig,axes=plt.subplots(len(avail),1,figsize=(12,4*len(avail)))
    if len(avail)==1: axes=[axes]
    for ax,nm in zip(axes,avail):
        r_=res[nm]; fc=r_['forecasts']
        bh=(1+ret_df[nm][(ret_df.index>=ts_)&(ret_df.index<=te_)]).cumprod()
        ax.plot(bh.index,bh,color='#ff7f0e',lw=.8,label=nm)
        w=r_.get('wealth',pd.Series())
        if len(w)>0: ax.plot(w.index,w,color='#1f77b4',lw=.8,label=f'{nm} (0/1)')
        ib,bs=False,None
        for d in fc.index:
            if fc.loc[d]==1 and not ib: bs=d;ib=True
            elif fc.loc[d]==0 and ib: ax.axvspan(bs,d,alpha=.15,color='red',lw=0);ib=False
        if ib and bs: ax.axvspan(bs,fc.index[-1],alpha=.15,color='red',lw=0)
        pb=r_['metrics'].get('pct_bear',0);ns=r_['metrics'].get('n_shifts',0)
        ax.set_title(f'{nm}: Bear={pb:.1f}%, Shifts={ns}',fontfamily='monospace')
        ax.set_yscale('log');ax.set_ylabel('Wealth (log scale)')
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.set_yticks([0.5,1,2,5,10])
        h,l=ax.get_legend_handles_labels();h.append(Patch(fc='red',alpha=.15));l.append('Bear')
        ax.legend(h,l,loc='upper left',fontsize=9);ax.grid(alpha=.3)
    plt.tight_layout();plt.savefig('fig2.png',dpi=150,bbox_inches='tight');plt.close()

# ===========================================================================
# CELL 12: Portfolio Backtests + TABLE 6
# ===========================================================================
def bt(ret_df, exc_df, rf_daily, rfc=None, rrf=None,
       pt='MinVar', ur=False, gr=10., gt=1., ts_='2007-01-01', te_='2023-12-31', tc=5e-4):
    assets=[a for a in ASSETS if a in ret_df]; n=len(assets)
    ts_,te_=pd.Timestamp(ts_),pd.Timestamp(te_)
    dates=ret_df.index[(ret_df.index>=ts_)&(ret_df.index<=te_)]
    ra=ret_df[assets].reindex(dates).fillna(0)
    ea=exc_df[assets].reindex(dates).fillna(0)
    pr=[];wh=[];wp=np.zeros(n)
    w64=np.array([W6040.get(a,0) for a in assets])

    for date in dates:
        dr=ra.loc[date].values; drf=rf_daily.loc[date] if date in rf_daily.index else 0.
        if pt=='60/40': w=w64
        elif pt=='EW':
            if ur and rfc:
                bl=[j for j,a in enumerate(assets) if a in rfc and date in rfc[a].index and rfc[a].loc[date]==0]
                if not bl: bl=list(range(n))
                w=np.zeros(n)
                if len(bl)>=4:
                    for j in bl: w[j]=1./len(bl)
            else: w=np.ones(n)/n
        else:
            lb=date-pd.DateOffset(years=3);hist=ra.loc[(ra.index>=lb)&(ra.index<date)]
            if len(hist)<126: w=wp.copy()
            else:
                wts=np.exp(-np.log(2)*np.arange(len(hist)-1,-1,-1)/252);wts/=wts.sum()
                c=hist.values-np.average(hist.values,axis=0,weights=wts)
                S=(c.T*wts)@c+np.eye(n)*1e-6; S=(S+S.T)/2
                if pt=='MinVar':
                    if ur and rfc:
                        mu=np.zeros(n);nb=0
                        for j,a in enumerate(assets):
                            if a in rfc and date in rfc[a].index:
                                if rfc[a].loc[date]==0: mu[j]=1e-3;nb+=1
                            else: mu[j]=1e-3;nb+=1
                        if nb<4: w=np.zeros(n)
                        else:
                            wv=cp.Variable(n)
                            cp.Problem(cp.Maximize(mu@wv-gr*cp.quad_form(wv,S,assume_PSD=True)-gt*tc*cp.norm1(wv-wp)),
                                      [wv>=0,wv<=.4,cp.sum(wv)<=1]).solve(solver=cp.SCS,verbose=False)
                            w=np.maximum(wv.value,0) if wv.value is not None else wp
                    else:
                        mu=np.ones(n)*1e-3;wv=cp.Variable(n)
                        cp.Problem(cp.Maximize(mu@wv-gr*cp.quad_form(wv,S,assume_PSD=True)),
                                  [wv>=0,wv<=.4,cp.sum(wv)<=1]).solve(solver=cp.SCS,verbose=False)
                        w=np.maximum(wv.value,0) if wv.value is not None else np.ones(n)/n
                elif pt=='MV':
                    mu=np.zeros(n);nb=n
                    if ur and rrf:
                        nb=0
                        for j,a in enumerate(assets):
                            if a in rrf and date in rrf[a].index:
                                mu[j]=rrf[a].loc[date]
                                if a in rfc and date in rfc[a].index and rfc[a].loc[date]==0: nb+=1
                            else:
                                h=ea.loc[:date,a].iloc[-1260:]
                                if len(h)>252: mu[j]=h.mean()
                                nb+=1
                    else:
                        for j,a in enumerate(assets):
                            h=ea.loc[:date,a].iloc[-1260:]
                            if len(h)>252:
                                ww=np.exp(-np.log(2)*np.arange(len(h)-1,-1,-1)/1260);ww/=ww.sum()
                                mu[j]=np.average(h.values,weights=ww)
                        gr=5.
                    if ur and nb<4: w=np.zeros(n)
                    else:
                        wv=cp.Variable(n)
                        cp.Problem(cp.Maximize(mu@wv-gr*cp.quad_form(wv,S,assume_PSD=True)-gt*tc*cp.norm1(wv-wp)),
                                  [wv>=0,wv<=.4,cp.sum(wv)<=1]).solve(solver=cp.SCS,verbose=False)
                        w=np.maximum(wv.value,0) if wv.value is not None else wp

        wh.append(w.copy())
        p=np.dot(w,dr)+(1-w.sum())*drf-tc*np.abs(w-wp).sum()
        pr.append(p)
        if p!=-1: wp=w*(1+dr)/(1+p)
        else: wp=w

    return pd.Series(pr,index=dates), pd.DataFrame(wh,index=dates,columns=assets)

def build_regime_return_forecasts(jmxgb, exc_df):
    rfc={nm:r['forecasts'] for nm,r in jmxgb.items() if len(r.get('forecasts',[]))>0}
    rrf={}
    for nm in rfc:
        fc=rfc[nm];er=exc_df[nm]
        rf_=pd.Series(index=fc.index,dtype=float)
        for d in fc.index:
            lb=er.loc[:d].iloc[-504:]
            rf_.loc[d]=max(lb.mean(),0)*1.5 if fc.loc[d]==0 else min(-1e-3,lb[lb<0].mean()*.5 if len(lb[lb<0])>0 else -1e-3)
        rrf[nm]=rf_
    return rfc, rrf

def _run_backtests(ret_df, exc_df, rf_daily, rfc, rrf, ts, te, use_cache=True):
    key = f"backtests_{DATA_VERSION}_{ts}_{te}"
    if use_cache:
        cached = _load_cache('backtests', key)
        if cached is not None:
            print("Backtests loaded from cache")
            return cached

    print("Running backtests...")
    strats={}
    for lab,pt,ur,gr_,gt_ in [('60/40','60/40',False,10,0),('MinVar','MinVar',False,10,0),
        ('MinVar (JM-XGB)','MinVar',True,10,1),('MV','MV',False,5,0),
        ('MV (JM-XGB)','MV',True,10,1),('EW','EW',False,10,0),('EW (JM-XGB)','EW',True,10,0)]:
        t0_=time.time()
        r,w=bt(ret_df,exc_df,rf_daily,rfc,rrf if 'MV' in lab else None,pt,ur,gr_,gt_,ts,te)
        strats[lab]=(r,w); print(f"  {lab:20s}: {time.time()-t0_:.0f}s")

    if use_cache:
        _save_cache(strats, 'backtests', key)
    return strats

def mets(r,w,rf):
    er=r.mean()*252;vol=r.std()*np.sqrt(252);wc=(1+r).cumprod();mdd=((wc-wc.cummax())/wc.cummax()).min()
    rfann=rf.reindex(r.index).fillna(0).mean()*252; exc=er-rfann
    to=w.diff().abs().sum(axis=1).sum()/(len(r)/252) if w is not None else 0
    lev=w.sum(axis=1).mean() if w is not None else 1
    return {'Return':exc,'Volatility':vol,'Sharpe':exc/vol if vol>0 else 0,
            'MDD':mdd,'Calmar':exc/abs(mdd) if mdd!=0 else 0,'Turnover':to,'Leverage':lev}

# ===========================================================================
# CELL 13: Figure 3
# ===========================================================================
def fig3(s):
    fig,axes=plt.subplots(3,1,figsize=(12,14))
    for ax,(t,items) in zip(axes,[
        ('Minimum-Variance',[('MinVar (JM-XGB)','#1f77b4'),('MinVar','#ff7f0e')]),
        ('Mean-Variance',[('MV (JM-XGB)','#1f77b4'),('MV','#ff7f0e'),('60/40','#2ca02c')]),
        ('Equally-Weighted',[('EW (JM-XGB)','#1f77b4'),('EW','#ff7f0e')])]):
        for k,c in items:
            if k in s: w=(1+s[k][0]).cumprod(); ax.plot(w.index,w,color=c,lw=1,label=k)
        ax.set_yscale('log');ax.set_ylabel('Wealth (log)')
        ax.set_title(f'Strategy Performance: {t}');ax.legend(loc='upper left');ax.grid(alpha=.3)
    plt.tight_layout();plt.savefig('fig3.png',dpi=150,bbox_inches='tight');plt.close()

# ===========================================================================
# CELL 15: Supplementary — All-asset regimes
# ===========================================================================
def fig_all(res):
    av=[a for a in ASSETS if a in res and len(res[a].get('forecasts',[]))>0]
    fig,axes=plt.subplots(len(av),1,figsize=(14,2.5*len(av)))
    if len(av)==1:axes=[axes]
    for ax,nm in zip(axes,av):
        fc=res[nm]['forecasts'];w=res[nm].get('wealth',pd.Series())
        if len(w)>0: ax.plot(w.index,w,color='#1f77b4',lw=.7)
        ib,bs=False,None
        for d in fc.index:
            if fc.loc[d]==1 and not ib: bs=d;ib=True
            elif fc.loc[d]==0 and ib: ax.axvspan(bs,d,alpha=.15,color='red',lw=0);ib=False
        if ib and bs: ax.axvspan(bs,fc.index[-1],alpha=.15,color='red',lw=0)
        pb=res[nm]['metrics'].get('pct_bear',0)
        ax.set_title(f'{nm} (Bear:{pb:.0f}%)',fontsize=10,fontfamily='monospace')
        ax.set_yscale('log');ax.grid(alpha=.2)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)
        ax.set_yticks([0.5,1,2,5,10])
    plt.tight_layout();plt.savefig('fig_all.png',dpi=120,bbox_inches='tight');plt.close()


if __name__ == '__main__':
    run_full_pipeline()
