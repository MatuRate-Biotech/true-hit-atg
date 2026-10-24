import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split

from . import schema as s
from .config import CFG

CPM = 1e6


def lib(df, rnd, camp=None):
    if camp in s.LIBS and rnd in s.LIBS[camp]:
        return s.LIBS[camp][rnd]
    return int(df[s.cnt(rnd)].sum())


def norm_ct(df, rnd, camp=None, floor=None):
    fl = CFG.cpm_floor if floor is None else floor
    n = df[s.cnt(rnd)] * CPM / lib(df, rnd, camp)
    return n.clip(lower=fl).fillna(fl)


def log_ct(df, rnd, camp=None, floor=None):
    return np.log10(norm_ct(df, rnd, camp, floor))


def log_ratio(df, late, early, camp=None, floor=None):
    return log_ct(df, late, camp, floor) - log_ct(df, early, camp, floor)


def has_read(df, rnd, fil=None):
    return df[s.cnt(rnd)] >= (CFG.fil_cnt if fil is None else fil)


def binarize(lr, elig, tr=None, seed=None):
    tr = CFG.gmm_tr if tr is None else tr
    sd = CFG.gmm_seed if seed is None else seed
    out = pd.Series(np.nan, index=lr.index, dtype=float)
    m = elig & lr.notna() & (lr != 0)
    if m.sum() < 2:
        return out
    v = lr[m].values.reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=sd).fit(v)
    lo, hi = gm.means_.argsort(0).squeeze()
    ps = gm.predict_proba(v)
    asg = np.full(len(ps), np.nan)
    asg[ps[:, 0] > tr] = lo
    asg[ps[:, 1] > tr] = hi
    out[m] = asg
    return out


def make_split(lb, test_size=None, seed=None):
    ts = CFG.test_size if test_size is None else test_size
    sd = CFG.split_seed if seed is None else seed
    out = pd.Series(np.nan, index=lb.index, dtype=object)
    idx = lb.index[lb.isin([0.0, 1.0])]
    if len(idx) < 2:
        return out
    tr, te = train_test_split(idx, test_size=ts, random_state=sd)
    out[tr] = "train"
    out[te] = "test"
    return out


def pair(df, late, early, camp=None, use_shipped_has=True):
    out = df.copy()
    out[s.ENR] = log_ratio(out, late, early, camp)
    if use_shipped_has and s.has(late) in out.columns and s.has(early) in out.columns:
        el = out[s.has(late)].astype(bool) | out[s.has(early)].astype(bool)
    else:
        el = has_read(out, late) | has_read(out, early)
    out[s.ENR_BIN] = binarize(out[s.ENR], el)
    out[s.SPLIT] = make_split(out[s.ENR_BIN])
    return out


def rank(df, col=None):
    return df.sort_values(col or s.ENR, ascending=False).reset_index(drop=True)


def baseline_auc(early_enr, late_bin):
    m = late_bin.isin([0.0, 1.0]) & early_enr.notna()
    if m.sum() < 2 or late_bin[m].nunique() < 2:
        return float("nan")
    return roc_auc_score(late_bin[m].astype(int), early_enr[m])
