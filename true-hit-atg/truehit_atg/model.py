import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from .config import CFG


def fit(X, y, lam=None, max_iter=None):
    lm = CFG.lam if lam is None else lam
    mi = CFG.max_iter if max_iter is None else max_iter
    m = LogisticRegression(
        random_state=0,
        penalty="l1",
        solver="saga",
        fit_intercept=True,
        C=1.0 / lm,
        max_iter=mi,
    )
    m.fit(X, np.asarray(y).astype(int))
    m.conv = bool(np.all(np.asarray(m.n_iter_) < mi))
    return m


def score(m, X):
    return m.predict_proba(X)[:, 1]


def auc(m, X, y):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score(m, X)))


def pr_auc(m, X, y):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score(m, X)))


def roc(m, X, y):
    fpr, tpr, _ = roc_curve(np.asarray(y).astype(int), score(m, X))
    return fpr, tpr


def coefs(m, names):
    b = m.coef_.ravel()
    nz = b != 0
    out = pd.DataFrame({"feat": np.asarray(names)[nz], "beta": b[nz]})
    out[["imgt", "aa"]] = out["feat"].str.rsplit("_", n=1, expand=True)
    return out.reindex(out["beta"].abs().sort_values(ascending=False).index).reset_index(drop=True)


# AUC within strata, weighted by comparable pairs. a predictor that is constant inside a
# stratum scores 0.500 here by construction, which is what makes this the comparison with
# the mutation-load confound removed. pooling ranks across strata instead biases low,
# because cross-stratum pairs contribute 0.5.
def cond_auc(y, p, strat):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    strat = np.asarray(strat)
    num = den = 0.0
    for v in np.unique(strat):
        m = strat == v
        yy = y[m]
        npos, nneg = int(yy.sum()), int((1 - yy).sum())
        if npos == 0 or nneg == 0:
            continue
        w = npos * nneg
        num += w * roc_auc_score(yy, p[m])
        den += w
    return num / den if den else float("nan")
