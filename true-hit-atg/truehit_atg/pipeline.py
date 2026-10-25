from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data, encode, enrich, model, schema as s
from .config import CFG

PRED = "pred_enr"
CLS = "pred_cls"


@dataclass
class Res:
    camp: str
    train: tuple
    test: tuple
    tbl: pd.DataFrame
    n_tr: int
    n_te: int
    n_abst: int
    n_feat: int
    auc_tr: float
    auc_te: float
    pr_te: float
    auc_strict: float
    n_strict: int
    auc_nmut: float
    auc_within: float
    auc_cnt: float
    roc: tuple
    pk: tuple
    base: float
    mix: tuple
    cf: pd.DataFrame
    ranked: pd.DataFrame
    conv: bool
    imgt: list = field(default_factory=list)
    wt: str = ""
    m: object = None


def n_mut(seqs, wt):
    return seqs.apply(lambda q: sum(a != b for a, b in zip(q, wt)))


def run(camp=None, train=None, test=None, lam=None, min_ct=None):
    camp = camp or CFG.camp
    tp = tuple(train or CFG.train)
    sp = tuple(test or CFG.test)
    raw = data.load(camp, pair=tp, min_ct=min_ct)
    a = enrich.pair(raw, tp[0], tp[1], camp)
    b = enrich.pair(raw, sp[0], sp[1], camp)

    imgt, wt = encode.wt_cols(camp)
    tr = a[(a[s.SPLIT] == "train") & a[s.ENR_BIN].notna()]
    te = b[(b[s.SPLIT] == "test") & b[s.ENR_BIN].notna()]
    if len(tr) < 10 or len(te) < 10:
        raise ValueError(
            f"{camp} {tp[0]}/{tp[1]} gives {len(tr)} train and {len(te)} test rows, too few to fit"
        )

    Xtr, nm, _ = encode.encode(tr[s.AA_ALN], imgt=imgt, wt=wt)
    Xte, _, _ = encode.encode(te[s.AA_ALN], imgt=imgt, wt=wt)
    m = model.fit(Xtr, tr[s.ENR_BIN], lam=lam)

    # upstream splits each round pair independently, so a variant can sit in the early pair's
    # train split and the late pair's test split at once. strict drops that overlap.
    keep = ~te[s.VAR].isin(set(tr[s.VAR])).values
    auc_st = model.auc(m, Xte[keep], te[s.ENR_BIN][keep]) if keep.sum() > 10 else float("nan")

    nmu = n_mut(te[s.AA_ALN], wt)
    auc_nm = enrich.baseline_auc(nmu, te[s.ENR_BIN])
    # mutation count cannot order variants that carry the same number of substitutions,
    # so within a load it scores 0.500 and whatever the model scores is sequence signal
    auc_wi = model.cond_auc(te[s.ENR_BIN], model.score(m, Xte), nmu)
    # counts baseline: circular, the label is arithmetic on these same counts. kept to show that.
    auc_ct = enrich.baseline_auc(a.loc[te.index, s.ENR], te[s.ENR_BIN])

    Xall, _, _ = encode.encode(a[s.AA_ALN], imgt=imgt, wt=wt)
    rk = a.copy()
    rk[PRED] = model.score(m, Xall)
    rk[CLS] = np.where(rk[PRED] >= 0.5, "enriched", "depleted")
    cols = [s.VAR, PRED, CLS, s.CDR3, s.ENR, s.cnt(tp[1]), s.cnt(tp[0])]
    ranked = rk.sort_values(PRED, ascending=False)[cols].reset_index(drop=True)

    return Res(
        camp=camp,
        train=tp,
        test=sp,
        tbl=a,
        n_tr=len(tr),
        n_te=len(te),
        n_abst=int(a[s.ENR_BIN].isna().sum()),
        n_feat=Xtr.shape[1],
        auc_tr=model.auc(m, Xtr, tr[s.ENR_BIN]),
        auc_te=model.auc(m, Xte, te[s.ENR_BIN]),
        pr_te=model.pr_auc(m, Xte, te[s.ENR_BIN]),
        auc_strict=auc_st,
        n_strict=int(keep.sum()),
        auc_nmut=auc_nm,
        auc_within=auc_wi,
        auc_cnt=auc_ct,
        roc=model.roc(m, Xte, te[s.ENR_BIN]),
        pk=model.prec_at_k(te[s.ENR_BIN], model.score(m, Xte)),
        base=float(te[s.ENR_BIN].mean()),
        mix=enrich.pair_mix(raw, tp[0], tp[1], camp),
        cf=model.coefs(m, nm),
        ranked=ranked,
        conv=m.conv,
        imgt=imgt,
        wt=wt,
        m=m,
    )


def score_seqs(res, seqs):
    qs = [q.strip().upper() for q in seqs if q.strip()]
    if not qs:
        raise ValueError("no sequences given")
    bad = [(i, len(q)) for i, q in enumerate(qs) if len(q) != len(res.wt)]
    if bad:
        raise ValueError(
            f"{res.camp} is aligned to {len(res.wt)} positions; "
            f"sequence {bad[0][0] + 1} is {bad[0][1]}. aligned input only — "
            f"padding or trimming would move every position after the change"
        )
    X, _, unk = encode.encode(qs, imgt=res.imgt, wt=res.wt)
    p = model.score(res.m, X)
    out = pd.DataFrame({
        "seq": qs,
        PRED: p,
        CLS: np.where(p >= 0.5, "enriched", "depleted"),
        "n_sub": n_mut(pd.Series(qs), res.wt).values,
    })
    return out.sort_values(PRED, ascending=False).reset_index(drop=True), unk


def heat(res):
    h = res.cf.pivot_table(index="aa", columns="imgt", values="beta", aggfunc="first")
    order = [p for p in res.imgt if p in h.columns]
    return h.reindex(columns=order)


@dataclass
class UpRes:
    rnds: list
    mode: str
    n: int
    n_tr: int
    n_te: int
    n_feat: int
    auc: float
    pr: float
    ranked: pd.DataFrame
    cf: pd.DataFrame
    conv: bool
    head: pd.DataFrame
    spread: float = 0.0
    weak: bool = False
    labels: str = "enrichment"
    imgt: list = field(default_factory=list)
    wt: str = ""
    m: object = None


# fits an uploaded table. rounds come in earliest first, from the Spec's insertion order.
# with three rounds the later pair gives an honest cross-round test; with two there is only
# a within-pair split, which is the weaker check the honest-notes forbid leaning on.
# camp reuses a vendored campaign's ANARCI numbering; parent is for any other dataset, which
# brings its own reference sequence and gets plain 1..L positions. requiring an upload to match
# a vendored alignment would make "it runs on other data" false.
def run_upload(raw, camp=None, parent=None, lam=None):
    from . import ingest

    if parent:
        imgt, wt = encode.ref_cols(parent)
    elif camp:
        imgt, wt = encode.wt_cols(camp)
    else:
        raise ValueError("need either a campaign to borrow an alignment from, or a parent sequence")

    if ingest.scored(raw):
        a, src = ingest.from_scored(raw)
        b, mode, rnds, labels = a, "pre-scored", [], src
    else:
        rnds = ingest.rnds_of(raw)
        if len(rnds) < 2:
            raise ValueError(f"need at least two rounds, the upload has {rnds}")
        early, late = rnds[0], rnds[1]
        a = ingest.to_internal(raw, (late, early))
        if len(rnds) >= 3:
            b, mode = ingest.to_internal(raw, (rnds[2], early)), "cross-round"
        else:
            b, mode = a, "within-pair"
        labels = "enrichment"
    tr = a[(a[s.SPLIT] == "train") & a[s.ENR_BIN].notna()]
    te = b[(b[s.SPLIT] == "test") & b[s.ENR_BIN].notna()]
    if len(tr) < 10 or len(te) < 10:
        raise ValueError(f"only {len(tr)} train and {len(te)} test rows survive labelling")

    Xtr, nm, _ = encode.encode(tr[s.AA_ALN], imgt=imgt, wt=wt)
    Xte, _, _ = encode.encode(te[s.AA_ALN], imgt=imgt, wt=wt)
    m = model.fit(Xtr, tr[s.ENR_BIN], lam=lam)

    Xall, _, _ = encode.encode(a[s.AA_ALN], imgt=imgt, wt=wt)
    rk = a.copy()
    rk[PRED] = model.score(m, Xall)
    rk[CLS] = np.where(rk[PRED] >= 0.5, "enriched", "depleted")
    cols = [s.VAR, PRED, CLS, s.ENR] + [s.cnt(r) for r in rnds]

    ranked = rk.sort_values(PRED, ascending=False)[cols].reset_index(drop=True)
    cf = model.coefs(m, nm)
    auc = model.auc(m, Xte, te[s.ENR_BIN])
    # spread alone is not enough: a model that memorises its training rows produces confident,
    # well-spread scores and still generalises at chance, which is exactly what an all-singles
    # assay does. so skill on held-out rows counts too.
    spread = float(ranked[PRED].std())
    weak = bool(spread < 0.02 or len(cf) < 30 or (np.isfinite(auc) and auc < 0.6))
    return UpRes(
        rnds=rnds,
        mode=mode,
        labels=labels,
        spread=spread,
        weak=weak,
        n=len(a),
        n_tr=len(tr),
        n_te=len(te),
        n_feat=Xtr.shape[1],
        auc=auc,
        pr=model.pr_auc(m, Xte, te[s.ENR_BIN]),
        ranked=ranked,
        cf=cf,
        conv=m.conv,
        head=a.head(8)[[s.VAR] + [s.cnt(r) for r in rnds] + [s.ENR, s.ENR_BIN]],
        imgt=imgt,
        wt=wt,
        m=m,
    )
