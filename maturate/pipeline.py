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


# substitution notation on the coefficient frame: parent residue, position, substituted
# residue (N66D). the parent residue at a position is wt[imgt.index(p)]; mapped rather than
# looked up per row.
def with_sub(cf, imgt, wt):
    parent = dict(zip(imgt, wt))
    out = cf.copy()
    out["sub"] = out["imgt"].map(parent) + out["imgt"].astype(str) + out["aa"]
    return out


def run(camp=None, train=None, test=None, lam=None, min_ct=None):
    camp = camp or CFG.camp
    tp = tuple(train or CFG.train)
    sp = tuple(test or CFG.test)
    # each pair gets its own view of the vendored union table, because data.load() filters
    # rows to the variants that pair actually observed. loading once and reusing it would
    # judge the test pair on the train pair's row set.
    raw_a = data.load(camp, pair=tp, min_ct=min_ct)
    a = enrich.pair(raw_a, tp[0], tp[1], camp)
    if sp == tp:
        b = a
    else:
        b = enrich.pair(data.load(camp, pair=sp, min_ct=min_ct), sp[0], sp[1], camp)

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
    early = pd.Series(
        a.set_index(s.VAR)[s.ENR].reindex(te[s.VAR]).values, index=te.index
    )
    auc_ct = enrich.baseline_auc(early, te[s.ENR_BIN])

    Xall, _, _ = encode.encode(a[s.AA_ALN], imgt=imgt, wt=wt)
    rk = a.copy()
    rk[PRED] = model.score(m, Xall)
    rk[CLS] = np.where(rk[PRED] >= 0.5, "enriched", "depleted")
    # aa_aln rides along so a downloaded ranking carries the sequence; the tab does not show it
    cols = [s.VAR, PRED, CLS, s.SPLIT, s.CDR3, s.ENR, s.cnt(tp[1]), s.cnt(tp[0]), s.AA_ALN]
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
        mix=enrich.pair_mix(raw_a, tp[0], tp[1], camp),
        cf=with_sub(model.coefs(m, nm), imgt, wt),
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


# a Res with the upload-only extras bolted on. a subclass rather than optional fields on
# Res, so Res stays the single thing the tabs read and no vendored run carries six fields
# that are always None.
@dataclass
class UpRes(Res):
    mode: str = "within-pair"
    rnds: list = field(default_factory=list)
    labels: str = "enrichment"
    spread: float = 0.0
    weak: bool = False
    head: pd.DataFrame = None

    @property
    def n(self):
        return len(self.tbl)


# fits an uploaded table. rounds come in earliest first, from the Spec's insertion order.
# with three rounds the later pair gives an honest cross-round test; with two there is only
# a within-pair split, which is the weaker check the honest-notes forbid leaning on.
# camp reuses a vendored campaign's ANARCI numbering; parent is for any other dataset, which
# brings its own reference sequence and gets plain 1..L positions. requiring an upload to match
# a vendored alignment would make "it runs on other data" false.
def run_upload(raw, camp=None, parent=None, lam=None, name=None):
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
    cols = [s.VAR, PRED, CLS, s.SPLIT, s.ENR] + [s.cnt(r) for r in rnds] + [s.AA_ALN]

    ranked = rk.sort_values(PRED, ascending=False)[cols].reset_index(drop=True)
    cf = with_sub(model.coefs(m, nm), imgt, wt)
    auc = model.auc(m, Xte, te[s.ENR_BIN])
    # spread alone is not enough: a model that memorises its training rows produces confident,
    # well-spread scores and still generalises at chance, which is exactly what an all-singles
    # assay does. so skill on held-out rows counts too.
    spread = float(ranked[PRED].std())
    weak = bool(spread < 0.02 or len(cf) < 30 or (np.isfinite(auc) and auc < 0.6))

    nmu = n_mut(te[s.AA_ALN], wt)
    # with one round pair there is no second split to overlap with: train and test are
    # disjoint by construction, so a strict figure would just restate auc_te.
    if mode == "cross-round":
        keep = ~te[s.VAR].isin(set(tr[s.VAR])).values
        auc_st = model.auc(m, Xte[keep], te[s.ENR_BIN][keep]) if keep.sum() > 10 else float("nan")
        n_st = int(keep.sum())
    else:
        auc_st, n_st = float("nan"), 0

    early_enr = pd.Series(
        a.set_index(s.VAR)[s.ENR].reindex(te[s.VAR]).values, index=te.index
    )
    pair = (rnds[1], rnds[0]) if len(rnds) >= 2 else ("", "")
    test_pair = (rnds[2], rnds[0]) if mode == "cross-round" else pair

    return UpRes(
        camp=name or "upload",
        train=pair,
        test=test_pair,
        tbl=a,
        n_tr=len(tr),
        n_te=len(te),
        n_abst=int(a[s.ENR_BIN].isna().sum()),
        n_feat=Xtr.shape[1],
        auc_tr=model.auc(m, Xtr, tr[s.ENR_BIN]),
        auc_te=auc,
        pr_te=model.pr_auc(m, Xte, te[s.ENR_BIN]),
        auc_strict=auc_st,
        n_strict=n_st,
        auc_nmut=enrich.baseline_auc(nmu, te[s.ENR_BIN]),
        auc_within=model.cond_auc(te[s.ENR_BIN], model.score(m, Xte), nmu),
        auc_cnt=enrich.baseline_auc(early_enr, te[s.ENR_BIN]),
        roc=model.roc(m, Xte, te[s.ENR_BIN]),
        pk=model.prec_at_k(te[s.ENR_BIN], model.score(m, Xte)),
        base=float(te[s.ENR_BIN].mean()),
        mix=enrich.mix_params(a[s.ENR], a[s.ENR].notna()),
        cf=cf,
        ranked=ranked,
        conv=m.conv,
        imgt=imgt,
        wt=wt,
        m=m,
        mode=mode,
        rnds=rnds,
        labels=labels,
        spread=spread,
        weak=weak,
        head=a.head(8)[[s.VAR] + [s.cnt(r) for r in rnds] + [s.ENR, s.ENR_BIN]],
    )


def heat(res):
    h = res.cf.pivot_table(index="aa", columns="imgt", values="beta", aggfunc="first")
    order = [p for p in res.imgt if p in h.columns]
    return h.reindex(columns=order)
