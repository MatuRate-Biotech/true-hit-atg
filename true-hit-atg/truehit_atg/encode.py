import numpy as np

from . import data

AAS = "ACDEFGHIKLMNPQRSTVWY"
GAP = "-"
ALPH = AAS + GAP
D = len(ALPH)
IDX = {a: i for i, a in enumerate(ALPH)}
GAP_I = IDX[GAP]


# the WT alignment carries more columns than the data: the IMGT positions that are gaps in the
# parent (10, 31-34, 60-62, 73 in all three campaigns) were dropped when the rounds were
# compiled. the data's positions are exactly the parent's non-gap columns.
def wt_cols(camp):
    wt = data.load_wt(camp)
    pos = list(wt.columns[13:])
    seq = "".join(wt.loc[0, pos].astype(str))
    keep = [(p, a) for p, a in zip(pos, seq) if a != GAP]
    return [p for p, _ in keep], "".join(a for _, a in keep)


def onehot(seqs):
    seqs = list(seqs)
    if not seqs:
        raise ValueError("no sequences to encode")
    n, L = len(seqs), len(seqs[0])
    X = np.zeros((n, L, D), dtype=np.uint8)
    unk = 0
    for i, q in enumerate(seqs):
        if len(q) != L:
            raise ValueError(f"sequence {i} is {len(q)} long, expected {L}")
        for j, a in enumerate(q):
            k = IDX.get(a)
            if k is None:
                k, unk = GAP_I, unk + 1
            X[i, j, k] = 1
    return X.reshape(n, L * D), unk


def wt_mask(wt):
    m = np.zeros((len(wt), D), dtype=bool)
    for j, a in enumerate(wt):
        m[j, IDX.get(a, GAP_I)] = True
    return m.reshape(-1)


# dropping the parent's own column at every position makes each feature a substitution
# indicator, which is what lets the fitted weights read as per-position preferences
def feats(imgt, wt):
    keep = ~wt_mask(wt)
    nm = [f"{p}_{a}" for p in imgt for a in ALPH]
    return [n for n, k in zip(nm, keep) if k], keep


def encode(seqs, camp=None, imgt=None, wt=None):
    if wt is None:
        imgt, wt = wt_cols(camp)
    X, unk = onehot(seqs)
    nm, keep = feats(imgt, wt)
    if X.shape[1] != keep.size:
        raise ValueError(
            f"sequences are {X.shape[1] // D} positions, parent alignment is {len(wt)}; "
            f"they must match for one-hot columns to mean the same thing"
        )
    return X[:, keep], nm, unk
