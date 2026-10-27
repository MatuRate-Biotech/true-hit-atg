import numpy as np

from . import data
from .config import CFG

AAS = "ACDEFGHIKLMNPQRSTVWY"
GAP = "-"
ALPH = AAS + GAP
AA_SET = frozenset(AAS)
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


_BL = None
# BLOSUM62 raw scores run about -4..+11; the one-hot columns are 0/1, and L1 penalizes a
# large-valued feature less for the same effect. a fixed divisor keeps them comparable and,
# unlike standardizing on batch statistics, cannot leak between train and test.
BL_SCALE = 4.0


def blosum():
    global _BL
    if _BL is None:
        from Bio.Align import substitution_matrices

        _BL = substitution_matrices.load("BLOSUM62")
    return _BL


# one column per position carrying the parent->variant exchangeability, zero where unchanged.
# a substitution the model has never seen still lands in a column that may carry weight, which
# the one-hot alone cannot do.
def prop_cols(seqs, wt):
    m = blosum()
    P = np.zeros((len(seqs), len(wt)), dtype=np.float32)
    for i, q in enumerate(seqs):
        for j, (a, b) in enumerate(zip(wt, q)):
            if a != b:
                try:
                    P[i, j] = m[a, b]
                except (KeyError, IndexError):
                    P[i, j] = 0.0
    return P / BL_SCALE


def encode(seqs, camp=None, imgt=None, wt=None, props=None):
    if wt is None:
        imgt, wt = wt_cols(camp)
    seqs = list(seqs)
    X, unk = onehot(seqs)
    nm, keep = feats(imgt, wt)
    if X.shape[1] != keep.size:
        raise ValueError(
            f"sequences are {X.shape[1] // D} positions, parent alignment is {len(wt)}; "
            f"they must match for one-hot columns to mean the same thing"
        )
    X = X[:, keep]
    if props if props is not None else CFG.props:
        X = np.hstack([X.astype(np.float32), prop_cols(seqs, wt)])
        nm = list(nm) + [f"{p}_blosum" for p in imgt]
    return X, nm, unk


# the parent in model space has no gaps in any of the three campaigns, so for proposals built
# by substituting into it these two are the identity and a plain splice. they exist for the
# general case: an uploaded campaign whose parent does carry a gap in a retained column.
def gap_map(ref):
    ung, idx = [], []
    for i, a in enumerate(ref):
        if a != GAP:
            ung.append(a)
            idx.append(i)
    return "".join(ung), idx


def place(ref, pos, aa):
    _, idx = gap_map(ref)
    if not 0 <= pos < len(idx):
        raise ValueError(f"ungapped position {pos} out of range 0..{len(idx) - 1}")
    # membership against the string would accept "" and any substring such as "VW",
    # either of which changes the width and shifts every position after it
    if aa not in AA_SET:
        raise ValueError(f"{aa!r} is not one of the 20 amino acids")
    i = idx[pos]
    return ref[:i] + aa + ref[i + 1 :]


# IMGT numbering is antibody-specific and meaningless for a generic assay, so an upload that
# brings its own parent gets plain 1..L positions. the heatmap then reads as positions.
def ref_cols(parent):
    return [str(i + 1) for i in range(len(parent))], parent
