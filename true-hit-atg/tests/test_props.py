import numpy as np
import pytest

from truehit_atg import data, encode, enrich, model, schema as s
from truehit_atg.config import CFG


@pytest.fixture(scope="module")
def at110():
    raw = data.load("AT110", pair=("FACS1", "MACS"))
    a = enrich.pair(raw, "FACS1", "MACS", "AT110")
    b = enrich.pair(raw, "FACS2", "MACS", "AT110")
    imgt, wt = encode.wt_cols("AT110")
    tr = a[(a[s.SPLIT] == "train") & a[s.ENR_BIN].notna()]
    te = b[(b[s.SPLIT] == "test") & b[s.ENR_BIN].notna()]
    return imgt, wt, tr, te


def test_props_default_off_so_reproduction_is_untouched():
    assert CFG.props is False


def test_props_add_one_column_per_position(at110):
    imgt, wt, tr, _ = at110
    a, na, _ = encode.encode(tr[s.AA_ALN].head(20), imgt=imgt, wt=wt, props=False)
    b, nb, _ = encode.encode(tr[s.AA_ALN].head(20), imgt=imgt, wt=wt, props=True)
    assert b.shape[1] == a.shape[1] + len(wt)
    assert nb[-1].endswith("_blosum")
    assert len(nb) == b.shape[1]


def test_blosum_column_is_zero_where_unchanged(at110):
    imgt, wt, _, _ = at110
    p = encode.prop_cols([wt, encode.place(wt, 40, "W")], wt)
    assert p[0].sum() == 0
    assert (p[1] != 0).sum() == 1
    assert p[1][40] == pytest.approx(encode.blosum()[wt[40], "W"] / encode.BL_SCALE)


def novel_subs(imgt, wt, tr, n=400):
    X, _, _ = encode.encode(tr[s.AA_ALN], imgt=imgt, wt=wt, props=False)
    seen = set(np.asarray(encode.feats(imgt, wt)[0])[X.sum(0) > 0])
    out = [(j, aa) for j in range(len(wt)) for aa in encode.AAS
           if aa != wt[j] and f"{imgt[j]}_{aa}" not in seen]
    return out[:n]


# the finding that broke Feature B, and the fix for it, as one test
def test_novel_substitutions_tie_without_props_and_separate_with(at110):
    imgt, wt, tr, _ = at110
    qs = [encode.place(wt, j, aa) for j, aa in novel_subs(imgt, wt, tr)]
    counts = {}
    for pr in (False, True):
        X, _, _ = encode.encode(tr[s.AA_ALN], imgt=imgt, wt=wt, props=pr)
        m = model.fit(X, tr[s.ENR_BIN])
        Xq, _, _ = encode.encode(qs, imgt=imgt, wt=wt, props=pr)
        counts[pr] = len(np.unique(model.score(m, Xq).round(6)))
    # one-hot alone cannot tell any unseen substitution from any other
    assert counts[False] == 1
    assert counts[True] > 20


def test_props_do_not_disturb_the_cross_round_result(at110):
    imgt, wt, tr, te = at110
    got = {}
    for pr in (False, True):
        Xtr, _, _ = encode.encode(tr[s.AA_ALN], imgt=imgt, wt=wt, props=pr)
        Xte, _, _ = encode.encode(te[s.AA_ALN], imgt=imgt, wt=wt, props=pr)
        m = model.fit(Xtr, tr[s.ENR_BIN])
        got[pr] = model.auc(m, Xte, te[s.ENR_BIN])
    assert abs(got[True] - got[False]) < 0.02
    assert got[True] > 0.9
