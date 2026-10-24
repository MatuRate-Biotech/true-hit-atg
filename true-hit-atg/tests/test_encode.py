import numpy as np
import pytest

from truehit_atg import data, encode, schema as s


@pytest.mark.parametrize("camp", list(s.CAMPS))
def test_feature_count(camp):
    imgt, wt = encode.wt_cols(camp)
    assert len(wt) == s.CAMPS[camp].n_pos
    nm, keep = encode.feats(imgt, wt)
    assert keep.sum() == len(wt) * 20
    assert len(nm) == len(wt) * 20


@pytest.mark.parametrize("camp", list(s.CAMPS))
def test_wt_is_all_zero(camp):
    imgt, wt = encode.wt_cols(camp)
    X, _, _ = encode.encode([wt], imgt=imgt, wt=wt)
    assert X.sum() == 0


def test_no_feature_names_the_parent_residue():
    imgt, wt = encode.wt_cols("AT110")
    nm, _ = encode.feats(imgt, wt)
    assert not set(nm) & {f"{p}_{a}" for p, a in zip(imgt, wt)}


def test_one_substitution_gives_one_feature():
    imgt, wt = encode.wt_cols("AT110")
    alt = "W" if wt[40] != "W" else "A"
    q = wt[:40] + alt + wt[41:]
    X, nm, _ = encode.encode([q], imgt=imgt, wt=wt)
    assert X.sum() == 1
    assert np.asarray(nm)[X[0].astype(bool)][0] == f"{imgt[40]}_{alt}"


def test_unknown_letter_counted_as_gap():
    imgt, wt = encode.wt_cols("AT110")
    _, _, unk = encode.encode([("X" * len(wt))], imgt=imgt, wt=wt)
    assert unk == len(wt)


def test_ragged_input_rejected():
    imgt, wt = encode.wt_cols("AT110")
    with pytest.raises(ValueError, match="expected"):
        encode.encode([wt, wt[:-1]], imgt=imgt, wt=wt)


def test_wrong_width_rejected():
    imgt, wt = encode.wt_cols("AT110")
    with pytest.raises(ValueError, match="positions"):
        encode.encode([wt[:-1]], imgt=imgt, wt=wt)


def test_chk_rejects_ragged_aa_aln():
    df = data.load("AT110", pair=("FACS1", "MACS")).head(20).copy()
    df.loc[df.index[0], s.AA_ALN] = df.loc[df.index[0], s.AA_ALN][:-1]
    with pytest.raises(ValueError, match="scramble positions"):
        s.chk(df, s.CAMPS["AT110"].rnds, s.CAMPS["AT110"].n_pos)
