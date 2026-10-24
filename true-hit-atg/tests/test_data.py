import pandas as pd
import pytest

from truehit_atg import data, schema as s


@pytest.mark.parametrize("camp", list(s.CAMPS))
def test_load(camp):
    c = s.CAMPS[camp]
    df = data.load(camp, pair=("FACS1", "MACS"))
    assert len(df) > 1000
    assert df[s.AA_ALN].str.len().eq(c.n_pos).all()
    assert df[s.SPLIT].dropna().isin(["train", "test"]).all()
    assert df[s.ENR].notna().all()
    assert set(df[s.ENR_BIN].dropna().unique()) <= {0.0, 1.0}
    for r in c.rnds:
        assert df[s.cnt(r)].min() >= 0


@pytest.mark.parametrize("camp", list(s.CAMPS))
def test_wt(camp):
    assert len(wt := data.wt_seq(camp)) >= s.CAMPS[camp].n_pos
    assert wt.startswith("QVQL")


def test_min_ct():
    df = data.load("AT110", pair=("FACS1", "MACS"), min_ct=5)
    assert df[s.cnt("MACS")].min() >= 5


def test_bad_pair():
    with pytest.raises(ValueError):
        data.load("AT110", pair=("FACS3", "MACS"))


def test_labeled():
    df = data.load("AT110", pair=("FACS1", "MACS"))
    lab = data.labeled(df)
    assert len(lab) == 3866
    assert len(data.split(lab, "train")) == 3092
    assert len(data.split(lab, "test")) == 774
    assert lab[s.ENR_BIN].notna().all()
