import pandas as pd
import pytest

from truehit_qc import schema as s


def tbl(**over):
    d = {
        s.VAR: ["v1", "v2"],
        s.DNA: ["ACGT", "ACGA"],
        s.AA: ["QVQL", "QVQM"],
        s.FLAG: [s.CLEAN, s.CHIMERA],
        s.RSN: [s.rsn("ok"), "x"],
        s.FMT: [s.FMT_NB, s.FMT_NB],
        s.FMT_CONF: [0.99, 0.80],
        s.GERM: ["IGHV3-66*01", "IGHV3-66*01"],
        s.SPEC: ["camel", "camel"],
        s.CHIM: [0.01, 0.92],
        s.PAR: [None, "v1|v3"],
        s.cnt(1): [10, 1],
    }
    d.update(over)
    return pd.DataFrame(d)


def test_ok():
    s.chk(tbl(), rnds=(1,))


def test_missing_round():
    with pytest.raises(ValueError, match="missing"):
        s.chk(tbl(), rnds=(1, 2))


def test_bad_flag():
    with pytest.raises(ValueError, match="qc_flag"):
        s.chk(tbl(**{s.FLAG: [s.CLEAN, "weird"]}), rnds=(1,))


def test_dupe_id():
    with pytest.raises(ValueError, match="duplicate"):
        s.chk(tbl(**{s.VAR: ["v1", "v1"]}), rnds=(1,))


def test_flag_needs_reason():
    with pytest.raises(ValueError, match="no reason"):
        s.chk(tbl(**{s.RSN: ["ok", None]}), rnds=(1,))


def test_reason_strings():
    r = s.rsn("chim_parents", bp=64, pa="v12", ca=4100, pb="v88", cb=3200, cnt=3)
    assert r == "first 64 aa match v12 (4100 reads), remainder matches v88 (3200 reads); 3 reads of its own"
    assert s.rsn("fw_no_germ", germ="IGHV3-23*01", idt=0.61, thr=0.7).endswith("below the 70% floor")
