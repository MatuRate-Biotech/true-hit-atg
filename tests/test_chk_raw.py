import pandas as pd
import pytest

from maturate import encode, schema as s

RNDS = ("MACS", "FACS1")


def raw(**over):
    d = {
        s.VAR: ["v1", "v2", "v3"],
        s.AA_ALN: ["QVQL", "QVWL", "QV-L"],
        s.cnt("MACS"): [10, 0, 7],
        s.cnt("FACS1"): [3, 42, 0],
    }
    d.update(over)
    return pd.DataFrame(d)


def test_accepts_a_good_upload():
    out = s.chk_raw(raw(), RNDS, alph=encode.ALPH)
    assert len(out) == 3


def test_infers_width_when_not_given():
    s.chk_raw(raw(), RNDS)


def test_width_mismatch_against_campaign():
    with pytest.raises(ValueError, match="must be 5 long"):
        s.chk_raw(raw(), RNDS, n_pos=5)


def test_ragged_names_offenders():
    with pytest.raises(ValueError) as e:
        s.chk_raw(raw(**{s.AA_ALN: ["QVQL", "QVW", "QV-L"]}), RNDS)
    assert "v2" in str(e.value) and "scramble positions" in str(e.value)


def test_missing_core_columns():
    with pytest.raises(ValueError, match="upload missing"):
        s.chk_raw(raw().drop(columns=[s.AA_ALN]), RNDS)


def test_needs_two_rounds():
    with pytest.raises(ValueError, match="at least two rounds"):
        s.chk_raw(raw().drop(columns=[s.cnt("FACS1")]), RNDS)


def test_duplicate_ids_named():
    with pytest.raises(ValueError, match="duplicate var_id"):
        s.chk_raw(raw(**{s.VAR: ["v1", "v1", "v3"]}), RNDS)


def test_unknown_alphabet_named():
    with pytest.raises(ValueError) as e:
        s.chk_raw(raw(**{s.AA_ALN: ["QVQL", "QVZL", "QV-L"]}), RNDS, alph=encode.ALPH)
    assert "'Z'" in str(e.value) and "v2" in str(e.value)


def test_gaps_are_allowed():
    s.chk_raw(raw(**{s.AA_ALN: ["----", "QVWL", "QV-L"]}), RNDS, alph=encode.ALPH)


def test_negative_counts_named():
    with pytest.raises(ValueError) as e:
        s.chk_raw(raw(**{s.cnt("MACS"): [10, -1, 7]}), RNDS)
    assert "negative" in str(e.value) and "v2" in str(e.value)


def test_non_integer_counts_rejected():
    with pytest.raises(ValueError, match="whole numbers"):
        s.chk_raw(raw(**{s.cnt("MACS"): [10.0, 1.5, 7.0]}), RNDS)


def test_vendored_table_satisfies_the_raw_contract():
    from maturate import data

    df = data.load("AT110", pair=("FACS1", "MACS"))
    s.chk_raw(df, s.CAMPS["AT110"].rnds, s.CAMPS["AT110"].n_pos, alph=encode.ALPH)
