import pytest

from truehit_atg import encode, schema as s

GAPPY = "QV-LQ--SGG-"


@pytest.mark.parametrize("camp", list(s.CAMPS))
def test_parent_has_no_gaps_in_model_space(camp):
    _, wt = encode.wt_cols(camp)
    ung, idx = encode.gap_map(wt)
    assert ung == wt
    assert idx == list(range(len(wt)))


def test_gap_map_on_gapped_reference():
    ung, idx = encode.gap_map(GAPPY)
    assert ung == "QVLQSGG"
    assert idx == [0, 1, 3, 4, 7, 8, 9]
    assert all(GAPPY[i] == a for i, a in zip(idx, ung))


@pytest.mark.parametrize("ref", [GAPPY, "QVQLQ", "-" * 3 + "AC"])
def test_place_round_trips_every_position(ref):
    ung, _ = encode.gap_map(ref)
    for i, a in enumerate(ung):
        out = encode.place(ref, i, a)
        assert out == ref
        assert len(out) == len(ref)


def test_place_width_is_invariant():
    ung, _ = encode.gap_map(GAPPY)
    for i in range(len(ung)):
        assert len(encode.place(GAPPY, i, "W")) == len(GAPPY)


def test_place_substitutes_the_right_column():
    assert encode.place(GAPPY, 2, "W") == "QV-WQ--SGG-"
    assert encode.place(GAPPY, 4, "W") == "QV-LQ--WGG-"


def test_place_never_touches_a_gap():
    ung, _ = encode.gap_map(GAPPY)
    for i in range(len(ung)):
        out = encode.place(GAPPY, i, "W")
        assert [j for j, c in enumerate(out) if c == "-"] == [
            j for j, c in enumerate(GAPPY) if c == "-"
        ]


def test_place_substitution_encodes_as_one_feature():
    imgt, wt = encode.wt_cols("AT110")
    q = encode.place(wt, 40, "W" if wt[40] != "W" else "A")
    X, _, _ = encode.encode([q], imgt=imgt, wt=wt)
    assert X.sum() == 1


def test_place_rejects_bad_position():
    ung, _ = encode.gap_map(GAPPY)
    with pytest.raises(ValueError, match="out of range"):
        encode.place(GAPPY, len(ung), "W")
    with pytest.raises(ValueError, match="out of range"):
        encode.place(GAPPY, -1, "W")


def test_place_rejects_non_amino_acid():
    for bad in ["-", "X", "WW", ""]:
        with pytest.raises(ValueError, match="amino acid"):
            encode.place(GAPPY, 0, bad)


def test_place_rejects_multi_letter_substring():
    # "VW" is a substring of the alphabet; accepting it would change the width
    for bad in ["VW", "AC", "TVW"]:
        with pytest.raises(ValueError, match="amino acid"):
            encode.place(GAPPY, 0, bad)
