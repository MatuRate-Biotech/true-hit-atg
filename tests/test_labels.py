import pytest

from maturate import schema as s

EXPECT = {
    "AT110": ["Input", "Output 1", "Output 2"],
    "RX002": ["Input", "Output 1", "Output 2"],
    "NbB7": ["Input", "Output 1", "Output 2", "Output 3", "Output 4"],
}


@pytest.mark.parametrize("camp", list(EXPECT))
def test_round_labels_are_positional(camp):
    assert [s.rnd_lab(camp, r) for r in s.CAMPS[camp].rnds] == EXPECT[camp]


def test_index_zero_is_input_whatever_the_assay_is():
    # AT110 starts at MACS, NbB7 starts at naive; both are Input
    assert s.rnd_lab("AT110", "MACS") == "Input"
    assert s.rnd_lab("NbB7", "naive") == "Input"
    # and the same assay is not the same label across campaigns
    assert s.rnd_lab("NbB7", "MACS") == "Output 1"


def test_pair_label_reads_late_over_early():
    assert s.pair_lab("AT110", ("FACS1", "MACS")) == "Output 1 over Input"
    assert s.pair_lab("AT110", ("FACS2", "MACS")) == "Output 2 over Input"
    assert s.pair_lab("AT110", ("FACS2", "FACS1")) == "Output 2 over Output 1"


def test_every_pair_of_every_campaign_labels_without_collision():
    for camp in s.CAMPS:
        labs = [s.pair_lab(camp, p) for p in s.pairs(camp)]
        assert len(set(labs)) == len(labs)
        assert all("FACS" not in x and "MACS" not in x and "naive" not in x for x in labs)


# an upload has no campaign: the order comes from its own cnt_* columns
def test_an_explicit_round_order_works_the_same():
    order = ["pre", "post"]
    assert s.rnd_lab(order, "pre") == "Input"
    assert s.rnd_lab(order, "post") == "Output 1"
    assert s.pair_lab(order, ("post", "pre")) == "Output 1 over Input"


def test_upload_order_is_positional_not_alphabetical():
    order = ["zz_input", "aa_first", "mm_second"]
    assert [s.rnd_lab(order, r) for r in order] == ["Input", "Output 1", "Output 2"]


def test_rnd_order_accepts_a_name_or_a_sequence():
    assert s.rnd_order("AT110") == ["MACS", "FACS1", "FACS2"]
    assert s.rnd_order(["a", "b"]) == ["a", "b"]


def test_unknown_round_passes_through_unchanged():
    assert s.rnd_lab("AT110", "FACS9") == "FACS9"
