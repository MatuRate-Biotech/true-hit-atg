import numpy as np
import pandas as pd
import pytest

from maturate import data, enrich, schema as s

PAIR = ("FACS1", "MACS")


@pytest.mark.parametrize("camp", ["AT110", "RX002"])
def test_norm_matches_shipped(camp):
    raw = data.load_raw(camp)
    df = data.load(camp, pair=PAIR)
    for r in s.CAMPS[camp].rnds:
        mine = enrich.norm_ct(df, r, camp)
        assert np.abs(mine - raw[s.up_norm(r)]).max() < 1e-6


@pytest.mark.parametrize("camp", ["AT110", "NbB7", "RX002"])
def test_hasread_is_count_threshold(camp):
    df = data.load(camp, pair=PAIR)
    for r in s.CAMPS[camp].rnds:
        assert (enrich.has_read(df, r) == df[s.has(r)].astype(bool)).all()


@pytest.mark.parametrize("camp", ["AT110", "RX002"])
def test_log_ratio_matches_shipped(camp):
    df = data.load(camp, pair=PAIR)
    p = enrich.pair(df, *PAIR, camp)
    assert np.abs(p[s.ENR] - df[s.ENR]).max() < 1e-9


@pytest.mark.parametrize("camp", ["AT110", "RX002"])
def test_labels_reproduce_paper(camp):
    df = data.load(camp, pair=PAIR)
    p = enrich.pair(df, *PAIR, camp)
    ours, ship = p[s.ENR_BIN], df[s.ENR_BIN]
    both = ours.notna() & ship.notna()
    assert (ours[both] == ship[both]).mean() > 0.99
    assert abs(ours.notna().sum() - ship.notna().sum()) / ship.notna().sum() < 0.03


def test_abstains_on_ambiguous():
    df = data.load("AT110", pair=PAIR)
    p = enrich.pair(df, *PAIR, "AT110")
    assert p[s.ENR_BIN].isna().sum() > 0
    assert set(p[s.ENR_BIN].dropna().unique()) == {0.0, 1.0}


def test_split_is_80_20():
    df = data.load("AT110", pair=PAIR)
    p = enrich.pair(df, *PAIR, "AT110")
    v = p[s.SPLIT].value_counts()
    assert abs(v["test"] / (v["train"] + v["test"]) - 0.2) < 0.01


def test_lib_falls_back_to_column_sum():
    df = data.load("AT110", pair=PAIR)
    assert enrich.lib(df, "MACS", "AT110") == s.LIBS["AT110"]["MACS"]
    assert enrich.lib(df, "MACS", "UnknownCamp") == int(df[s.cnt("MACS")].sum())
