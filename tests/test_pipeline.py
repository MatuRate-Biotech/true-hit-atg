import warnings

import numpy as np

import pytest

from maturate import pipeline as pl, schema as s


# the paper's experiment: train on the early pair, test on a later one. pinned wherever a
# published or previously measured figure is asserted, so those checks keep measuring the
# cross-round configuration now that CFG defaults to same-pair.
CROSS = {"train": ("FACS1", "MACS"), "test": ("FACS2", "MACS")}


# deliberately unpinned: this fixture tracks whatever CFG defaults to, so the app's default
# configuration stays covered.
@pytest.fixture(scope="module")
def res():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pl.run("AT110")


def test_fits_and_converges(res):
    assert res.conv
    assert res.n_feat == 2520
    assert res.n_tr > 3000
    # the held-out size depends on which pair CFG points at, so assert the split that
    # make_split() is supposed to produce rather than a count tied to one configuration
    assert res.n_te == pytest.approx(0.2 * (res.n_tr + res.n_te), rel=0.02)


def test_beats_sequence_baseline(res):
    assert res.auc_te > 0.9
    assert res.auc_te > res.auc_nmut + 0.1


def test_strict_auc_close_to_reported(res):
    assert res.n_strict > 100
    assert abs(res.auc_te - res.auc_strict) < 0.05


def test_coefs_are_sparse_and_named(res):
    assert 50 < len(res.cf) < 400
    assert res.cf["beta"].abs().is_monotonic_decreasing
    assert set(res.cf.columns) >= {"feat", "beta", "imgt", "aa"}


def test_ranked_never_says_affinity(res):
    for c in res.ranked.columns:
        assert "affinity" not in c.lower() and "binding" not in c.lower()


def test_parent_encodes_and_scores(res):
    out, unk = pl.score_seqs(res, [res.wt])
    assert unk == 0
    assert out.loc[0, "n_sub"] == 0
    assert 0.0 <= out.loc[0, pl.PRED] <= 1.0


def test_score_seqs_rejects_wrong_length(res):
    with pytest.raises(ValueError, match="aligned input only"):
        pl.score_seqs(res, [res.wt[:-3]])


def test_score_seqs_rejects_empty(res):
    with pytest.raises(ValueError, match="no sequences"):
        pl.score_seqs(res, ["", "   "])


def test_bad_pair_rejected():
    with pytest.raises(ValueError, match="no round pair"):
        pl.run("AT110", train=("FACS3", "MACS"))


def test_heat_covers_only_hit_positions(res):
    h = pl.heat(res)
    assert h.shape[1] == res.cf["imgt"].nunique()
    assert list(h.columns) == [p for p in res.imgt if p in set(res.cf["imgt"])]


# paul et al. table 1, L1 logistic, measured in the CROSS configuration
PUB = {"AT110": (0.964, 0.949), "RX002": (0.935, 0.947)}


@pytest.mark.parametrize("camp", list(PUB))
def test_matches_published_numbers(camp):
    roc, pr = PUB[camp]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pl.run(camp, **CROSS)
    assert abs(r.auc_te - roc) < 0.03, f"{camp} ROC {r.auc_te:.3f} vs published {roc}"
    assert abs(r.pr_te - pr) < 0.03, f"{camp} PR {r.pr_te:.3f} vs published {pr}"


def test_cond_auc_constant_within_stratum_is_half():
    strat = [1, 1, 1, 2, 2, 2]
    y = [0, 1, 1, 0, 0, 1]
    p = [7.0, 7.0, 7.0, 3.0, 3.0, 3.0]
    assert pl.model.cond_auc(y, p, strat) == pytest.approx(0.5)


def test_cond_auc_mutation_count_itself_is_half():
    strat = [1, 1, 2, 2]
    y = [0, 1, 0, 1]
    assert pl.model.cond_auc(y, strat, strat) == pytest.approx(0.5)


def test_cond_auc_perfect_ordering_is_one():
    strat = [1, 1, 2, 2]
    y = [0, 1, 0, 1]
    p = [0.1, 0.9, 0.2, 0.8]
    assert pl.model.cond_auc(y, p, strat) == pytest.approx(1.0)


def test_cond_auc_skips_single_class_strata():
    strat = [1, 1, 2, 2]
    y = [0, 1, 1, 1]
    p = [0.1, 0.9, 0.5, 0.5]
    assert pl.model.cond_auc(y, p, strat) == pytest.approx(1.0)
    assert np.isnan(pl.model.cond_auc([1, 1], [0.1, 0.2], [1, 1]))


# measured in the CROSS configuration, so these calls pin it too
WITHIN = {"AT110": 0.945, "RX002": 0.785}


@pytest.mark.parametrize("camp", list(WITHIN))
def test_within_load_auc(camp):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pl.run(camp, **CROSS)
    assert abs(r.auc_within - WITHIN[camp]) < 0.03
    # the claim the pitch rests on: with mutation load held fixed the heuristic has no
    # resolution at all, and the model still orders variants
    assert r.auc_within > 0.6


def test_prec_at_k_perfect_ranking():
    y = [1, 1, 1, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1]
    k, pk = pl.model.prec_at_k(y, p)
    assert list(k) == [1, 2, 3, 4, 5, 6, 7]
    # every K up to the number of positives is all-positive
    assert (pk[:3] == 1.0).all()
    # and after that it can only decay, since the rest are negatives
    assert pk[3] == pytest.approx(3 / 4)
    assert pk[-1] == pytest.approx(3 / 7)


def test_prec_at_k_sorts_by_score_not_input_order():
    y = [0, 1, 0, 1]
    p = [0.1, 0.9, 0.2, 0.8]
    _, pk = pl.model.prec_at_k(y, p)
    assert (pk[:2] == 1.0).all()


def test_prec_at_k_worst_ranking():
    y = [1, 1, 0, 0]
    p = [0.1, 0.2, 0.8, 0.9]
    _, pk = pl.model.prec_at_k(y, p)
    assert pk[0] == 0.0
    assert pk[-1] == pytest.approx(0.5)


def test_res_carries_prec_at_k_and_base(res):
    k, pk = res.pk
    assert len(k) == len(pk) == res.n_te
    assert k[0] == 1 and k[-1] == res.n_te
    # the last point is every variant, so precision there is the positive rate
    assert pk[-1] == pytest.approx(res.base)
    assert 0.0 < res.base < 1.0
