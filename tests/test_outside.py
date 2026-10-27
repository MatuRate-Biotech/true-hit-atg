import numpy as np
import pandas as pd
import pytest

from maturate import encode, ingest, model, pipeline as pl, schema as s

# a tiny synthetic "outside" dataset: its own parent, no counts, a supplied score.
PARENT = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
AAS = list("ACDEFGHIKLMNPQRSTVWY")


POOL = [3, 7, 10, 14, 18, 20, 25, 29]


def make(tmp_path, n=400, binned=True, seed=0, singles=False):
    """Variants carry 2-4 substitutions drawn from a small pool, so substitutions recur across
    variants and a held-out one has usually been trained on. With singles=True each variant
    carries exactly one, which is the PAI-1 shape: a substitution in test is by construction
    absent from train, every row encodes to the intercept, and AUC is 0.500 by arithmetic."""
    rng = np.random.default_rng(seed)
    rows = []
    if singles:
        # a systematic scan: every (position, residue) appears at most once in the whole set,
        # so a substitution that lands in test cannot also be in train. this is deterministic,
        # not a sampling accident, and it is what makes the assay unlearnable here.
        allp = [(p, a) for p in range(len(PARENT)) for a in AAS if a != PARENT[p]]
        idx = rng.permutation(len(allp))[:n]
        picks = [[allp[i]] for i in idx]
    else:
        picks = []
        for _ in range(n):
            k = int(rng.integers(2, 5))
            ps = rng.choice(POOL, size=k, replace=False)
            picks.append([(int(p), rng.choice([x for x in AAS if x != PARENT[p]])) for p in ps])

    for i, subs in enumerate(picks):
        seq, sc = list(PARENT), float(rng.normal(0, 1))
        for p, a in subs:
            seq[p] = a
            sc += 2 if p == 10 else (-2 if p == 20 else 0)
        rows.append({"vid": f"v{i}", "prot": "".join(seq), "score": sc, "hit": int(sc > 0)})
    d = pd.DataFrame(rows)
    if not binned:
        d = d.drop(columns=["hit"])
    p_ = tmp_path / "outside.csv"
    d.to_csv(p_, index=False)
    return p_


def spec(binned=True):
    return ingest.Spec(var="vid", seq="prot", rnds={}, enr_col="score",
                       enr_bin_col="hit" if binned else "")


def test_ref_cols_are_plain_indices():
    imgt, wt = encode.ref_cols(PARENT)
    assert imgt == [str(i + 1) for i in range(len(PARENT))]
    assert wt == PARENT
    assert not any(c.isalpha() for c in "".join(imgt))


def test_a_scored_upload_needs_no_counts(tmp_path):
    raw = ingest.apply_spec(make(tmp_path), spec(), n_pos=len(PARENT))
    assert ingest.scored(raw)
    assert ingest.rnds_of(raw) == []
    assert s.ENR in raw.columns and s.ENR_BIN in raw.columns


def test_unscored_upload_still_demands_counts(tmp_path):
    p = make(tmp_path)
    bare = ingest.Spec(var="vid", seq="prot", rnds={})
    with pytest.raises(ValueError, match="at least two rounds"):
        ingest.apply_spec(p, bare, n_pos=len(PARENT))


def test_from_scored_skips_enrichment(tmp_path):
    raw = ingest.apply_spec(make(tmp_path), spec(), n_pos=len(PARENT))
    out, src = ingest.from_scored(raw)
    assert src == "supplied"
    assert out[s.SPLIT].dropna().isin(["train", "test"]).all()
    # no round columns exist at all, so no enrichment ran
    assert not [c for c in out.columns if c.startswith("cnt_") or c.startswith("norm_")]


def test_binarizes_when_only_a_score_is_given(tmp_path):
    raw = ingest.apply_spec(make(tmp_path, binned=False), spec(binned=False), n_pos=len(PARENT))
    out, src = ingest.from_scored(raw)
    assert src == "gmm"
    assert set(out[s.ENR_BIN].dropna().unique()) <= {0.0, 1.0}


def test_run_upload_with_its_own_parent(tmp_path):
    raw = ingest.apply_spec(make(tmp_path, n=600), spec(), n_pos=len(PARENT))
    u = pl.run_upload(raw, parent=PARENT, lam=0.05)
    assert u.mode == "pre-scored"
    assert u.labels == "supplied"
    assert u.rnds == []
    assert u.n_feat == len(PARENT) * 20
    assert u.auc_te > 0.6


# the PAI-1 result, pinned. a one-hot over substitutions cannot transfer to a substitution it
# has never seen, and in an all-singles assay every held-out substitution is one of those.
def test_all_singles_cannot_be_learned(tmp_path):
    raw = ingest.apply_spec(make(tmp_path, n=800, singles=True), spec(), n_pos=len(PARENT))
    u = pl.run_upload(raw, parent=PARENT, lam=0.05)
    assert u.auc_te == pytest.approx(0.5, abs=0.02)
    assert u.weak


def test_all_singles_leave_every_test_row_unseen(tmp_path):
    raw = ingest.apply_spec(make(tmp_path, n=800, singles=True), spec(), n_pos=len(PARENT))
    out, _ = ingest.from_scored(raw)
    imgt, wt = encode.ref_cols(PARENT)
    tr = out[out[s.SPLIT] == "train"]
    te = out[out[s.SPLIT] == "test"]
    Xtr, _, _ = encode.encode(tr[s.AA_ALN], imgt=imgt, wt=wt)
    Xte, _, _ = encode.encode(te[s.AA_ALN], imgt=imgt, wt=wt)
    assert Xtr.sum(1).max() == 1
    seen = Xtr.sum(0) > 0
    assert (Xte[:, seen].sum(1) == 0).mean() == 1.0


def test_run_upload_needs_a_parent_or_a_campaign(tmp_path):
    raw = ingest.apply_spec(make(tmp_path), spec(), n_pos=len(PARENT))
    with pytest.raises(ValueError, match="campaign .* or a parent"):
        pl.run_upload(raw)


def test_parent_wins_over_campaign(tmp_path):
    raw = ingest.apply_spec(make(tmp_path, n=600), spec(), n_pos=len(PARENT))
    u = pl.run_upload(raw, camp="AT110", parent=PARENT, lam=0.05)
    assert u.n_feat == len(PARENT) * 20  # not AT110's 2520


def test_coefs_survives_a_penalty_that_zeroes_everything():
    class M:
        coef_ = np.zeros((1, 6))

    out = model.coefs(M(), [f"{i}_A" for i in range(6)])
    assert out.empty
    assert list(out.columns) == ["feat", "beta", "imgt", "aa"]


def test_everything_zeroed_is_flagged_weak(tmp_path):
    raw = ingest.apply_spec(make(tmp_path, n=600), spec(), n_pos=len(PARENT))
    u = pl.run_upload(raw, parent=PARENT, lam=50.0)
    assert len(u.cf) == 0
    assert u.weak
