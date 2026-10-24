import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from truehit_atg import data, encode, enrich, ingest, model, schema as s

PAIR = ("FACS1", "MACS")
SPEC = ingest.Spec(
    var="clone", seq="protein", rnds={"MACS": "input_ct", "FACS1": "r1_ct"}, note="fixture"
)


@pytest.fixture(scope="module")
def vendored():
    return data.load("AT110", pair=PAIR)


@pytest.fixture(scope="module")
def ours(vendored):
    return enrich.pair(vendored, *PAIR, "AT110")


@pytest.fixture(scope="module")
def src(tmp_path_factory, vendored):
    p = tmp_path_factory.mktemp("up") / "campaign.csv"
    pd.DataFrame({
        "clone": vendored[s.VAR],
        "protein": vendored[s.AA_ALN],
        "input_ct": vendored[s.cnt("MACS")],
        "r1_ct": vendored[s.cnt("FACS1")],
        "junk": 1,
    }).to_csv(p, index=False)
    return p


def test_probe(src):
    pr = ingest.probe(src)
    assert pr["n_rows"] == 5257
    assert pr["cols"] == ["clone", "protein", "input_ct", "r1_ct", "junk"]
    assert pr["dtypes"]["input_ct"].startswith("int")
    assert len(pr["head"]) == 5


def test_apply_spec_renames_and_drops_junk(src, vendored):
    raw = ingest.apply_spec(src, SPEC)
    assert list(raw.columns) == [s.VAR, s.AA_ALN, s.cnt("MACS"), s.cnt("FACS1")]
    assert len(raw) == len(vendored)
    assert raw[s.AA_ALN].tolist() == vendored[s.AA_ALN].tolist()


def test_apply_spec_names_missing_source_column(src):
    bad = ingest.Spec(var="clone", seq="nope", rnds=SPEC.rnds)
    with pytest.raises(ValueError, match="source columns not in"):
        ingest.apply_spec(src, bad)


def test_apply_spec_rejects_wrong_width_for_campaign(src):
    with pytest.raises(ValueError, match="must be 5 long"):
        ingest.apply_spec(src, SPEC, n_pos=5)


def test_apply_spec_rejects_ragged(tmp_path, vendored):
    p = tmp_path / "ragged.csv"
    q = vendored[s.AA_ALN].tolist()
    q[1] = q[1][:-1]
    pd.DataFrame({"clone": vendored[s.VAR], "protein": q,
                  "input_ct": vendored[s.cnt("MACS")],
                  "r1_ct": vendored[s.cnt("FACS1")]}).to_csv(p, index=False)
    with pytest.raises(ValueError) as e:
        ingest.apply_spec(p, SPEC)
    assert "scramble positions" in str(e.value)


def test_rnds_of(src):
    assert set(ingest.rnds_of(ingest.apply_spec(src, SPEC))) == {"MACS", "FACS1"}


# the only proof the upload path and the vendored path are the same code
def test_equivalent_to_vendored_path_with_campaign(src, ours):
    out = ingest.to_internal(ingest.apply_spec(src, SPEC), PAIR, camp="AT110")
    both = out[s.ENR_BIN].notna() & ours[s.ENR_BIN].notna()
    assert (out.loc[both, s.ENR_BIN] == ours.loc[both, s.ENR_BIN]).mean() >= 0.98
    assert out[s.ENR_BIN].notna().sum() == pytest.approx(ours[s.ENR_BIN].notna().sum(), rel=0.02)
    assert np.abs(out[s.ENR] - ours[s.ENR]).max() < 1e-9


def test_equivalent_without_campaign(src, ours, vendored):
    """The column-sum fallback shifts enr by a constant only where neither round sits on
    the 1 CPM floor. A different library size moves the floor, so floored rows shift by
    different amounts and the ranking is very slightly perturbed."""
    out = ingest.to_internal(ingest.apply_spec(src, SPEC), PAIR)
    d = out[s.ENR] - ours[s.ENR]
    assert abs(d.mean()) > 1e-6, "expected the fallback to shift enr"

    unfloored = (vendored[s.cnt("MACS")] > 0) & (vendored[s.cnt("FACS1")] > 0)
    du = d[unfloored]
    assert du.max() - du.min() < 1e-9, "shift must be constant where the floor does not bite"
    assert d.nunique() > 1, "floored rows should not share that constant"

    assert spearmanr(out[s.ENR], ours[s.ENR]).statistic > 0.999
    both = out[s.ENR_BIN].notna() & ours[s.ENR_BIN].notna()
    assert (out.loc[both, s.ENR_BIN] == ours.loc[both, s.ENR_BIN]).mean() >= 0.98


def test_norm_matches_vendored_when_campaign_given(src, vendored):
    out = ingest.to_internal(ingest.apply_spec(src, SPEC), PAIR, camp="AT110")
    for r in ("MACS", "FACS1"):
        assert np.abs(out[s.norm(r)] - vendored[s.norm(r)]).max() < 1e-6


def test_has_read_derived_from_counts(src, vendored):
    out = ingest.to_internal(ingest.apply_spec(src, SPEC), PAIR, camp="AT110")
    for r in ("MACS", "FACS1"):
        assert (out[s.has(r)] == vendored[s.has(r)].astype(bool)).all()


def test_upload_trains_end_to_end(src):
    out = ingest.to_internal(ingest.apply_spec(src, SPEC), PAIR, camp="AT110")
    tr = out[(out[s.SPLIT] == "train") & out[s.ENR_BIN].notna()]
    imgt, wt = encode.wt_cols("AT110")
    X, _, _ = encode.encode(tr[s.AA_ALN], imgt=imgt, wt=wt)
    m = model.fit(X, tr[s.ENR_BIN])
    assert model.auc(m, X, tr[s.ENR_BIN]) > 0.8


def test_to_internal_rejects_a_round_without_counts(src):
    with pytest.raises(ValueError, match="no counts"):
        ingest.to_internal(ingest.apply_spec(src, SPEC), ("FACS2", "MACS"))


def test_drop_gaps(tmp_path, vendored):
    p = tmp_path / "gapped.csv"
    pd.DataFrame({"clone": ["a", "b"], "protein": ["QV-L", "QVWL"],
                  "input_ct": [5, 9], "r1_ct": [7, 2]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="must be|ragged"):
        ingest.apply_spec(p, ingest.Spec(var="clone", seq="protein", rnds=SPEC.rnds,
                                         drop_gaps=True))


def write(tmp_path, vendored, rnds, name):
    cols = {"cid": vendored[s.VAR], "seq": vendored[s.AA_ALN]}
    cols.update({f"c_{r}": vendored[s.cnt(r)] for r in rnds})
    p = tmp_path / name
    pd.DataFrame(cols).to_csv(p, index=False)
    sp = ingest.Spec(var="cid", seq="seq", rnds={r: f"c_{r}" for r in rnds})
    return ingest.apply_spec(p, sp, n_pos=126)


def test_run_upload_two_rounds_is_within_pair(tmp_path, vendored):
    from truehit_atg import pipeline as pl

    u = pl.run_upload(write(tmp_path, vendored, ["MACS", "FACS1"], "two.csv"), "AT110")
    assert u.mode == "within-pair"
    assert u.rnds == ["MACS", "FACS1"]
    assert u.n_tr > 1000 and u.n_te > 100
    assert 0.0 <= u.auc <= 1.0


def test_run_upload_three_rounds_is_cross_round(tmp_path, vendored):
    from truehit_atg import pipeline as pl

    u = pl.run_upload(write(tmp_path, vendored, ["MACS", "FACS1", "FACS2"], "three.csv"), "AT110")
    assert u.mode == "cross-round"
    assert u.auc > 0.85
    assert len(u.cf) > 20
    assert u.n_feat == 2520


def test_run_upload_never_says_affinity(tmp_path, vendored):
    from truehit_atg import pipeline as pl

    u = pl.run_upload(write(tmp_path, vendored, ["MACS", "FACS1"], "cols.csv"), "AT110")
    for c in u.ranked.columns:
        assert "affinity" not in c.lower() and "binding" not in c.lower()


def test_run_upload_rejects_one_round(tmp_path, vendored):
    from truehit_atg import pipeline as pl

    raw = write(tmp_path, vendored, ["MACS", "FACS1"], "one.csv").drop(columns=[s.cnt("FACS1")])
    with pytest.raises(ValueError, match="at least two rounds"):
        pl.run_upload(raw, "AT110")


def test_run_upload_rejects_too_few_labelled_rows(tmp_path, vendored):
    from truehit_atg import pipeline as pl

    raw = write(tmp_path, vendored, ["MACS", "FACS1"], "small.csv").head(12)
    with pytest.raises(ValueError, match="survive labelling"):
        pl.run_upload(raw, "AT110")


def test_weak_fit_is_flagged(tmp_path, vendored):
    from truehit_atg import pipeline as pl

    # a 900-row slice carries almost no signal; the ranking must not pass as a result
    u = pl.run_upload(write(tmp_path, vendored.head(900), ["MACS", "FACS1"], "weak.csv"), "AT110")
    assert u.weak
    assert u.spread < 0.05


def test_full_file_is_not_flagged_weak(tmp_path, vendored):
    from truehit_atg import pipeline as pl

    u = pl.run_upload(write(tmp_path, vendored, ["MACS", "FACS1", "FACS2"], "strong.csv"), "AT110")
    assert not u.weak
