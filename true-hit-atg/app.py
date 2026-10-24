import json
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from truehit_atg import agent, ingest, pipeline as pl, schema as s
from truehit_atg.config import CFG

st.set_page_config(page_title="TrueHit ATG", layout="wide")


@st.cache_resource(show_spinner=False)
def get_run(camp, tp, sp, lam, min_ct):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pl.run(camp, train=tp, test=sp, lam=lam, min_ct=min_ct)


# paul et al. table 1, L1 logistic, for the round pairs they published
PUB = {
    ("AT110", ("FACS1", "MACS"), ("FACS2", "MACS")): (0.964, 0.949),
    ("RX002", ("FACS1", "MACS"), ("FACS2", "MACS")): (0.935, 0.947),
}


def pair_lab(p):
    return f"{p[0]} / {p[1]}"


st.title("TrueHit ATG")
st.caption(
    "Ranks nanobody variants by **predicted enrichment** — what survives selection — from "
    "sequence alone. Enrichment is a proxy for binding confounded by expression and "
    "stickiness, so this does not predict binding affinity. An open implementation of "
    "Paul et al. (2026)."
)

with st.sidebar:
    st.header("Campaign")
    camp = st.selectbox("Nanobody", list(s.CAMPS), index=list(s.CAMPS).index(CFG.camp))
    c = s.CAMPS[camp]
    st.caption(f"target {c.tgt} · rounds {', '.join(c.rnds)} · {c.n_pos} aligned positions")
    ps = s.pairs(camp)
    labs = [pair_lab(p) for p in ps]
    tp = ps[labs.index(st.selectbox("Train on", labs, index=labs.index(pair_lab(CFG.train))
                                    if pair_lab(CFG.train) in labs else 0))]
    sp = ps[labs.index(st.selectbox("Test on", labs, index=labs.index(pair_lab(CFG.test))
                                    if pair_lab(CFG.test) in labs else 0))]
    st.header("Model")
    lam = st.select_slider("L1 strength", [0.5, 1.0, 2.0, 5.0, 10.0], value=CFG.lam)
    min_ct = st.number_input("Drop clones below this count", 0, 100, CFG.min_ct)
    st.caption("The vendored tables are already filtered upstream, so this does little here.")
    go = st.button("Run pipeline", type="primary", use_container_width=True)

if not go and "ran" not in st.session_state:
    st.info("Pick a campaign and press **Run pipeline**. A full run takes a few seconds on CPU.")
    st.stop()
st.session_state["ran"] = True

try:
    with st.spinner("Computing enrichment, encoding, fitting…"):
        r = get_run(camp, tuple(tp), tuple(sp), lam, min_ct)
except ValueError as e:
    st.error(str(e))
    st.stop()

if not r.conv:
    st.warning("The solver hit its iteration cap — coefficients may not be settled.")

t1, t2, t3, t4, t5, t6 = st.tabs(
    ["Enrichment", "Model", "Candidates", "Substitution preferences", "Score sequences",
     "Onboard a dataset"]
)

with t1:
    st.subheader(f"Enrichment, {pair_lab(tp)}")
    k = st.columns(4)
    k[0].metric("Variants", f"{len(r.tbl):,}")
    k[1].metric("Labelled", f"{r.n_tr + r.n_te:,}")
    k[2].metric("Abstained", f"{r.n_abst:,}")
    k[3].metric("Train / test", f"{r.n_tr:,} / {r.n_te:,}")
    st.caption(
        "Counts become counts per million with a 1 CPM floor, then log10, then late minus "
        "early. A two-component mixture splits the result and abstains where neither "
        "component is 80% confident — those are the abstained rows, not failures."
    )

    lo = r.tbl[r.tbl[s.ENR_BIN] == 0][s.ENR]
    hi = r.tbl[r.tbl[s.ENR_BIN] == 1][s.ENR]
    ab = r.tbl[r.tbl[s.ENR_BIN].isna()][s.ENR]
    f, ax = plt.subplots(figsize=(8, 3))
    bins = np.linspace(r.tbl[s.ENR].min(), r.tbl[s.ENR].max(), 70)
    ax.hist([lo, hi, ab], bins=bins, stacked=True, label=["depleted", "enriched", "abstained"],
            color=["#4c72b0", "#c44e52", "#cccccc"])
    ax.set_xlabel("log10 enrichment ratio")
    ax.set_ylabel("variants")
    ax.legend(frameon=False)
    f.tight_layout()
    st.pyplot(f)

    st.markdown("**Biggest risers by measured enrichment**")
    top = r.tbl.nlargest(15, s.ENR)[[s.VAR, s.CDR3, s.ENR, s.cnt(tp[1]), s.cnt(tp[0])]]
    st.dataframe(top, use_container_width=True, hide_index=True)

with t2:
    st.subheader(f"Trained on {pair_lab(tp)}, tested on {pair_lab(sp)}")
    k = st.columns(5)
    k[0].metric("Test ROC-AUC", f"{r.auc_te:.3f}")
    k[1].metric("Test PR-AUC", f"{r.pr_te:.3f}")
    k[2].metric("Strict ROC-AUC", f"{r.auc_strict:.3f}", help=f"{r.n_strict:,} rows never seen in training")
    k[3].metric("Train ROC-AUC", f"{r.auc_tr:.3f}")
    k[4].metric("Features", f"{r.n_feat:,}")

    pub = PUB.get((camp, tp, sp))
    if pub:
        st.caption(
            f"The paper's Table 1 reports ROC-AUC {pub[0]:.3f} and PR-AUC {pub[1]:.3f} for this "
            f"same model on {camp}. Ours are {r.auc_te:.3f} and {r.pr_te:.3f}."
        )

    st.markdown("**Against baselines**")
    b = st.columns(3)
    b[0].metric("Substitutions from parent", f"{r.auc_nmut:.3f}")
    b[1].metric("Observed counts", f"{r.auc_cnt:.3f}")
    b[2].metric("Model, within fixed load", f"{r.auc_within:.3f}")
    st.caption(
        "Counting substitutions from the parent is the fair comparison, because like the model "
        "it sees only sequence. Ranking by observed counts scores higher still, but the label "
        "is arithmetic on those same counts, so that number is circular and is shown only to "
        "make the circularity visible."
    )
    st.caption(
        f"**The third number is the one that settles it.** Mutation count cannot order two "
        f"variants carrying the same number of substitutions, so within a fixed load it scores "
        f"0.500 by construction. The model scores {r.auc_within:.3f} there — sequence signal "
        "with the mutation-load confound removed. Single substitutions are the exception: this "
        "data has too few enriched single mutants to measure that case either way."
    )

    f, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(*r.roc, color="#c44e52", lw=2, label=f"model, AUC {r.auc_te:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", c="#999", lw=1, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(frameon=False, loc="lower right")
    f.tight_layout()
    st.pyplot(f)

with t3:
    st.subheader("Ranked candidates")
    st.caption(
        f"Every variant in {camp} scored by the model. Predicted enrichment is a probability "
        "of surviving selection, not a predicted read count and not an affinity."
    )
    n = st.slider("Show", 10, 500, CFG.top_n, step=10)
    st.dataframe(r.ranked.head(n), use_container_width=True, hide_index=True)
    st.download_button(
        "Download full ranking (CSV)",
        r.ranked.to_csv(index=False).encode(),
        f"{camp}_{tp[0]}_{tp[1]}_ranked.csv",
        "text/csv",
    )

with t4:
    st.subheader("Per-position substitution preferences")
    st.caption(
        f"{len(r.cf)} of {r.n_feat:,} features survived the L1 penalty. Every feature is "
        "'this position carries a residue the parent does not', so the fitted weight reads "
        "directly as a preference for that substitution."
    )
    h = pl.heat(r)
    f, ax = plt.subplots(figsize=(min(0.22 * h.shape[1] + 2, 16), 5))
    v = np.nanmax(np.abs(h.values)) if h.size else 1.0
    im = ax.imshow(h.values, aspect="auto", cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_xticks(range(h.shape[1]))
    ax.set_xticklabels(h.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(h.shape[0]))
    ax.set_yticklabels(h.index, fontsize=8)
    ax.set_xlabel("IMGT position")
    ax.set_ylabel("substituted residue")
    f.colorbar(im, ax=ax, label="weight")
    f.tight_layout()
    st.pyplot(f)

    st.markdown("**Strongest substitutions**")
    st.dataframe(r.cf.head(20), use_container_width=True, hide_index=True)

with t5:
    st.subheader("Score sequences the campaign never contained")
    st.caption(
        f"Paste aligned protein sequences, one per line, each {len(r.wt)} positions for "
        f"{camp}, gaps as '-'. This is the part the counts cannot do: an unsequenced clone "
        "has no enrichment to rank by, but it does have a sequence."
    )
    st.warning(
        f"The model learned {camp} against {s.CAMPS[camp].tgt} only. A sequence for a "
        "different target is out of distribution and its score is not meaningful."
    )
    txt = st.text_area("Sequences", value=r.wt, height=140)
    if st.button("Score", type="primary"):
        try:
            out, unk = pl.score_seqs(r, txt.splitlines())
        except ValueError as e:
            st.error(str(e))
        else:
            if unk:
                st.warning(f"{unk} residues were not recognised and were read as gaps.")
            st.dataframe(out, use_container_width=True, hide_index=True)

with t6:
    st.subheader("Onboard someone else's dataset")
    st.caption(
        "Upload a CSV of clones with per-round counts. A model reads the column names and "
        "writes the converter; the file then runs through the same enrichment, encoding and "
        "fitting code as the vendored campaigns. The model only ever proposes a column "
        "mapping — it cannot touch the pipeline."
    )
    st.info(
        f"The sequences must already be aligned to the same {len(r.wt)} positions as {camp}. "
        "An upload with its own parent needs its own ANARCI numbering, which is not built yet.",
        icon=None,
    )

    up = st.file_uploader("CSV", type="csv", key="onboard_csv")
    if up is not None:
        tmp = Path(tempfile.gettempdir()) / f"truehit_upload_{abs(hash(up.name)) % 10**8}.csv"
        tmp.write_bytes(up.getvalue())

        try:
            desc = ingest.probe(tmp)
        except Exception as e:
            st.error(f"could not read that CSV: {e}")
            st.stop()

        st.markdown(f"**{desc['n_rows']:,} rows, {len(desc['cols'])} columns**")
        st.dataframe(
            {"column": desc["cols"], "dtype": [desc["dtypes"][c] for c in desc["cols"]]},
            use_container_width=True, hide_index=True,
        )

        if st.button("Ask the model to map it", type="primary"):
            try:
                with st.spinner("Reading the columns…"):
                    raw, spec, hist = agent.onboard(tmp, n_pos=len(r.wt))
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
            st.session_state["ob"] = (str(tmp), spec, [a.as_dict() for a in hist], len(r.wt))

    ob = st.session_state.get("ob")
    if ob:
        path, spec, hist, n_pos = ob
        st.markdown(f"**{len(hist)} attempt(s)**")
        for i, a in enumerate(hist, 1):
            ok = a["ok"]
            with st.expander(f"Attempt {i} — {'accepted' if ok else 'rejected'}", expanded=ok):
                if a["spec"]:
                    st.json(a["spec"])
                if a["err"]:
                    st.error(a["err"])
                if ok:
                    st.caption(f"{a['n_rows']:,} rows converted")

        if spec is None:
            st.error(
                "The model could not produce a usable mapping. The errors above say what it got "
                "wrong — the column names may be too opaque, or the file may not be per-round "
                "counts of aligned protein."
            )
        else:
            raw = ingest.apply_spec(path, spec, n_pos=n_pos)
            rnds = ingest.rnds_of(raw)
            st.markdown(f"**Rounds, earliest first:** {' -> '.join(rnds)}")
            if len(rnds) < 3:
                st.warning(
                    "Two rounds, so there is no third round to test against. The model will "
                    "train and rank, but **no honest accuracy number is available from this "
                    "upload** — a split within one round pair measures the experiment's own "
                    "quirks, not whether the ranking generalises. Add a later round to get one."
                )
            if st.button("Run the pipeline on this upload"):
                try:
                    with st.spinner("Enriching, encoding, fitting…"):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            u = pl.run_upload(raw, camp, lam=lam)
                except ValueError as e:
                    st.error(str(e))
                else:
                    k = st.columns(4)
                    k[0].metric("Variants", f"{u.n:,}")
                    k[1].metric("Train / test", f"{u.n_tr:,} / {u.n_te:,}")
                    k[2].metric("Features", f"{u.n_feat:,}")
                    k[3].metric("Substitutions kept", f"{len(u.cf):,}")

                    if u.mode == "cross-round":
                        m1, m2 = st.columns(2)
                        m1.metric("Cross-round ROC-AUC", f"{u.auc:.3f}")
                        m2.metric("Cross-round PR-AUC", f"{u.pr:.3f}")
                        st.caption(
                            f"Trained on {u.rnds[1]}/{u.rnds[0]} and tested on "
                            f"{u.rnds[2]}/{u.rnds[0]} — a later round the model never saw."
                        )
                    else:
                        st.caption(
                            "No accuracy number is shown as a headline because this upload "
                            "cannot support one."
                        )
                        with st.expander("Within-pair number, which is the weaker check"):
                            st.write(
                                f"ROC-AUC {u.auc:.3f}, PR-AUC {u.pr:.3f}, from a split inside "
                                f"{u.rnds[1]}/{u.rnds[0]}. Not comparable with the cross-round "
                                "figures elsewhere in this app, and not a measure of whether "
                                "the ranking generalises to a later round."
                            )

                    if u.weak:
                        st.error(
                            f"**This fit found little signal.** Only {len(u.cf)} substitutions "
                            f"survived the penalty and the predicted scores span "
                            f"{u.spread:.3f}, so the ranking below is close to an ordering of "
                            "noise. Usually too few variants, too few labelled rows, or a "
                            "slice of a campaign rather than the whole of one."
                        )

                    st.markdown("**Ranked candidates**")
                    st.dataframe(u.ranked.head(CFG.top_n), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download full ranking (CSV)",
                        u.ranked.to_csv(index=False).encode(),
                        "upload_ranked.csv", "text/csv", key="dl_up",
                    )
