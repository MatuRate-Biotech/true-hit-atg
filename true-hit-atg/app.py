import json
import re
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import streamlit as st

from truehit_atg import agent, enrich, ingest, pipeline as pl, schema as s
from truehit_atg.config import CFG

st.set_page_config(page_title="TrueHit ATG", layout="wide")


UP = "Uploaded file"


# the four facts every tab needs about whichever dataset is loaded. a vendored campaign has
# them in s.CAMPS; an upload has no Camp record, so it carries its own. tgt is empty for an
# upload because nothing in the file says what it was raised against.
@dataclass
class Dset:
    name: str
    rnds: list
    n_pos: int
    tgt: str = ""
    imgt: bool = True


@st.cache_resource(show_spinner=False)
def get_run(camp, tp, sp, lam, min_ct):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pl.run(camp, train=tp, test=sp, lam=lam, min_ct=min_ct)


# _spec is not hashed; path, width and parent already identify the run.
@st.cache_resource(show_spinner=False)
def get_upload(path, n_pos, par, al, lam, nm, _spec):
    raw = ingest.apply_spec(path, _spec, n_pos=n_pos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pl.run_upload(raw, camp=None if par else al, parent=par or None, lam=lam,
                             name=nm)


# paul et al. table 1, L1 logistic, for the round pairs they published
PUB = {
    ("AT110", ("FACS1", "MACS"), ("FACS2", "MACS")): (0.964, 0.949),
    ("RX002", ("FACS1", "MACS"), ("FACS2", "MACS")): (0.935, 0.947),
}


@st.cache_data(show_spinner=False)
def try_direct(path, n_pos, _size):
    """The identity Spec, if the file already conforms. _size only busts the cache when a
    different file arrives under the same name."""
    cols = ingest.probe(path)["cols"]
    sp = ingest.identity_spec(cols)
    # apply_spec phrases its errors for a mapping the model wrote, naming the temp file it
    # was handed. say which contract columns are absent instead.
    want = [c for c in (s.VAR, s.AA_ALN) if c not in cols]
    if want or len(sp.rnds) < 2:
        if not want:
            want = [f"at least two {s.cnt('')} columns"]
        return None, f"missing {', '.join(want)}; the file has {', '.join(cols)}"
    try:
        ingest.apply_spec(path, sp, n_pos=n_pos)
    except ValueError as e:
        return None, str(e)
    return sp, ""


# only `key` is hashed: it identifies the run the other three came from.
@st.cache_data(show_spinner=False)
def enr_csv(_rr, _order, _tp, key):
    d = _rr.tbl.sort_values(s.ENR, ascending=False)
    out = cnt_cols(d[[s.ENR, s.cnt(_tp[1]), s.cnt(_tp[0])]], _order).rename(
        columns={s.ENR: "Enrichment"}
    )
    out.insert(0, "Substitutions", [subs(q, _rr.wt, _rr.imgt, cap=None) for q in d[s.AA_ALN]])
    out.insert(0, "Row label", d[s.VAR].values)
    out["Sequence"] = d[s.AA_ALN].values
    return out.to_csv(index=False).encode()


def pair_lab(p, order=None):
    return s.pair_lab(ds.rnds if order is None else order, p)


# an uploaded file's name goes into a download filename, so strip what a filesystem minds
def slug(x):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", x).strip("_") or "dataset"


def cnt_cols(df, order):
    """count columns get the round's display name; field names are left alone here"""
    return df.rename(columns={s.cnt(rr): f"Reads, {s.rnd_lab(order, rr)}" for rr in order})


# parent residue, position, variant residue. positions come from the alignment the model
# used: IMGT labels for a vendored campaign, 1..L for an upload with its own parent.
def subs(seq, wt, imgt, cap=10):
    d = [f"{a}{p}{b}" for a, b, p in zip(wt, seq, imgt) if a != b]
    if cap is not None and len(d) > cap:
        return ", ".join(d[:cap]) + f", +{len(d) - cap} more"
    return ", ".join(d)


def norm(x):
    return x.replace(" ", "").replace("_", "").lower()


def provenance(order):
    # an upload whose rounds are already called Input and Output1 would otherwise get
    # "Input is the Input round", which says nothing
    if all(norm(s.rnd_lab(order, rr)) == norm(rr) for rr in order):
        return ""
    bits = [f"{s.rnd_lab(order, rr)} is the {rr} round" if i == 0
            else f"{s.rnd_lab(order, rr)} is {rr}" for i, rr in enumerate(order)]
    return "; ".join(bits) + "."


st.title("TrueHit ATG")
st.caption(
    "Ranks nanobody variants by **predicted enrichment** — what survives selection — from "
    "sequence alone. Enrichment is a proxy for binding confounded by expression and "
    "stickiness, so this does not predict binding affinity. An open implementation of "
    "Paul et al. (2026)."
)

ob = st.session_state.get("ob")
# a pre-scored upload has no rounds and no count columns, so the round-pair and enrichment
# tabs have nothing to show for it. it stays on the Onboard tab, where it already works.
have_up = bool(ob and ob[1] is not None and not ob[1].enr_col and len(ob[1].rnds) >= 2)

with st.sidebar:
    st.header("Dataset")
    # an onboarded file joins the vendored campaigns here, so every tab can render it
    opts = list(s.CAMPS) + ([UP] if have_up else [])
    camp = st.selectbox("Dataset", opts, index=opts.index(CFG.camp))
    up = camp == UP

    if up:
        ob_path, ob_spec, _, ob_npos, ob_par, _, ob_al, ob_nm = ob
        ds = Dset(ob_nm, list(ob_spec.rnds), ob_npos, imgt=not ob_par)
        st.caption(
            f"uploaded · rounds {', '.join(s.rnd_lab(ds.rnds, rr) for rr in ds.rnds)} · "
            f"{ds.n_pos} aligned positions"
            + ("" if ds.imgt else ", numbered 1..L from the parent you supplied")
        )
    else:
        c = s.CAMPS[camp]
        ds = Dset(camp, list(c.rnds), c.n_pos, tgt=c.tgt)
        st.caption(
            f"target {c.tgt} · rounds "
            f"{', '.join(s.rnd_lab(ds.rnds, rr) for rr in ds.rnds)} · {ds.n_pos} aligned positions"
        )

    if up:
        # run_upload takes the pair from round order rather than from a choice, so there is
        # nothing to select. shown so the pair on screen is never a guess.
        tp = (ds.rnds[1], ds.rnds[0])
        sp = (ds.rnds[2], ds.rnds[0]) if len(ds.rnds) >= 3 else tp
        st.caption(
            f"**Train on** {pair_lab(tp)}  \n**Test on** {pair_lab(sp)}"
            + ("" if sp != tp else " — the same pair, no later round to test against")
        )
    else:
        ps = s.pairs(camp)
        labs = [pair_lab(p) for p in ps]
        tp = ps[labs.index(st.selectbox("Train on", labs, index=labs.index(pair_lab(CFG.train))
                                        if pair_lab(CFG.train) in labs else 0))]
        sp = ps[labs.index(st.selectbox("Test on", labs, index=labs.index(pair_lab(CFG.test))
                                        if pair_lab(CFG.test) in labs else 0))]

    st.header("Model")
    lam = st.select_slider("L1 strength", [0.5, 1.0, 2.0, 5.0, 10.0], value=CFG.lam)
    # run_upload has no count filter, so the control is not offered where it would do nothing
    min_ct = CFG.min_ct
    if not up:
        min_ct = st.number_input("Drop clones below this count", 0, 100, CFG.min_ct)
        st.caption("The vendored tables are already filtered upstream, so this does little here.")
    go = st.button("Run pipeline", type="primary", use_container_width=True)

if not go and "ran" not in st.session_state:
    st.info("Pick a dataset and press **Run pipeline**. A full run takes a few seconds on CPU.")
    st.stop()
st.session_state["ran"] = True

try:
    with st.spinner("Computing enrichment, encoding, fitting…"):
        if up:
            r = get_upload(ob_path, ob_npos, ob_par, ob_al, lam, ob_nm, ob_spec)
        else:
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
    st.caption(provenance(ds.rnds))
    k = st.columns(3)
    k[0].metric("Variants", f"{len(r.tbl):,}")
    k[1].metric("Labelled", f"{r.n_tr + r.n_te:,}")
    k[2].metric("Abstained", f"{r.n_abst:,}")
    st.caption(
        "Abstained: variants with too few reads to call, or whose enrichment sits between "
        "the two humps."
    )
    lab_in, lab_out = s.rnd_lab(ds.rnds, tp[1]), s.rnd_lab(ds.rnds, tp[0])
    l_in, l_out = enrich.lib(r.tbl, tp[1], camp), enrich.lib(r.tbl, tp[0], camp)
    with st.expander("How enrichment is calculated"):
        st.markdown(
            "Each variant has a read count in every round. The rounds were sequenced to "
            "different depths, so raw counts are not comparable — "
            f"{ds.name} has {l_in:,} reads in {lab_in} and {l_out:,} in {lab_out}."
        )
        st.markdown(
            """
1. Each count becomes a share of its own round, in counts per million: count divided by
   the round's total, times one million.
2. Anything below 1 count per million is raised to 1. A variant with no reads would
   otherwise be zero, and the next step takes a logarithm.
3. Take the base-10 logarithm of both values.
4. Subtract the earlier round from the later one.
"""
        )
        st.markdown(
            "Subtracting logarithms divides the original numbers, so the result is the "
            "fold change in the variant's share of the library, on a log scale. +1 is ten "
            "times more abundant, 0 is unchanged, −1 is ten times less."
        )
        # run through the loaded dataset's own library sizes, so the arithmetic on screen is
        # the arithmetic the pipeline did
        e_in, e_out = 148 * 1e6 / l_in, 482 * 1e6 / l_out
        d = np.log10(e_out) - np.log10(e_in)
        st.markdown(
            f"A worked example. A variant with 148 reads in {lab_in} and 482 in {lab_out} "
            f"becomes {e_in:,.1f} and {e_out:,.1f} counts per million, then "
            f"{np.log10(e_in):.4f} and {np.log10(e_out):.4f} after the logarithm, giving an "
            f"enrichment of {d:+.2f} — a {(e_out / e_in if d >= 0 else e_in / e_out):.1f}-fold "
            f"{'rise' if d >= 0 else 'fall'}."
        )
        st.markdown(
            "Enrichment measures what survived selection. It is a proxy for binding, "
            "confounded by how well a variant is expressed and by general stickiness, so "
            "it is not a measured binding affinity."
        )

    with st.expander("How the enriched and depleted calls are made"):
        txt = (
            "The enrichment scores form two separate humps rather than a spread around "
            "zero. Two bell curves are fitted to them, and each variant gets a probability "
            "of belonging to each one. A variant at least "
            f"{CFG.gmm_tr:.0%} likely to belong to one hump gets that label. "
        )
        if r.mix:
            lo, hi, w = r.mix
            txt += (f"On {ds.name} the humps sit at {lo:.2f} and {hi:+.2f}, and {w:.0%} of "
                    "variants fall in the lower one. ")
        st.markdown(
            txt + "Variants between the humps get no label, and neither do variants with "
            "too few reads to be scored at all."
        )

    lo = r.tbl[r.tbl[s.ENR_BIN] == 0][s.ENR]
    hi = r.tbl[r.tbl[s.ENR_BIN] == 1][s.ENR]
    ab = r.tbl[r.tbl[s.ENR_BIN].isna()][s.ENR]
    f, ax = plt.subplots(figsize=(8, 3))
    bins = np.linspace(r.tbl[s.ENR].min(), r.tbl[s.ENR].max(), 70)
    ax.hist([lo, hi, ab], bins=bins, stacked=True, label=["depleted", "enriched", "abstained"],
            color=["#4c72b0", "#c44e52", "#cccccc"])
    ax.set_xlabel(f"Enrichment (log10, {pair_lab(tp)})")
    ax.set_ylabel("variants")
    ax.legend(frameon=False)
    f.tight_layout()
    st.pyplot(f)

    st.markdown("**Biggest risers by measured enrichment**")
    top = r.tbl.nlargest(15, s.ENR)
    show = cnt_cols(top[[s.ENR, s.cnt(tp[1]), s.cnt(tp[0])]], ds.rnds).rename(
        columns={s.ENR: "Enrichment"}
    )
    show.insert(0, "Substitutions", [subs(q, r.wt, r.imgt) for q in top[s.AA_ALN]])
    st.dataframe(show, use_container_width=True, hide_index=True)

    st.download_button(
        f"Download all {len(r.tbl):,} variants (CSV)",
        enr_csv(r, ds.rnds, tuple(tp), (camp, ds.name, tuple(tp), tuple(sp), lam, min_ct)),
        f"{slug(ds.name)}_enrichment.csv",
        "text/csv",
        key="dl_enr",
    )
    st.caption(
        "The table shows the top 15 and truncates the substitution list at ten. The CSV carries "
        "every variant, every substitution, the full aligned sequence, and the upstream "
        "row label."
    )

with t2:
    same = tp == sp

    if same:
        st.subheader(f"Trained and tested on {pair_lab(tp)}")

        # percentages come from the split that actually happened, not from CFG.test_size,
        # so they stay true if the threshold or the labelling changes
        lab = r.n_tr + r.n_te
        st.caption(
            f"{len(r.tbl):,} variants. {lab:,} received a confident enriched/depleted label "
            f"from the mixture model; {r.n_abst:,} abstained. The labelled set was split at "
            f"random into {r.n_tr / lab:.0%} training ({r.n_tr:,} variants) and "
            f"{r.n_te / lab:.0%} held-out ({r.n_te:,} variants). All numbers on this tab come "
            "from the held-out variants."
        )

        a1, a2 = st.columns(2)
        a1.metric("Test ROC-AUC", f"{r.auc_te:.3f}")
        a1.caption(
            "Take one variant that enriched and one that didn't. This is how often the model "
            "scores the enriched one higher. 0.5 means random, 1.0 means always correct. "
            f"On the variants used for training, the same score is {r.auc_tr:.3f}."
        )
        a2.metric("Features", f"{r.n_feat:,}")
        a2.caption(
            "One slot for every amino acid substitution the model could use: "
            f"{len(r.wt)} aligned positions x 20 alternatives to the parent's own residue. "
            f"{len(r.cf)} of them received a non-zero weight at the current L1 strength."
        )

        b1, b2 = st.columns(2)
        b1.metric("Substitutions from parent", f"{r.auc_nmut:.3f}")
        b1.caption(
            "The same score for a model that uses only the number of positions differing "
            "from the parent, ignoring which ones."
        )
        b2.metric("Model within fixed load", f"{r.auc_within:.3f}")
        b2.caption(
            "The same score computed only between variants carrying an equal number of "
            "substitutions. A model using substitution count alone scores 0.500 here."
        )

        kk, pk = r.pk
        f, ax = plt.subplots(figsize=(6, 4))
        ax.plot(kk, pk, color="#c44e52", lw=2, label="model")
        ax.axhline(r.base, ls="--", c="#999", lw=1, label=f"random ({r.base:.0%})")
        # K stops at the held-out set: an unlabelled variant has no ground truth to score
        # against. the head is noisy by construction, so the axis starts at 10.
        ax.set_xlim(10, int(kk[-1]))
        ax.set_ylim(0, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.set_xlabel("K, variants taken from the top of the ranking")
        ax.set_ylabel("share that enriched")
        ax.legend(frameon=False, loc="upper right")
        f.tight_layout()
        st.pyplot(f)
        st.caption(
            "Of the top K variants the model ranks, the share that enriched in the held-out "
            f"data. The flat line is the share expected from picking K variants at random "
            f"({r.base:.0%})."
        )

    else:
        st.subheader(f"Trained on {pair_lab(tp)}, tested on {pair_lab(sp)}")
        k = st.columns(5)
        k[0].metric("Test ROC-AUC", f"{r.auc_te:.3f}")
        k[1].metric("Test PR-AUC", f"{r.pr_te:.3f}")
        k[2].metric("Strict ROC-AUC", f"{r.auc_strict:.3f}", help=f"{r.n_strict:,} rows never seen in training")
        k[3].metric("Train ROC-AUC", f"{r.auc_tr:.3f}")
        k[4].metric("Features", f"{r.n_feat:,}")

        # the paper published figures for its own campaigns and round pairs only
        pub = None if up else PUB.get((camp, tp, sp))
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
        f"Every variant in {ds.name} scored by the model. Predicted enrichment is a probability "
        "of surviving selection, not a predicted read count and not an affinity."
    )
    n = st.slider("Show", 10, 500, CFG.top_n, step=10)
    st.dataframe(cnt_cols(r.ranked.head(n), ds.rnds), use_container_width=True,
                 hide_index=True)
    st.download_button(
        "Download full ranking (CSV)",
        r.ranked.to_csv(index=False).encode(),
        f"{slug(ds.name)}_{slug(tp[0])}_{slug(tp[1])}_ranked.csv",
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
    ax.set_xlabel("IMGT position" if ds.imgt else "position")
    ax.set_ylabel("amino acid substituted in")
    f.colorbar(im, ax=ax, label="weight (beta)")
    f.tight_layout()
    st.pyplot(f)
    # an upload numbers positions 1..L from its own parent, so there is no IMGT scheme to explain
    if ds.imgt:
        st.caption(
            "Positions use IMGT numbering, a standard scheme where the same number refers to "
            "the same structural position in any antibody. Positions with a letter, such as "
            "111A, are insertion codes for loops that vary in length between antibodies."
        )

    st.markdown("**Strongest substitutions**")
    pos_lab = "Position (IMGT)" if ds.imgt else "Position"
    tbl = r.cf.head(20)[["sub", "imgt", "aa", "beta"]].rename(
        columns={"sub": "Substitution", "imgt": pos_lab, "aa": "Amino acid"}
    )
    st.dataframe(tbl, use_container_width=True, hide_index=True)
    st.caption(
        "Substitution is the parent residue, position and substituted residue."
    )
    # the two worked figures are fixed illustrations of the scale, not values from this fit
    st.caption(
        "**beta** — the model's learned weight for that substitution, in log-odds. Positive "
        "means the substitution raises the odds of a variant being enriched, negative lowers "
        "them. A beta of +4.34 multiplies the odds by about 76; −1.0 divides them by about 2.7."
    )

with t5:
    st.subheader("Score sequences the dataset never contained")
    st.caption(
        f"Paste aligned protein sequences, one per line, each {len(r.wt)} positions for "
        f"{ds.name}, gaps as '-'. This is the part the counts cannot do: an unsequenced clone "
        "has no enrichment to rank by, but it does have a sequence."
    )
    # an upload carries no target name, so the warning is phrased against the dataset itself
    st.warning(
        f"The model learned {ds.name} against {ds.tgt} only. A sequence for a different "
        "target is out of distribution and its score is not meaningful."
        if ds.tgt else
        f"The model learned the selection in {ds.name} only. A sequence from a different "
        "parent, target or assay is out of distribution and its score is not meaningful."
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
    st.caption(
        "Leave the parent box empty to borrow a vendored campaign's alignment, or paste the "
        "dataset's own reference sequence to run on anything else. With a parent supplied, "
        "positions are numbered 1..L rather than IMGT, which is antibody-specific."
    )
    # the alignment to borrow has to be a real campaign; the sidebar may be on the upload
    al = ob_al if up else camp
    al_pos = s.CAMPS[al].n_pos
    par = "".join(
        st.text_area(
            f"Parent sequence (optional; blank uses {al}'s {al_pos}-position alignment)",
            value="", height=68, key="ob_parent",
        ).split()
    ).upper()

    fu = st.file_uploader("CSV", type="csv", key="onboard_csv")
    # a stored mapping belongs to the file it was built from. without this a second upload
    # renders the first file's result underneath its own error.
    prev = st.session_state.get("ob")
    if fu is None and prev:
        del st.session_state["ob"]
    if fu is not None:
        nm = Path(fu.name).stem
        tmp = Path(tempfile.gettempdir()) / f"truehit_upload_{abs(hash(fu.name)) % 10**8}.csv"
        tmp.write_bytes(fu.getvalue())

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

        n_pos = len(par) if par else al_pos
        direct, why = try_direct(str(tmp), n_pos, len(fu.getvalue()))
        if direct is not None:
            # the sidebar is drawn before this tab, so it only sees a new mapping on the
            # next run. rerun once, when the mapping actually changes.
            new = (str(tmp), direct, [], n_pos, par, "direct", al, nm)
            if prev != new:
                st.session_state["ob"] = new
                st.rerun()
        else:
            if prev and prev[0] != str(tmp):
                del st.session_state["ob"]
            st.warning(
                "The file does not use the contract's column names, so it cannot be read "
                f"directly. {why}"
            )
            if st.button("Ask the model to map it", type="primary"):
                try:
                    with st.spinner("Reading the columns…"):
                        raw, spec, hist = agent.onboard(tmp, n_pos=n_pos)
                except RuntimeError as e:
                    st.error(str(e))
                    st.stop()
                st.session_state["ob"] = (
                    str(tmp), spec, [a.as_dict() for a in hist], n_pos, par, "agent", al, nm
                )
                st.rerun()

    ob = st.session_state.get("ob")
    if ob:
        path, spec, hist, n_pos, par, how, al, nm = ob
        if how == "direct":
            st.success(
                "Read directly. The file already uses var_id, aa_aln and cnt_ columns, so no "
                "model was called."
            )
        else:
            st.info("Mapped by the model.")
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
            if ingest.scored(raw):
                st.markdown("**Already scored**, so there are no rounds to enrich.")
                st.warning(
                    "This upload arrives with its own score, so **the enrichment code does "
                    "not run at all** — only the encoding and the model. Nothing here says "
                    "anything about how enrichment behaves on outside data, because it was "
                    "never exercised on it."
                )
                st.caption(
                    "A pre-scored upload has no rounds, so it stays on this tab. The other "
                    "tabs are built around per-round counts."
                )
            else:
                st.markdown(
                    "**Rounds, earliest first:** "
                    + " -> ".join(s.rnd_lab(rnds, rr) for rr in rnds)
                )
                st.caption(provenance(rnds))
                if len(rnds) >= 2:
                    st.success(
                        f"**{UP}** is now in the sidebar's dataset list. Pick it there and "
                        "press Run pipeline to see this file in every tab."
                    )
            if rnds and len(rnds) < 3:
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
                            u = pl.run_upload(raw, camp=None if par else al,
                                              parent=par or None, lam=lam, name=nm)
                except ValueError as e:
                    st.error(str(e))
                else:
                    k = st.columns(4)
                    k[0].metric("Variants", f"{u.n:,}")
                    k[1].metric("Train / test", f"{u.n_tr:,} / {u.n_te:,}")
                    k[2].metric("Features", f"{u.n_feat:,}")
                    k[3].metric("Substitutions kept", f"{len(u.cf):,}")

                    if u.mode == "pre-scored":
                        st.caption(
                            f"Labels came from the file ({u.labels}). Held-out ROC-AUC "
                            f"{u.auc_te:.3f}, PR-AUC {u.pr_te:.3f}, on a random split of the scored "
                            "variants."
                        )
                    elif u.mode == "cross-round":
                        m1, m2 = st.columns(2)
                        m1.metric("Cross-round ROC-AUC", f"{u.auc_te:.3f}")
                        m2.metric("Cross-round PR-AUC", f"{u.pr_te:.3f}")
                        st.caption(
                            f"Trained on {pair_lab((u.rnds[1], u.rnds[0]), u.rnds)} and "
                            f"tested on {pair_lab((u.rnds[2], u.rnds[0]), u.rnds)} — a later "
                            "round the model never saw."
                        )
                    else:
                        st.caption(
                            "No accuracy number is shown as a headline because this upload "
                            "cannot support one."
                        )
                        with st.expander("Within-pair number, which is the weaker check"):
                            st.write(
                                f"ROC-AUC {u.auc_te:.3f}, PR-AUC {u.pr_te:.3f}, from a split inside "
                                f"{pair_lab((u.rnds[1], u.rnds[0]), u.rnds)}. Not comparable "
                                "with the cross-round "
                                "figures elsewhere in this app, and not a measure of whether "
                                "the ranking generalises to a later round."
                            )

                    if u.weak:
                        st.error(
                            f"**This fit found little signal.** {len(u.cf)} substitutions "
                            f"survived the penalty, the scores span {u.spread:.3f}, and "
                            f"held-out ROC-AUC is {u.auc_te:.3f}. The ranking below is close to "
                            "an ordering of noise. Common causes: too few labelled rows, a "
                            "slice of a campaign rather than the whole of one, or a dataset "
                            "where every variant carries a substitution seen nowhere else — "
                            "a one-hot over substitutions cannot transfer to one it never saw."
                        )

                    st.markdown("**Ranked candidates**")
                    st.dataframe(cnt_cols(u.ranked.head(CFG.top_n), u.rnds),
                                 use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download full ranking (CSV)",
                        u.ranked.to_csv(index=False).encode(),
                        "upload_ranked.csv", "text/csv", key="dl_up",
                    )
