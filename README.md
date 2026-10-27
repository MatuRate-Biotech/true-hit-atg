# MatuRate

ML-guided nanobody affinity maturation.

A nanobody is improved by building a library of mutated versions of a starting binder and
running rounds of selection that keep what sticks to the target and wash the rest away. Good
variants rise in share round over round; bad ones fall. That rise is **enrichment**, and it is
measurable from per-round sequencing counts.

This directory computes enrichment from per-round data, then trains a model to predict it from
sequence, so candidates can be ranked from an early round instead of running the whole
campaign. Fewer rounds means less time and less reagent.

**New here, or not a Python person?** Read [WALKTHROUGH.md](WALKTHROUGH.md) instead — the same
project in plain language, with setup for macOS and Windows and a tour of what each screen
means. This README is the developer view.

### Things to keep straight

1. **Not novel.** This is an open implementation of Paul et al. (2026). Pitch it as a clean,
   usable, open version. Our own contributions are the app, scoring outside clones, the
   cross-target comparison, and our own validation.
2. **Enrichment is not affinity.** The model predicts what survives selection, a proxy for
   binding confounded by expression and stickiness. Say "predicts enrichment" or "ranks
   candidates", never "predicts binding affinity" — not in the pitch, not in a column header,
   not in a variable name.
3. **It classifies, it does not forecast counts.** The result is binary and AUC-style: ranked by
   predicted selection success. Not "predicts exact later-round counts".
4. **Validate across rounds, not within.** Train on an early round, test on a later one.
   Held-out variants from the same round risk learning the experiment's quirks.
5. **Out of distribution is stated, not hidden.** A sequence from a different target gets a
   plain warning, not a confident-looking score.

The honest baseline is the enrichment ratios themselves: rank by what rose. The model has to
beat that to be worth anything.

### Data

Vendored at `data/paul2026/`, pinned to upstream commit `96c5b0a`. Three campaigns: AT110
(AT1R), NbB7 (β2AR), RX002 (RXFP1). No wifi needed. See [data/DATASET.md](data/DATASET.md) for
what ships, how many variants are actually labeled per round pair, what the dataset does *not*
include (raw FASTQ, measured affinities), and why AT110 `FACS1/MACS` → `FACS2/MACS` is the
recommended starting point.

### The handoff contract

`maturate/schema.py` defines the one table both layers pass around — one row per variant:

| Column | Meaning |
|---|---|
| `var_id` | upstream `seq_ID` |
| `aa` | protein sequence, gaps stripped |
| `aa_aln` | aligned sequence, fixed width per campaign, one-hots directly |
| `cdr3` | CDR3 |
| `cnt_<round>` | raw read count for that round |
| `norm_<round>` | normalized count for that round |
| `enr` | enrichment: log ratio of the chosen round pair |
| `enr_bin` | binary enriched label, NaN where upstream made no call |
| `split` | upstream's train/test assignment, NaN where unlabeled |

`enr` is set for every row; `enr_bin` and `split` are not. Use `data.labeled()` to get the
callable subset and `data.split(df, "train")` to pick a side. The `chk()` validator raises on a
table missing columns or carrying duplicate ids — call it on anything handed over.

Round pairs, round order and aligned width per campaign live in `schema.CAMPS`. Tunables —
which campaign, which train and test pair, count threshold, normalization — live in
`config.CFG`, not as constants in the analysis code.

### Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m streamlit run app.py
```

Run from this directory: `pythonpath = .` in `pytest.ini` is relative to here. The tests load
all three campaigns from the vendored files and assert the reproduction numbers below, so they
fail loudly if the data or the recipe drifts.

### The app

Five tabs, a few seconds per run on CPU, no GPU and no job queue.

| Tab | What it shows |
|---|---|
| Enrichment | log-ratio histogram with the mixture split drawn over it, and the abstained rows made visible rather than hidden |
| Model | test AUC, strict AUC, train AUC, both baselines, ROC curve |
| Candidates | every variant ranked by predicted enrichment, downloadable |
| Substitution preferences | per-position heatmap straight from the fitted weights |
| Score sequences | paste aligned sequences the campaign never contained and rank them |

`app.py` is a renderer. There is no `streamlit` import anywhere inside `maturate/`:
everything goes through `pipeline.run()`, which returns a plain `Res`, so a FastAPI service and
a React frontend can sit on the same call later.

### Usage

```python
from maturate import pipeline as pl

r = pl.run("AT110")                     # train FACS1/MACS, test FACS2/MACS
r.auc_te, r.auc_strict, r.auc_nmut      # 0.960, 0.958, 0.752
r.ranked.head(10)                       # var_id, pred_enr, pred_cls, cdr3, enr, counts
r.cf.head(10)                           # strongest substitutions, as IMGT position + residue

out, unk = pl.score_seqs(r, [some_seq]) # sequences the campaign never contained
```

### How faithful it is to the paper

Every number below is asserted by the test suite.

| Check | Result |
|---|---|
| Normalized counts vs shipped `norm_*` | exact |
| `hasread` | exactly `count >= 5`, every round, every campaign |
| Log ratios vs shipped | max error 7e-16 |
| Binary labels vs shipped | 99.4% agreement |
| Coefficients vs upstream's logistic params | sign agreement 1.000, Pearson r 0.983 |
| AT110 test AUC | 0.960, vs upstream's CNN at 0.938 on the same rows |

The recipe, recovered from the upstream code: counts per million against the **pre-filter**
library size with a 1 CPM floor, `log10`, late minus early, then a two-component Gaussian
mixture that abstains below 80% posterior, then an 80/20 split. The model is
`LogisticRegression(penalty='l1', solver='saga', C=0.5)` over a 21-symbol one-hot of the
aligned sequence with the parent's own residue dropped at each position — which is what makes
every feature a substitution indicator and every weight readable as a preference.

The pre-filter library sizes are not in the shipped tables; they are recovered exactly from the
normalized columns and live in `schema.LIBS`. An upload has only its own column sum, which
shifts log ratios by a constant only where neither round sits on the 1 CPM floor — a different
library size moves the floor, so floored rows shift by different amounts. Measured on AT110
that is Spearman 0.9994 against the true sizes, with the labels unchanged.

### Read the baselines carefully

On AT110, ranking by the observed counts scores **higher** than the model (0.989 vs 0.960).
That comparison is circular: the binary label is arithmetic on those same counts. 88% of
enriched test rows have a MACS count of exactly zero, against 0% of depleted rows, because a
clone absent from the input round is floored to 1 CPM and any later appearance then yields a
large positive ratio.

So the fair comparison is sequence-only, and there the model is far ahead: **0.960 against
0.752** for counting substitutions from the parent. The app displays the counts number anyway,
labelled as circular — better disclosed by us than discovered by a judge.

This is also the argument for the model. Ranking by counts only works for clones you already
sequenced. The model needs no counts, so it can score clones the experiment never contained.

### Next

Stretch work, in order, each independently droppable: a CSV upload path on the public
`{clone id, sequence, per-round counts}` format; a run on a dataset outside the paper
(ProteinGym first, then MaveDB); an ANARCI alignment layer for ragged input; a React frontend
over the same `pipeline.run()`.

Planned extension, not the foundation: a growth-versus-binding flag for clones that rise
because they replicate faster rather than because they bind. Confirm the framing with Jessica
before building — see [claude-docs/context.md](claude-docs/context.md) section 9.
