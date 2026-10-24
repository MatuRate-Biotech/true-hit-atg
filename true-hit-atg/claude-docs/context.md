# TrueHit ATG — working context

Scope: the inner `true-hit-atg/` directory only. This is the context for Idea 2, trimmed from
the repo-level `claude-docs/context.md`, which stays the source of truth for anything covering
both ideas. `true-hit-qc/` is parked; nothing here depends on it.

Event: Cal Hacks 13.0, Palace of Fine Arts, San Francisco. Hacking from Friday 2026-10-23
evening, judging Sunday. ~36 hours.

---

## 1. Who

- **Pratham Jangra** (GitHub `heller1729`). MS CS thesis student at Virginia Tech, GRA on the
  NSF-funded CyberSMART project. Backend and software, near-zero hardware experience. Owns the
  ML layer: sequence encoding, training, evaluation, and the app that returns ranked candidates.
- **Jessica Pan** (Slack "Percabeth"). Antibody and molecular biology background, authored the
  proposal this directory implements. Owns the data and enrichment layer: assembling per-round
  data, enrichment ratios, count thresholds and normalization, and the align-and-count front end
  if it survives scoping.

Two other teammates dropped out on Oct 23. Plan for a 2-person team.

Code style that applies to anything generated here:
- No comments that read as AI-authored.
- Short variable names (`mtc` over `matched`).
- Practical, forward-moving output over exhaustive scaffolding.

---

## 2. What this is

A tool that learns from antibody selection data to predict which binders are worth pursuing,
turning several rounds of lab screening into a model that ranks candidates after one.

Feed it sequencing data from an affinity-maturation campaign; it measures how much each variant
enriched round over round, trains a model to predict that enrichment, and ranks candidates — so
the good ones surface early with less lab work.

The vocabulary, in four ideas:

- A **binder** is a protein, here a **nanobody** (a small single-domain antibody), that sticks to
  a target. You have an okay one and want a better one.
- **Affinity maturation** improves it: build a library of mutated versions of the starting
  binder, then run rounds of selection that keep the best binders and wash the rest away.
- **Enrichment is the signal.** Good binders survive each round, so their share of the pool rises
  round over round; bad ones fall. Sequence the pool after each round and this is measurable.
  Rising is good, falling is bad.
- **The data is one table per round**: each variant and how many times it was seen. Compare
  across rounds for an enrichment value per variant.

### The problem

Affinity maturation works but is slow and expensive — narrowing a huge mutant library to a few
good binders takes multiple rounds, meaning weeks to months and a lot of pricey reagent. The
readout is also noisy: raw enrichment counts correlate only loosely with true binding, and later
rounds can enrich the wrong things — sticky, over-expressed or non-specific binders — alongside
the real ones.

The opportunity: a model trained on the sequencing data can learn which sequence features drive
enrichment and predict good variants from an early round instead of running the campaign to the
end. Recent work shows this works, with a model trained on a single sorting round predicting what
enriches several rounds later.

### Two layers

```
per-round sequencing data in
  -> Layer 1: data + enrichment    clean, count per variant per round, compute ratios
  -> Layer 2: ML                   train on enrichment, predict, rank
  -> ranked binders out
```

Layer 1 is Jessica's, Layer 2 is Pratham's. The table between them is the handoff contract and is
already locked down in code — see section 7.

**The honest baseline is the enrichment ratios themselves**: rank by what rose. The ML layer is
what adds value beyond that, and it has to beat it to be worth anything. The baseline ships even
if the ML turns out shaky.

---

## 3. The workflow

**Built and running as of 2026-10-24**, except step 1's upload, which reads the vendored tables
for now. Each step below names the module that implements it.

End to end, with two simplifying choices baked in: **input is assumed pre-aligned** (equal-length,
position-matched sequences), and **the model is logistic regression only — no CNN**. Because the
data is small and the model is simple, a full run takes seconds on a laptop CPU with no GPU, so
the app is a plain click-run-see-results flow — no job queue, nothing to babysit on stage.

**The model is trained fresh on every upload.** It is specific to that one campaign — that binder,
that target — and is never reused across datasets.

1. **Upload** — `data.py`, `schema.chk()`. One row per clone, pre-aligned sequence, read counts
   per round: Input/R1, optionally R2. *Guardrail built:* `chk()` refuses ragged input and names
   the offending rows rather than padding them, because padding moves every position after the
   change. Today the input is the vendored tables; a CSV upload is the first stretch item.
2. **Compute enrichment.** Per round: drop ultra-low-count clones as noise, normalize each clone's
   abundance by its earlier-round (Input/MACS) abundance, take the log — a log-enrichment ratio.
   Also assign a binary **enriched / depleted** label, since the data is bimodal and the binary
   task is what actually predicts well. Instant. **The paper does not threshold at zero** — it
   fits a 2-component GMM to the log-ratio distribution and abstains on the ambiguous middle.
   The exact recipe, verified against the vendored files, is in section 5. Implemented in
   `enrich.py`, reproducing the shipped labels at 99.4%.
3. **Encode** — `encode.py`. One-hot each aligned sequence, position × amino acid, 21 symbols
   including the gap, then drop the parent's own residue column at every position so each
   feature is a substitution indicator. 2,520 features for AT110. Seconds.
4. **Train** — `model.py`. `LogisticRegression(penalty='l1', solver='saga', C=0.5)` on an early
   round's labels. Converges in 127 iterations on AT110. Seconds, CPU.
5. **Predict / score.** Apply the trained model two ways:
   - **Skip a round.** Rank the campaign's own clones by predicted enrichment in a later round,
     so the scientist does not have to run it. Train FACS1/MACS, test FACS2/MACS — not R3, which
     this dataset does not usefully have (section 5).
   - **Score outside clones.** Paste in any other sequences for the same target — a different
     sub-library, or AI-designed candidates — and rank them against what the model learned.
     **This is our distinctive feature**, and it is nearly free once the model exists: scoring is
     just applying the model.
6. **Results** — `pipeline.py`, `app.py`. A ranked table of clone, predicted enrichment and
   predicted class, plus the **per-position substitution preferences** as a heatmap straight
   from the fitted weights. The 2D map is still unbuilt and still optional.

### Decisions baked in

- **Per-campaign, on the fly.** No universal model; each upload trains its own. "Score any clone"
  means any clone in that campaign's target neighborhood. A sequence from a different target is
  out of distribution and not reliable, and we say so plainly rather than returning a confident
  number.
- **The standard input format is the keystone.** Everything speaks one format —
  `{clone id, sequence, per-round counts}`. Onboarding a new or external dataset then means
  writing a small converter into that format, not touching the pipeline. That is what makes the
  tool general, and it is the path to testing on data beyond this paper. **Generality is the
  goal, not just a demo of the paper** — the paper's campaigns are the baseline we prove it on.
- **Alignment is the one real threat to that generality.** "Pre-aligned" is not a format a
  scientist naturally has. The vendored sequences are equal-length because upstream ran every
  campaign through ANARCI numbering against the parent nanobody, and a one-hot model needs that:
  column 47 has to mean the same residue position in every row. A raw per-variant count table
  from somebody else's campaign will be ragged, and padding or trimming it to equal length
  silently scrambles the positions, which is worse than refusing it. So: accept equal-length
  input directly, detect ragged input and say exactly what is wrong, and offer ANARCI numbering
  as an explicit align-for-me step rather than assuming it was done. Upstream's protocol budgets
  ~3.7 min per 10k sequences for ANARCI, so it is affordable at campaign scale. The same
  numbering has to be applied to pasted outside clones, or they land in the wrong columns.
- **The honest claim.** It ranks and classifies clones by predicted selection success — an
  AUC-style binary result. Not "predicts exact R3 counts", and never "predicts binding affinity".

---

## 4. Constraints on how this is described

Not optional framing. These hold in code, column names, README text, UI labels and anything that
reaches a judge.

1. **This isn't novel.** It is an open implementation of Paul et al. (2026). Do not pitch it as
   new. Pitch it as a clean, usable, open version, and put original energy into our own twist:
   the app, scoring outside clones, the cross-target comparison, our own validation.
2. **Enrichment is not affinity.** The model predicts enrichment — what survives selection —
   which is a proxy for binding confounded by expression and stickiness. Claim "predicts
   enrichment", "ranks candidates" or "classifies by predicted selection success", **never**
   "predicts binding affinity". A sharp judge knows the difference. Never name a model output
   `affinity` or `binding` anywhere, including variable names and column headers. Use `enr` /
   "predicted enrichment".
3. **It classifies, it does not forecast counts.** The result is binary and AUC-style. Do not
   claim it predicts a clone's exact later-round count.
4. **Validate across rounds, not within.** Testing on held-out variants from the same round risks
   learning the experiment's quirks. Train on an early round and test on a later one.
5. **Out of distribution is stated, not hidden.** Scoring a sequence from a different target is
   unreliable; the app says so instead of returning a confident-looking score.

Also: **never quote an accuracy number that has not actually been measured.** No placeholder
metrics surviving into the pitch.

---

## 5. The data

Real published data, not simulated. Paul et al. (2026), *Machine learning enables efficient and
effective affinity maturation of nanobodies*, repo
[`debbiemarkslab/ML_affinity_maturation`](https://github.com/debbiemarkslab/ML_affinity_maturation).

**Verified on 2026-10-23 and vendored** at `data/paul2026/`, pinned to upstream commit `96c5b0a`.
The day-one blocker — does the repo ship processed per-round tables, not just code — is
**resolved yes**. No wifi needed to build.

Three campaigns, each a nanobody matured against a different GPCR target:

| Campaign | Target | Rounds | Variants | Aligned width |
|---|---|---|---|---|
| AT110 | AT1R | MACS, FACS1, FACS2 | 5,257 | 126 |
| NbB7 | β2AR | naive, MACS, FACS1, FACS2, FACS3 | 54,102 | 125 |
| RX002 | RXFP1 | MACS, FACS1, FACS2 | 21,507 | 122 |

**[data/DATASET.md](../data/DATASET.md) is the full record** — per-file contents, labeled counts
for all 16 round pairs, and the two things the dataset does not ship. Do not re-derive those
numbers; read that file. The essentials:

- Start with **AT110, train `FACS1/MACS`, test `FACS2/MACS`**. It mirrors upstream's own
  experiment (their directories are named `Train_FACS1_MACS_*/Test_FACS2_MACS_*`, and they ship
  both `Test_scores.csv` and a trained **logistic** parameter file), so our numbers are checkable
  against theirs. RX002 is the balanced second choice at 52% positive.
- **NbB7 looks richest but its labels are degenerate** — its FACS3 pairs are 85–96% positive, so
  a model there scores well by predicting "enriched" for everything.
- **No raw FASTQ ships** (upstream says too large), so an align-and-count front end has no input
  here. This is also why the workflow assumes pre-aligned input.
- **No measured affinities ship in the download**, but they exist in the paper. Their ESM2 script
  references an `AT1R_affinity` dataset not present in the repo. The paper reports measurements
  in two places, both **RX002**: single substitutions measured by biolayer interferometry in
  **Supp Table 4** and Supp Fig 9 (9 of 11 improved, 7 by at least tenfold), and designed
  higher-order variants measured by SPR in **Figure 5A** (58 of 90 measured, 32 failed to
  express, best K_D 0.18 nM with three sub-nanomolar). So validation is **not blocked on
  existence, only on transcription**: pull Supp Table 4 from the preprint and RX002 becomes the
  campaign for a real non-circular check. That the measurements are RX002 is also why RX002, not
  AT110, is the campaign to use if this feature is attempted.

### Mapping the workflow's round names onto this dataset

The workflow says Input / R1 / R2 / R3. In the vendored data that reads: Input → `MACS` (or
`naive`, which only NbB7 has), R1 → `FACS1`, R2 → `FACS2`, R3 → `FACS3`.

**Target the Input → R1 pair, validate on R2.** `FACS1/MACS` for training, `FACS2/MACS` for
testing. That is upstream's own experiment — their directories are literally
`Train_FACS1_MACS_enriched/Test_FACS2_MACS_enriched`.

**Do not build on FACS3. Settled by the paper on 2026-10-24.** Only NbB7 has one and its
`FACS3/*` labels are 85–96% positive. The upstream repo never says what that round selected for,
but the paper does: they performed a third FACS sort that included a **polyreactivity
counter-selection** alongside the affinity selection, using a polyspecificity reagent made from
Sf9 insect cell membranes. So FACS3 is a counter-selection, not a deeper binding round,
**no campaign here has a third binding round**, and "validate on R3" must not be said at all.
Say "a later round" and mean FACS2.

Two follow-ons from that. The paper reports cysteine enrichment driven by polyreactivity in B7,
which FACS3 and a polyreactivity filter were introduced to address, so the degenerate labels are
explained rather than merely odd. And FACS3 screens for **stickiness, not growth**, so it does
not serve as the control arm for the growth-versus-binding extension in question 10.

### The exact enrichment recipe, verified

Read out of upstream's `utils/affinity_maturation_utilities.py` and reproduced against the
vendored tables. Replicate this for uploads rather than inventing a variant:

1. `normct` = counts per million, then **clipped to a floor of 1 CPM**. That floor is the
   pseudocount, and it is why a log ratio exists for every row including clones absent from one
   round.
2. `logct` = `log10(normct)`. Base 10, not natural log — `enr` runs about −3.3 to +3.6, not in
   natural-log units.
3. `logratio` = `logct_late − logct_early`.
4. Binary label: fit `GaussianMixture(n_components=2, random_state=274)` to the log-ratio values,
   keep an assignment only where its posterior exceeds **0.8**, and mark the rest *not confident*
   and drop them. Also drop `logratio == 0` exactly, and require the clone to have reads in **at
   least one** of the two rounds (a union, not an intersection — the overlap-only variant labels
   ~100 rows and is clearly not what shipped).
5. Train/test: `sklearn.train_test_split(test_size=0.2, random_state=42)` over the confidently
   labeled rows. Hence 3,092/774 on AT110 FACS1/MACS.

Reproducing steps 1–4 gives 3,853 labeled rows against the shipped 3,866 at **99.4% agreement**
on AT110, and 16,553 against 16,858 at 99.4% on RX002. Close enough to call the recipe
confirmed; the residual is most likely their length filter or a sklearn version difference.

Two consequences worth holding onto:

- **The GMM abstains, and that flatters the metric.** Dropping the ambiguous middle is part of
  why the published numbers look as good as they do. A zero-threshold label is a defensible
  simplification, but it keeps the hard cases, so it will score worse and cannot be compared
  head-to-head with their `Test_scores.csv`. Pick one and say which.
- **The vendored tables are already filtered.** Clone ids read `AT110/fil5_filabund1_0`:
  raw count ≥ 5 in at least one round, abundance floored at 1 CPM, and sequences shorter than 110
  aa removed. So the drop-low-counts step matters for user uploads, not for reproducing the paper.

---

## 6. Scope and priorities

Jessica's stated priorities: enrichment-ratio ranking and ML enrichment prediction are the core,
one each, first. Antigen-specific models are a natural framing on top. Validation against
measured affinities is the highest-value credibility add if time allows — but see above, it is
currently blocked on data. The align front end is the piece she most wants for the experience and
also the part drifting furthest from the ML core; her own note says drop it entirely if it starts
feeling like a separate project, and the missing FASTQ settles it.

Reframed on 2026-10-24.

**Minimum: demo the paper, working, in an app.** The method running end to end on the vendored
campaigns, live, in a UI a scientist could drive. **Done** — see section 7.

**Stretch: generalize.** Prove the pipeline is not welded to this dataset: an upload path, a run
on data not in the paper, and an alignment layer if the first two land. Ordered in section 7.

The split is this way round because the minimum is a complete, honest deliverable on its own,
and because generality's hard dependency — alignment — is optional rather than blocking.

### The baseline question, settled by measurement

The earlier plan said the model must beat "rank by observed enrichment". **That comparison is
circular and has been dropped.** The binary label is arithmetic on the same counts that
baseline reads, so the counts predict the label almost perfectly without any learning:

| Predictor on AT110 FACS2/MACS test rows | AUC |
|---|---|
| MACS count alone, negated | 0.998 |
| Early-pair log ratio | 0.989 |
| **The paper's own L1 logistic, AT110, as published** | **0.964** |
| **Our L1 logistic, sequence only** | **0.960** |
| Upstream's CNN, sequence only, same rows | 0.938 |
| Substitutions from parent, sequence only | 0.752 |
| Chance | 0.500 |

The mechanism: 88% of *enriched* test rows have a MACS count of exactly 0, against 0% of
*depleted* rows. A clone absent from the input round is floored to 1 CPM, so any later
appearance produces a large positive ratio. "Enriched" therefore means, to a first
approximation, "was not seen in the input round".

**The margin over the fair baseline is campaign-dependent. Measured 2026-10-24:**

| Campaign | model | substitutions from parent | margin | observed counts (circular) |
|---|---|---|---|---|
| AT110 | 0.960 | 0.752 | **+0.208** | 0.989 |
| RX002 | 0.952 | 0.933 | **+0.019** | 0.962 |

On RX002 counting substitutions from the parent gets 0.933 on its own, so the model adds almost
nothing over it. AT110 is the campaign where the learned model clearly earns its place, and it
is the one to lead with for that claim. If RX002 is on screen — and it will be, if the affinity
validation happens — **do not repeat the AT110 margin**; the honest line there is that the model
matches a simple mutation-count heuristic, and its value is scoring unsequenced clones rather
than out-ranking that heuristic. Worth understanding why before the pitch: RX002 variants carry
more substitutions on average (4.9 against AT110's 3.5, max 67 against 35), so mutation load
alone separates its classes better.

Two consequences, both load-bearing for the pitch:

1. **Never claim to beat enrichment ranking.** State the fair comparison, which is sequence-only:
   0.960 against 0.752. The app shows the counts number too, labelled as circular, so the
   weakness is ours to disclose rather than a judge's to find.
2. **This is the argument for the model.** Ranking by counts only works for clones you already
   sequenced. The model needs no counts, so it scores clones the experiment never contained —
   which is exactly the distinctive feature.

**The paper states this confound itself, so stop treating it as our discovery or our weak spot.**
Read on 2026-10-24: the authors write that the enrichment labels are highly bimodal and *mostly
reflect whether the sequence was seen before or after enrichment*. They then check it the same
way we do, against distance baselines, and conclude the models learn affinity-related signal
rather than round membership. Our `auc_nmut` at 0.752 is that control. The paper also notes the
imputed floor abundance of 1e-6 for sequences under the 5-count threshold, which is the same
1 CPM pseudocount already implemented, and that most sequences appear in only one round
(their Supp Table 2).

So the framing for a judge is: this is a known, stated property of the method, here is the
control the authors used, here is our number for that control. That is stronger than volunteering
it as a flaw. The paper additionally flags two confounds we do not currently handle: high
expression clones can dominate FACS2 counts, and polyreactivity drove cysteine enrichment in B7.

### Feature menu, with status

| Feature | Status | Note |
|---|---|---|
| Enrichment-ratio ranking | **done** | `enrich.py`; the honest core, and it reproduces the paper's numbers |
| ML enrichment prediction | **done** | `model.py`; 0.960 AUC on AT110, above their CNN's 0.938 |
| Score outside clones | **done** | our distinctive feature; valid only inside that target's neighbourhood, and the app says so |
| Per-position substitution preferences | **done** | falls straight out of the L1 weights; misses combinations, and passenger mutations still mislead |
| Click-run app | **done** | five tabs, seconds per run, nothing to babysit on stage |
| Upload path | next | the public `{clone id, sequence, counts}` CSV through the same code |
| A dataset outside the paper | next | ProteinGym first, then MaveDB — section 7 |
| Antigen-specific comparison | open | three targets to compare; less data per target, more models to juggle |
| 2D landscape map | open | striking but purely presentational; does not improve the model |
| Developability / polyreactivity filter | open | real-world credibility, mostly rule-based, but a pretrained model already exists |
| Alignment layer | open | the thing that makes uploads general; adds an HMMER dependency |
| Validation vs. measured affinities | **unblocked, needs transcription** | not in the download, but in the paper: RX002 BLI singles in Supp Table 4, SPR designs in Fig 5A. Switch campaign to RX002 for this |
| Align + count from raw FASTQ | **blocked** | no raw FASTQ ships, and input is assumed pre-aligned |
| CNN / novel sequence design | **cut** | logistic regression only; generation is replaced by scoring externally designed candidates |

---

## 7. What is built

The minimum goal is met: the method runs end to end, live, in a Streamlit app.

```
truehit_atg/
  schema.py    handoff contract, campaign facts, library sizes, chk() validator
  config.py    CFG, every tunable
  data.py      load vendored tables into the contract
  enrich.py    CPM floor, log10, ratio, GMM labels, split, baseline AUC
  encode.py    21-symbol one-hot, parent-column drop, IMGT feature names
  model.py     L1 logistic, AUC, ROC, coefficient table
  pipeline.py  run() -> Res; the only thing a UI calls
app.py         Streamlit, thin: no UI import exists anywhere in the package
data/
  paul2026/    AT110, NbB7, RX002
  DATASET.md   the data record
tests/         43 tests
```

Setup is `python -m venv .venv`, `.venv/Scripts/pip install -r requirements.txt`, then
`.venv/Scripts/python -m pytest -q` and `.venv/Scripts/python -m streamlit run app.py` from
this directory. `pythonpath = .` in `pytest.ini` is relative to here, so run from here.

A full AT110 run is a few seconds on CPU: the solver converges in 127 iterations. The app is
five tabs — enrichment with the GMM split drawn over the log-ratio histogram, model metrics
against both baselines, ranked candidates, the substitution-preference heatmap, and a box for
scoring sequences the campaign never contained.

### Reproduction, measured

| Check | Result |
|---|---|
| Normalized counts vs shipped `norm_*` | exact, once the pre-filter library sizes in `schema.LIBS` are used |
| `hasread` | exactly `count >= 5`, on every round of every campaign |
| Log ratios vs shipped | max error 7e-16 |
| GMM labels vs shipped | 99.4% agreement, 3,853 labelled against 3,866 |
| Coefficients vs upstream's shipped logistic params | 101 of ~132 shared, **sign agreement 1.000, Pearson r 0.983** |
| Strict AUC, rows never seen in training | 0.958 over 384 rows, so the split overlap was not inflating anything |

Against every figure the paper publishes for this model, measured 2026-10-24:

| Campaign | Train -> test | ours ROC | paper ROC | ours PR | paper PR |
|---|---|---|---|---|---|
| AT110 | FACS1/MACS -> FACS2/MACS | 0.960 | 0.964 | 0.942 | 0.949 |
| RX002 | FACS1/MACS -> FACS2/MACS | 0.952 | 0.935 | 0.952 | 0.947 |

All four reproduce within 0.017, and the suite asserts all four.

A number landing above the published one is not evidence of a better model — it is a
methodological difference, most likely our 99.4% label agreement or a sklearn version. Do not
present it as anything else; section 4 applies. Upstream's CNN sits at 0.938 on the AT110 rows,
a side reference rather than the comparison.

No logistic `Test_scores.csv` ships upstream — only CNN and MLP ones — so the coefficient diff
is the comparison for our model, and their CNN score file is the separate reference point.

**The paper's Table 1 is now the headline comparison, not the CNN.** It reports L1 logistic at
ROC-AUC 0.964 and PR-AUC 0.949 on AT110, and 0.935 / 0.947 on RX002. Our 0.960 reproduces the
published AT110 figure to within 0.004, which is a far stronger claim than beating their CNN, and
RX002 gives a second checkable number worth running. The paper also compared linear regression,
a CNN, an **ESM2-MLP**, and two distance baselines (distance to WT and median MACS distance).
Their Spearman figures on the regression task are CNN 0.695 on AT110 and 0.744 on B7, and linear
regression 0.709 on RX002.

**The ESM2-MLP performed sub-optimally in their hands.** That closes the question of swapping
one-hot for protein language model embeddings: it has been tried on this exact data and it did
not help. Note this does not touch using a PLM to *propose* candidate sequences that our model
then scores, which is a different use and still open.

### The parent carries no gaps in model space

Checked 2026-10-24, and it corrects a premise in `plan-ai-features.md` B1. That plan quotes
`wt_seq('AT110')` as `QVQLQESGG-GLVQAGG...` and concludes gaps sit in the middle of the
sequence the model sees. They do not. `wt_seq()` returns the **135-column full ANARCI
alignment**; the model's feature space is `encode.wt_cols()`, the 126 retained columns, and the
parent has **zero gaps** there in all three campaigns.

Variants do carry gaps — 3.1% of AT110, 56.8% of NbB7, 0.8% of RX002 — so the encoder's 21st
symbol is earning its place. But for candidates built by substituting into the parent,
`gap_map` is the identity and `place` is a plain splice, so the position-shifting hazard that
plan feared does not arise on that path. Both functions are still implemented for the general
case, since an uploaded campaign's parent may carry a gap in a retained column.

`encode.place()` validates its residue against a **set**, not the alphabet string: `"VW"` and
`""` are both substrings of `"ACDEFGHIKLMNPQRSTVWY"`, so string membership would have accepted
them and silently changed the sequence width. A test covers it.

### The handoff contract

`schema.py`, one row per variant:

| Column | Meaning |
|---|---|
| `var_id` | upstream `seq_ID` |
| `aa` | protein sequence, gaps stripped |
| `aa_aln` | aligned sequence, fixed width per campaign, one-hots directly |
| `cdr3` | CDR3 |
| `cnt_<round>` | raw read count, present for every round in the campaign |
| `norm_<round>` | normalized count, counts per million, from upstream |
| `enr` | enrichment: log ratio of the chosen round pair |
| `enr_bin` | binary enriched label, NaN where upstream made no call |
| `split` | upstream's train/test assignment, NaN where unlabeled |

This is the internal table. The **upload format** the workflow specifies is the smaller public
triple — `{clone id, sequence, per-round counts}` — and `data.py` is the first converter into the
internal table from it. A second dataset means a second converter, not a second pipeline.

Things that bite:

- **`enr` is defined for every row; `enr_bin` and `split` are not.** 3,866 of 5,257 AT110 rows
  are labeled for FACS1/MACS. The binary model trains on the labeled subset; the baseline ranking
  can use every row. Use `data.labeled(df)` then `data.split(lab, "train")`.
- **Count columns come for every round**, not just the requested pair. Only `enr`, `enr_bin` and
  `split` change when the pair changes.
- **`aa_aln` gaps are `-`**, so the one-hot encoder needs a 21st symbol. `wt_seq('AT110')` starts
  `QVQLQESGG-GLVQAGGSLRLSCAASGNIF----DADIMG`.
- **`schema.pairs(camp)`** enumerates valid `(late, early)` pairs; `load()` raises on one the
  campaign does not have. **`schema.chk()`** raises on missing columns or duplicate `var_id` and
  returns the frame — call it on anything handed between layers.
- **`enr` stays even though the model is binary.** It is the honest baseline ranking and the
  source of `enr_bin`. Dropping it would remove the thing the model has to beat.

### Config

`CFG` in `config.py`: `camp="AT110"`, `train=("FACS1","MACS")`, `test=("FACS2","MACS")`,
`min_ct=0`, `min_ct_rnd=""`, `norm="cpm"`, `seq_col="aa_aln"`, `seed=0`, `out="data/out"`.

Keep count thresholds, normalization choice, campaign and round pairs here as configuration, not
as constants inline in analysis code. The L1 penalty strength belongs here too once the model
lands. `norm` is declared but unused so far — upstream's `norm_*` columns are already counts per
million.

### Stretch, in order, each independently droppable

1. **Upload path.** Accept the public format `{clone id, sequence, count per round}` as CSV and
   send it down the same `enrich → encode → model` path. The equal-length guardrail is already
   in `chk()` and refuses ragged input by name rather than padding it. This is why recomputing
   from counts mattered: the upload path is the code that already runs.
2. **A dataset not in the paper.** Two tiers, cheapest first.
   - *Tier A, model layer only:* [ProteinGym](https://github.com/OATML-Markslab/ProteinGym).
     Its DMS substitution files carry `mutant`, `mutated_sequence`, `DMS_score` and
     `DMS_score_bin` — fixed-length and position-matched by construction, with a binary label
     already provided — so it drops into `encode → model` with no counts and no alignment, as
     one converter function. 217 assays over 186 proteins, MIT, from
     [proteingym.org/download](https://proteingym.org/download). Same lab as the paper. First
     action: pull the substitutions reference file and filter for an antibody, VHH or nanobody
     assay so the molecule class matches.
   - *Tier B, the full pipeline:* [MaveDB](https://www.mavedb.org). Score sets can carry
     per-variant **count** columns and there are antibody-binding sets; REST API at
     `api.mavedb.org/api/v1`. Riskier, because counts are optional per submitter.
   - Criteria for any candidate: one row per variant; counts for ≥ 2 rounds or conditions
     (Tier B only); all sequences equal length and position-matched; protein, not DNA.
3. **Alignment layer.** ANARCI numbering so ragged uploads become usable. ~3.7 min per 10k
   sequences by upstream's protocol, but it adds a conda-installable dependency (HMMER).
4. **React frontend** over the same `pipeline.run()`. The core holds no UI import, so this is
   additive.

Also still open: antigen-specific comparison across all three campaigns, the 2D landscape map,
and validation against measured affinities (blocked on data).

Stack stays small: pandas and numpy for the tables, scikit-learn for the logistic regression and
AUC, matplotlib for the figures, streamlit for the app. **No PyTorch** — the CNN is cut.

---

## 8. The demo

Runs live in the app, AT110 by default. Numbers below are measured, not placeholders.

1. "Here's sequencing data from a nanobody being improved over rounds of selection." 5,257
   variants, three rounds.
2. **Enrichment tab.** The log-ratio histogram splits into two humps — the losers around −2.5
   and the winners around +2. That picture is the whole concept in one image, and it is why the
   task is binary.
3. "Now we train on just the early round and predict which variants enrich later, without
   running the full campaign." Press the button; it finishes in seconds on a laptop.
4. **Model tab.** 0.960 AUC against the held-out later round, where counting substitutions from
   the parent gets 0.752. Mention that the paper's own CNN gets 0.938 on the same rows.
5. **The distinctive beat.** Paste in clones the campaign never contained and rank those too,
   against a model that only ever saw this target. The counts cannot do this: an unsequenced
   clone has no enrichment to rank by.
6. Close: "This is how you cut an affinity-maturation campaign from many rounds to one. Less
   time, less reagent."

If a judge asks why ranking by the observed counts scores higher, the answer is ready and is
in the app: the label is arithmetic on those counts, so that comparison is circular — see
section 6. Volunteering it is stronger than being caught by it.

Historically no sequencing, genomics or protein project has won Cal Hacks overall, and winners
are things judges can watch working live — so the explanation has to be extremely simple. The
seconds-on-CPU runtime is what makes beat 3 live rather than a replay.

---

## 9. Open questions

Settle with Jessica. Several have moved since the proposal; the resolved ones are marked.

1. **Framing of the growth-versus-binding flag.** On the 23 Oct call, Pratham and Jessica agreed
   to add a flag identifying clones that rise across rounds because they replicate faster rather
   than because they bind. The proposal predates that call and does not contain it as its core —
   it touches the confound only in the problem statement and in the developability filter. Two
   readings: **(a)** TrueHit ATG is enrichment prediction as proposed and the flag is one feature
   inside it, most likely an extension of the developability filter or a confound-aware model
   term; **(b)** TrueHit ATG *is* the flag and the proposal is a separate third idea. **(a) is
   the safer default and the likely intent** — it has real data and a defined MVP behind it.
   Build what the proposal specifies; treat the flag as a planned extension, not the foundation.
   Confirm before building on it.
2. Which campaign to lead the demo with. **Built against AT110** and it is the app's default,
   because it is the one upstream ships trained parameters for. All three load and run; RX002
   is the balanced alternative. Still worth her view before the pitch.
3. Which round pair to train and test on. **Settled: train FACS1/MACS, test FACS2/MACS**, which
   is upstream's own experiment. FACS3 is out — see section 5.
4. **What FACS3 actually selected for in NbB7. Resolved 2026-10-24 from the paper:** a
   polyreactivity counter-selection run alongside the affinity selection, using a polyspecificity
   reagent made from Sf9 insect cell membranes. So it is not a third binding round, "R3" stays
   out of the vocabulary, and the degenerate labels have an explanation. See section 5.
5. **What counts as an ultra-low count** for the drop step. **Mostly settled by reading their
   code:** `hasread` is exactly `count >= 5` per round, confirmed against every round of every
   campaign, and the abundance floor is 1 CPM. Both are in `CFG` as `fil_cnt` and `cpm_floor`.
   `min_ct` stays off by default because the vendored tables already went through that filter.
   The open part is only what to default to for an upload.
6. **GMM labels or a zero threshold.** **Settled: GMM**, implemented and reproducing the
   shipped labels at 99.4%. Worth telling Jessica that it abstains on ~1,400 of 5,257 AT110
   variants, since that is a real modelling choice and not an implementation detail.
7. **How much alignment work the app takes on.** Refusing ragged input is built: `chk()` names
   the offending rows. Whether to go further and ship ANARCI numbering is the open half, and it
   is what decides whether the tool is general or only runs on data shaped like this paper's.
8. Whether the align-and-count front end from raw FASTQ is in scope. **Resolved: no.** No raw
   FASTQ ships, and aligned per-variant counts are the input.
9. Whether the CNN is cut. **Resolved: cut for good**, per Pratham on 2026-10-23. Logistic
   regression only.
10. Whether a no-target or counter-selection arm can carry the growth-versus-binding extension.
    **Narrowed by question 4: NbB7's FACS3 will not serve.** It is a polyreactivity
    counter-selection, so it screens stickiness rather than growth rate, and nothing in this
    dataset isolates replication advantage. If the extension is wanted, it needs either a
    different dataset or an explicit model term, not a control arm from here.
11. **Where the outside clones for the demo's distinctive beat come from.** The feature works,
    but it is currently demonstrated on the parent sequence, which is not a compelling example.
    Options: a held-out slice of the same campaign, known optimizing substitutions from the
    paper, or variants we construct from the model's own top-weighted substitutions. **This is
    the highest-value unanswered question for the pitch** — the beat is only as good as the
    sequences we paste in.
12. **What counts as the MVP being done.** **Answered by building it:** the paper runs end to
    end in an app, reproduced against their coefficients. The remaining scope is all stretch.
13. **The label confound. Largely resolved 2026-10-24 from the paper.** The authors state the
    labels are highly bimodal and mostly reflect whether a sequence was seen before or after
    enrichment, and they control for it against distance baselines, concluding the models learn
    affinity-related signal rather than round membership. Our `auc_nmut` at 0.752 is that
    control. Still worth Jessica's read on the biology of "absent from input, present later",
    and on the two confounds the paper raises that we do not handle: high expression clones
    dominating FACS2 counts, and polyreactivity-driven cysteine enrichment. See section 6.
14. Who owns integration between the two workstreams — currently nobody, and `true-hit-qc/` is
    parked.

---

## 10. Sources

- The paper itself, Paul et al. (2026), *Machine learning enables efficient and effective
  affinity maturation of nanobodies*:
  https://www.biorxiv.org/content/10.64898/2026.01.11.698911.full.pdf
  Read on 2026-10-24 for FACS3, the label confound, published AUCs and the measured affinities.
  Table 1 has the model comparison; Supp Table 4 and Figure 5A have the affinity measurements.
- Dataset and reference code for that paper:
  https://github.com/debbiemarkslab/ML_affinity_maturation
  Note: this repo does not `git checkout` on Windows — it contains a file named
  `Test ROC: FACS2_MACS_logratio.pdf` and the colon is illegal. Read its tree from git objects,
  or clone on a POSIX filesystem.
- Jessica's proposal, "Cal Hacks Proposal: ML-Guided Nanobody Affinity Maturation", 20 Oct 2026:
  https://claude.ai/code/artifact/002ddc1a-6e10-4a3e-99d4-aeaf38275550
- Phage display false positives and propagation advantage (background for the
  growth-versus-binding extension): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6259164/
- AbLang: https://pmc.ncbi.nlm.nih.gov/articles/PMC9710568
- AntiBERTy: https://arxiv.org/abs/2112.07782v1
- Repo-level context covering both ideas: `../../claude-docs/context.md`
