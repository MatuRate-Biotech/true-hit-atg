# MatuRate — working context

Scope: the repo root. This is the working context for the MatuRate project.

GitHub: `MatuRate-Biotech/MatuRate` (repo rename pending transfer of ownership from `heller1729`).
Local directory: `maturate/`. Python package: `maturate/`.

Origin: Cal Hacks 13.0, Palace of Fine Arts, San Francisco, 2026-10-23 to 2026-10-25.
Now continuing as an open-source project post-hackathon.

---

## 1. Who

- **Pratham Jangra** (GitHub `heller1729`). MS CS thesis student at Virginia Tech, GRA on the
  NSF-funded CyberSMART project. Backend and software, near-zero hardware experience. Owns the
  ML layer: sequence encoding, training, evaluation, and the app that returns ranked candidates.
- **Jessica Pan** (Slack "Percabeth"). Antibody and molecular biology background, authored the
  proposal this project implements. Owns the data and enrichment layer: assembling per-round
  data, enrichment ratios, count thresholds and normalization, and the alignment layer
  (active next priority post-hackathon).

Two other teammates dropped out on Oct 23. The hackathon was a 2-person team; the project
continues post-hackathon with Jessica as the primary developer.

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
   **Reconciled with the default on 2026-10-24:** the app now *defaults* to within-pair, because
   a real user has an Input round and one Output round and nothing later. That does not weaken
   this constraint; it is why the Model tab shows a **reduced** set of statistics in that
   configuration. Fewer claims, because fewer are supportable. The published comparison, PR-AUC,
   strict AUC and the observed-counts baseline appear only when a different test pair is chosen,
   and the reproduction tests pin the cross-round pair so they keep measuring it.
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

**[data/DATASET.md](data/DATASET.md) is the full record** — per-file contents, labeled counts
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

On RX002 counting substitutions from the parent gets 0.933 on its own, so on the aggregate the
model adds almost nothing over it. RX002 variants carry more substitutions (mean 4.9 against
AT110's 3.5, max 67 against 35), so mutation load alone separates its classes well.

**Do not defend this by saying the model's value is scoring unsequenced clones.** That answers
the *counts* baseline, which needs a clone to have been sequenced. It does not answer
substitutions-from-parent, which is sequence-only and computable for any pasted sequence. A
judge gets there in one question.

### The within-load comparison, which is the real answer

Mutation count cannot order two variants that carry the same number of substitutions — two
different single substitutions are both n=1 — so **within a fixed load it scores exactly 0.500
by construction.** The model has no such limit. Measured 2026-10-24, `pipeline.Res.auc_within`:

| Campaign | aggregate model | aggregate heuristic | **within-load model** | within-load heuristic |
|---|---|---|---|---|
| AT110 | 0.960 | 0.752 | **0.945** | 0.500 |
| RX002 | 0.952 | 0.933 | **0.785** | 0.500 |

RX002's 0.933 is *entirely* mutation load. Strip the load and the heuristic has no resolution
left, while the model keeps 0.785. **This is the uncontaminated comparison and it holds on both
campaigns**, which is a stronger claim than either aggregate and the one to make under
questioning.

*Method:* pair-weighted conditional AUC — AUC within each stratum, weighted by
`n_pos * n_neg`. `model.cond_auc()`. The tempting alternative of pooling
percentile-ranks-within-stratum gives 0.853 and 0.596 and is **biased low**, because
cross-stratum pairs contribute 0.5 by construction. Do not quote those.

### Where it does not hold: single substitutions

| Campaign | single-mutant test rows | of which enriched | stratum AUC |
|---|---|---|---|
| AT110 | 99 | 3 (3.0%) | 0.403 |
| RX002 | 166 | 2 (1.2%) | 0.613 |

Across all labelled rows it is 22 of 442 and 11 of 760. At those positive counts both AUCs are
noise, and one is below chance. **We cannot demonstrate single-substitution skill on this
data in either campaign, and must not claim it.**

This matters twice over. It is Feature B's regime — that feature proposes single substitutions
and ranks them with a model whose skill there is unmeasured, so its tab must say so. And it is
where the paper's RX002 BLI measurements live, since those are single substitutions. So the BLI
data is not merely a nice validation: it is the only thing that could test this regime at all.
That makes it the honest next step rather than a gap to hide.

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
| Upload path | done | the public `{clone id, sequence, counts}` CSV through the same code, into every tab |
| A dataset outside the paper | next | ProteinGym first, then MaveDB — section 7 |
| Antigen-specific comparison | open | three targets to compare; less data per target, more models to juggle |
| 2D landscape map | open | striking but purely presentational; does not improve the model |
| Developability / polyreactivity filter | open | real-world credibility, mostly rule-based, but a pretrained model already exists |
| Alignment layer | **active, next priority** | the thing that makes uploads general; ANARCI (antibody-specific) + MAFFT (general) planned |
| Validation vs. measured affinities | **unblocked, needs transcription** | not in the download, but in the paper: RX002 BLI singles in Supp Table 4, SPR designs in Fig 5A. Switch campaign to RX002 for this |
| Align + count from raw FASTQ | **blocked** | no raw FASTQ ships, and input is assumed pre-aligned |
| CNN / novel sequence design | **cut** | logistic regression only; generation is replaced by scoring externally designed candidates |

---

## 7. What is built

The minimum goal is met: the method runs end to end, live, in a Streamlit app.

```
maturate/
  schema.py    handoff contract, campaign facts, library sizes, chk() validator
  config.py    CFG, every tunable
  data.py      load vendored tables into the contract
  enrich.py    CPM floor, log10, ratio, GMM labels, split, baseline AUC
  encode.py    21-symbol one-hot, parent-column drop, IMGT feature names
  model.py     L1 logistic, AUC, PR-AUC, conditional AUC, precision-at-K, coefficients
  ingest.py    an uploaded CSV, scored or with counts, into the contract
  agent.py     a model reads an unknown file's columns and writes the converter
  pipeline.py  run() -> Res and run_upload() -> UpRes(Res); the only things a UI calls;
               with_sub() puts substitution notation on the coefficient frame
app.py         Streamlit, thin: no UI import exists anywhere in the package
data/
  paul2026/    AT110, NbB7, RX002
  DATASET.md   the data record
tests/         142 tests
```

Setup is `python3 -m venv .venv`, `.venv/bin/pip install -r requirements.txt`, then
`.venv/bin/python -m pytest -q` and `.venv/bin/python -m streamlit run app.py` from
the repo root. On Windows use `.venv\Scripts\` instead of `.venv/bin/`.
`pythonpath = .` in `pytest.ini` is relative to the repo root.

A full AT110 run is a few seconds on CPU: the solver converges in 127 iterations. The app is
six tabs — enrichment, model, ranked candidates, the substitution-preference heatmap, a box for
scoring sequences the dataset never contained, and onboarding somebody else's dataset. An
onboarded file appears in the sidebar's dataset list beside the three campaigns and renders in
the five analysis tabs through the same code; an 18k-row upload takes about a minute to fit.

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

### A two-round upload cannot be validated, and the app must say so

Measured 2026-10-24 by running the upload path end to end. Training on Input/R1 and testing on
the same pair's held-out rows is a **within-pair** split, which section 4 constraint 4 says not
to trust. Cross-round validation needs a third round:

| Upload | What it can evaluate | AUC |
|---|---|---|
| Input + R1 | within-pair only | 0.915 |
| Input + R1 + R2 | cross-round, the honest one | 0.980 |

The workflow in section 3 takes Input and R1. **So the headline accuracy number is not
available on the input the app is specified to accept.** The Onboard tab needs two modes: with
a third round, report the cross-round AUC; with two, give the model and the ranking and say
plainly that no honest accuracy number is available from this upload. Reporting the within-pair
0.915 as if it were the cross-round figure would be the exact error the constraint forbids.

Related, measured the same day: the split seed moves test AUC very little — over 8 seeds, AT110
is 0.956 +/- 0.006 (range 0.945-0.965) and RX002 is 0.951 +/- 0.004 (0.944-0.958). The
published 0.964 and 0.935 both sit inside those ranges. So the reproduction claim is not a
lucky draw, but quoting our figures to three decimals overstates their precision; two is
honest.

### Outside data: what generalises and what does not

Run 2026-10-24 against ProteinGym (`OATML-Markslab/ProteinGym_v1`, DMS substitutions shard 0).
**ProteinGym has no antibody assay** — none of its 217 is a nanobody, VHH or antibody variable
domain — so scoping an outside test to assays matching a vendored alignment would have selected
an empty set. Uploads supply their own parent instead, and get plain `1..L` positions because
IMGT numbering is antibody-specific.

| Assay | Shape | Result |
|---|---|---|
| `HIS7_YEAST_Pokusaeva_2019` | 20k variants, 220 aa, **7.0 substitutions each** | **ROC-AUC 0.959** at the paper's own penalty (0.973 at lam 0.1) |
| `PAI1_HUMAN_Huttinger_2021` | 5,345 variants, 402 aa, **all single mutants** | **ROC-AUC 0.500**, at every penalty |

**The pipeline is general; the model class needs substitutions that recur.** HIS7 is a yeast
enzyme scored for growth — nothing to do with antibodies — and the same code reaches 0.959 on
it. PAI-1 fails for a reason that is arithmetic, not data quality: a systematic single-mutant
scan contains each substitution exactly once, so a substitution held out to test was **by
construction** never in training. Measured: 1,069 of 1,069 PAI-1 test rows had no trained
feature, every prediction collapsed to the intercept. AT110 escapes this because its variants
carry 3.18 substitutions that recur — only 1 of 771 test rows has no trained feature.

Both assays are pre-scored, so **the enrichment code did not run on either**. Outside data has
exercised `encode -> model` only, and the app says so. If a judge asks whether enrichment has
been tested on anything but Paul et al., the answer is no.

### Property features, and why Feature B needed them

The same arithmetic broke Feature B, which proposes *novel* single substitutions. Measured on
AT110: of 1,606 possible novel single substitutions, the one-hot model gives them **1 distinct
score** — every one tied on the intercept, spread 0.0000. A ranking of them would have been
meaningless.

`CFG.props` adds one column per position holding the BLOSUM62 score of parent to variant,
loaded from biopython rather than transcribed. Measured against criteria fixed before the run:

| Criterion | Result |
|---|---|
| AT110 cross-round unchanged | **pass** — 0.960 to 0.956, delta -0.003 |
| Novel substitutions separate | **pass** — 1 distinct score to **93**, spread 0.318. At position 4, L->D (BLOSUM -4) scores 0.036 against L->I (+2) at 0.169 |
| PAI-1 rises clearly above chance | **fail** — 0.500 to 0.515, not clearly above |

So Feature B is viable again and PAI-1 stays out of reach. No penalty sweep was run to chase
the third criterion; that was agreed in advance, and it remains the obvious thing to try if
anyone revisits it.

**`props` defaults to off.** The published reproduction depends on the paper's pure one-hot
encoding, and the reproduction tests are the tripwire that this stayed additive.

### The product reframe, 2026-10-24

The framing moved from "a reproduction of Paul et al." to "a tool for predicting enrichment".
The paper reproduction is still there and still tested, but it is no longer the headline.

**The default is now within-pair.** `CFG.test` equals `CFG.train`, both `FACS1`/`MACS`, because
that is the data a real user has. Measured on AT110 in that configuration:

| | within-pair default | cross-round |
|---|---|---|
| Test ROC-AUC | 0.922 | 0.960 |
| Model within fixed load | 0.907 | 0.945 |
| Substitutions from parent | 0.564 | 0.752 |
| Held-out variants | 771 | 852 |

Three consequences worth holding onto:

- **Strict AUC stops being a distinct number.** It is the AUC over test rows absent from
  training; with both drawn from one pair's split those sets are disjoint by construction, so it
  equals the test AUC exactly. It stays in the cross-round view, where it still means something.
- **The reproduction tests are pinned.** `tests/test_pipeline.py` defines `CROSS` and passes it
  wherever a published or previously measured figure is asserted, so moving the app default
  cannot quietly move those numbers.
- **The `res` fixture is deliberately unpinned**, so the default configuration stays covered. Its
  shape assertion checks the 80/20 split rather than a row count tied to one pair.

### Precision at K, which reads better than AUC for this product

`Res.pk` and `Res.base`, from `model.prec_at_k`. On AT110 at the default, with a base rate of
28%:

| K | 10 | 50 | 100 | 200 | 300 | 500 | 771 |
|---|---|---|---|---|---|---|---|
| share enriched | 100% | 98% | 94% | 87% | 64% | 41% | 28% |

A 3.5x lift over base at K=50. "Of the top 50 the tool hands you, 49 enriched, against 28% at
random" is a statement about a decision someone actually makes, which an AUC is not.

### Round names are positional everywhere in the UI

`schema.rnd_lab(camp, rnd)` and `schema.pair_lab(camp, pair)`: index 0 of `CAMPS[camp].rnds` is
**Input**, index i is **Output i**. The first argument takes a campaign name *or* an explicit
round order, so an upload's own `cnt_*` columns go through the same rule.

The assay names survive in exactly one place, a provenance line on the Enrichment tab: "Input is
the MACS round; Output 1 is FACS1; Output 2 is FACS2." Everywhere else — both sidebar dropdowns,
headings, table headers, plot axes — uses the generic label. Note the rule is positional, so
`MACS` is **Input** in AT110 and RX002 but **Output 1** in NbB7, which has a `naive` round first.

### What the Enrichment tab now states, and the numbers behind it

The tab explains the arithmetic in plain language rather than naming it: counts per million
against the pre-filter library size, the 1 CPM floor, log10, later minus earlier, then the
two-component fit. `enrich.mix_params()` exposes the fitted parameters through `Res.mix`, from
`enrich.mix_fit()` — the same fit the labels come from, so the quoted humps cannot drift from
the labelling. On AT110 the humps sit at −2.08 and +1.96 with 70% of the fit in the lower one.

**The abstained figure was being described wrongly** and is now corrected. Measured on AT110:

| Reason | Rows |
|---|---|
| Enrichment exactly 0, never scored (zero reads in both rounds) | 1,190 |
| Below the 5-read floor in both rounds | 201 |
| Scored, below 80% confidence | 13 |

The old caption attributed all 1,404 to the last row, which is 13 of them. The tab now says
"variants with too few reads to call, or whose enrichment sits between the two humps" — both
reasons, neither claimed as the only one.

### The union-table leak, and the row filter, 2026-10-24

The vendored CSVs hold **one row per variant across every round** — the union. `data.load()`
took a pair but used it only to choose which log-ratio and label columns to read, never to
filter rows. So the AT110 FACS1/MACS view returned all 5,257 variants, 1,190 of which had zero
reads in *both* selected rounds and existed only because FACS2 was sequenced.

Those rows were harmless to every metric and ruinous to the ranking. Their log ratio is exactly
0 (both rounds floored to 1 CPM), the mixture excludes zero before labelling, so they never
entered training or evaluation — but `run()` scores every row in the table, and they took
**81 of the top 100 candidates**, 45 of the top 50, 10 of the top 10.

`data.load()` now keeps only variants with at least one read in either round of the selected
pair, gated on that pair rather than a fixed one: a variant seen only in FACS2 is real data for
a FACS2 pair. AT110 FACS1/MACS goes 5,257 -> 4,067 rows and 1,404 -> 214 unlabelled (201 under
the 5-read floor, 13 below 80% confidence). Labelled stays 3,853. After the filter, 0 of the
top 100 come from that group.

**Do not use `CFG.min_ct` for this.** Filtering on input count instead would delete 852 clones
with zero input reads but real FACS1 reads, 750 of them labelled enriched — the strongest
risers in the dataset. The two controls are different things.

**Accuracy is untouched, verified rather than assumed.** Running the same pipeline with the
filter on and off: AT110 within-pair 0.9216 both ways, RX002 0.9187 both ways, with identical
train and test counts. No dropped row has a non-zero log ratio in either campaign, so none could
ever have been labelled.

#### `run()` loads once per pair

The filter exposed a latent bug. `run()` loaded the table once, gated on the **train** pair, and
then computed the test pair's frame from those same rows — judging the test pair on the train
pair's row set. Before the filter that was invisible, because no rows were ever dropped. After
it, the cross-round view lost 1,213 labelled test rows on AT110 and 3,858 on RX002, and the
published comparison fell to 0.908 and 0.820.

`run()` now calls `data.load()` once per pair and reuses the frame only when the pairs are
equal. Because `a` and `b` are no longer the same row set, the counts baseline joins them on
`var_id` instead of position. Cross-round figures are back to 0.960 / 0.942 on AT110 and
0.952 / 0.952 on RX002.

The upload path never had this problem: `run_upload()` does no pair-gated filtering, so a
two-round and a three-round export of the same campaign already behaved correctly, giving 0.915
within-pair and 0.980 cross-round. The filter is what derives that same per-pair view from a
union table.

### An upload renders in every tab

Three changes, in order, each depending on the one before.

#### `run_upload()` returns a `Res`

It used to return a smaller `UpRes` carrying only what the Onboard tab drew, so the other five
tabs could not read it. `UpRes` is now a **subclass** of `Res` with six extra fields
(`mode`, `rnds`, `labels`, `spread`, `weak`, `head`) and an `n` property. A subclass rather than
optional fields on `Res`, so `Res` stays the single thing the tabs read and no vendored run
carries six fields that are always `None`.

`run_upload()` now fills every `Res` field. `auc_strict` is real for a cross-round upload and
`nan`/`0` for a within-pair one: with one round pair, train and test are disjoint by
construction, so a strict figure would only restate `auc_te`. `Res.auc`/`Res.pr` were renamed
`auc_te`/`pr_te` so both paths name the same thing. `run()`'s behaviour is unchanged.

#### The agent is a fallback, not a requirement

The Onboard tab used to call `agent.onboard()` unconditionally, which needs `ANTHROPIC_API_KEY`.
`ingest.identity_spec(cols)` now builds a `Spec` from the contract's own column names — round
order from column order — and the tab tries that first. A file that already uses `var_id`,
`aa_aln` and `cnt_*` is read with no model call at all; only a `ValueError` offers the
"Ask the model to map it" button. The tab says which path was taken rather than leaving it
implicit.

A stored mapping is dropped when the uploader clears or the file changes, or a second upload
renders the first file's result underneath its own error.

#### One `Dset` record, so the tabs stop knowing about campaigns

Six places in `app.py` read `s.CAMPS[camp]`, which raises for an upload. A `Dset` dataclass now
carries the four facts the tabs need — display name, round order, aligned width, target, and
whether positions are IMGT — built from `s.CAMPS[camp]` for a campaign and from the stored
mapping for an upload. `s.CAMPS[camp]` is read in exactly one place. Everything downstream reads
`ds`, which is what made the rest string substitution.

- **Sidebar.** An onboarded file joins the vendored campaigns as **Uploaded file**; picking it
  and pressing Run pipeline renders it in all five analysis tabs. The sidebar draws before the
  Onboard tab, so a newly accepted mapping triggers one `st.rerun()`, guarded on the mapping
  actually changing.
- **Round pairs.** Not selectable for an upload — `run_upload()` derives them from round order —
  so the pair is printed rather than offered.
- **Count filter.** Hidden for an upload: `run_upload()` has no such parameter, and a control
  that does nothing is a lie.
- **Paper comparison.** Explicitly `None` for an upload. It was already `None` by accident via
  the key tuple; now it says so.
- **Heatmap axis.** "IMGT position", or "position in the parent sequence" when the upload brought
  its own parent.
- **Out-of-distribution warning.** An upload has no target name, so it is worded against the
  dataset: "The model learned the selection in X only. A sequence from a different parent, target
  or assay is out of distribution."

A **pre-scored** upload stays on the Onboard tab. It has no rounds and no count columns, so the
round-pair and enrichment tabs have nothing to draw; it is excluded from the sidebar list with a
caption saying why, rather than crashing.

#### The worked example was hardcoded to AT110

The enrichment explainer stated "148 reads ... becomes 1,087.6 and 3,205.5 counts per million
... +0.47 — a 2.9-fold rise". Those CPMs were computed against AT110's pre-filter library and
were already wrong for NbB7 and RX002. It is now computed from whichever dataset is loaded.
AT110 reproduces the old text character for character.

#### Verified on a dataset the app was not built around

RX002 reshaped as a two-round upload with its own 122-position parent in the box, measured
2026-10-24. The app reproduces a direct `run_upload()` call exactly:

| | direct call | the app |
|---|---|---|
| rows | 18,179 | 18,179 |
| train / test | 13,345 / 3,337 | 13,345 / 3,337 |
| test ROC-AUC | 0.930 | 0.930 |
| non-zero features | 464 of 2,440 | 464 of 2,440 |

All five tabs populate: histogram with both humps, the same-pair metric set, the ranked table,
the heatmap on 1..L positions, and the scoring box (parent 0.0835 depleted, a 24E/27H/78E mutant
0.9999 enriched). Switching back to AT110 cross-round still gives 0.960 / 0.942 / 0.945 with the
paper line, unchanged.

**This upload is not a cross-check of the vendored RX002.** An upload cannot know the pre-filter
library size, so CPM is taken against the summed counts — 272,860 and 261,609 — where
`schema.LIBS["RX002"]` holds 922,141 and 554,585. Different denominators give different
enrichment, a different mixture cut and different labels. The two numbers measure different
things.

### The Substitution preferences tab says what its columns mean

The table beside the heatmap rendered `model.coefs()` straight, so its headers were the internal
field names `feat`, `beta`, `imgt`, `aa`, none of which the page defined. `feat` also carried
nothing the other two did not: it is `imgt` and `aa` joined by an underscore (`66_D`).

#### Standard substitution notation

`pipeline.with_sub(cf, imgt, wt)` adds a `sub` column — parent residue, position, substituted
residue, so `66_D` becomes `N66D`. The parent residue at a position is `wt[imgt.index(p)]`,
mapped through `dict(zip(imgt, wt))` rather than looked up per row. It is applied where `cf` is
built in both `run()` and `run_upload()`.

**In the pipeline, not in `app.py`.** `cf` and `wt` are both already in scope there, and nothing
inside `maturate/` imports streamlit — `app.py` is a renderer. Keeping the derivation in the
pipeline is what keeps that true.

**`heat()` reads the same frame.** It pivots `aa` against `imgt` to build the heatmap, so those
two fields are a contract: `sub` is added beside them, and neither is renamed or dropped. Only
the *displayed* headers change.

| Field | Displayed as |
|---|---|
| `sub` | Substitution |
| `imgt` | Position (IMGT), or just Position for an upload |
| `aa` | Amino acid |
| `beta` | beta |

An upload numbers positions `1..L` from its own parent (`encode.ref_cols`), so the header, the
heatmap's x axis and the IMGT caption all key off the same `ds.imgt` flag and cannot disagree.

#### Two definitions the page was missing

- **beta**, under the table: the learned weight for that substitution, in log-odds, with the
  sign explained and two fixed worked figures for the scale (+4.34 multiplies the odds by about
  76; −1.0 divides them by about 2.7). `exp(4.34)` is 76.7 and `1/exp(-1)` is 2.72. The figures
  are illustrations of the scale, **not values read off the current fit** — do not "correct"
  them against a changed run. +4.34 does currently coincide with AT110's top weight.
- **IMGT numbering**, under the heatmap: the same number means the same structural position in
  any antibody, and a lettered position such as `111A` is an insertion code for a loop that
  varies in length. Rendered only for a vendored campaign. AT110's x axis does carry `111A`,
  `111B`, `111C`, `112A`, `112C` and `112D`, so the example is visible on the chart it sits under.

The existing caption counting the features that survived the L1 penalty is accurate and stays.

#### Heatmap labels

Colour bar `weight` -> `weight (beta)`; y axis `substituted residue` -> `amino acid substituted
in`; x axis `IMGT position`, or `position` for an upload. The map, the symmetric `vmin`/`vmax`,
the sizing and the ticks are untouched. Nothing here changes a number.

#### The notation uses each campaign's own parent, verified

Asserted in code across every row — all 133 AT110 and all 469 RX002 coefficients reconstruct as
`wt[imgt.index(p)] + p + aa` against their own campaign's parent — and checked on screen. Two
positions are decisive, because the campaigns differ there:

| Position | AT110 parent | RX002 parent | RX002 shows | Would show if it used AT110's |
|---|---|---|---|---|
| 36 | A | R | `R36L` | `A36L` |
| 38 | I | V | `V38G` | `I38G` |

AT110's own table has `A36V` at that position, so the two campaigns visibly disagree where they
should.

### The Candidates tab says what its columns hold

Same problem as the coefficient table: the tab rendered `r.ranked` straight, so its headers were
`var_id`, `pred_enr`, `pred_cls`, `enr` and `cnt_*`. Two of those read almost identically and
mean different things. The ranking, the model and the row order are untouched throughout.

| Field | Displayed as |
|---|---|
| — | Rank |
| `aa_aln` | Substitutions, via the shared `subs()` helper at its default cap of 10 |
| `split` | Data split, in plain words |
| `pred_enr` | Predicted enrichment |
| `pred_cls` | Model call |
| `enr` | Measured enrichment |
| `cnt_<round>` | Reads, Input / Reads, Output 1 |
| `var_id`, `cdr3` | dropped from the display, kept in the CSV |

#### A downloaded ranking has to be actionable

The download previously emitted `r.ranked` raw and **carried no sequence column at all**, so a
downloaded ranking could not be traced back to a clone and ordered. `s.AA_ALN` now rides along in
the `cols` list that builds `ranked`, in both `run()` and `run_upload()`, and a cached
`rank_csv()` beside `enr_csv()` writes everything the display drops: the upstream row label, the
**untruncated** substitution list, CDR3, the sequence, and the raw `split` value.

Raw values, not display words, for `split` in the CSV — it joins back to `r.tbl`. That matches
what was already there: "Row label" is a friendly header over a raw `var_id`. CDR3 is written
only when the column exists, which is the vendored campaigns; `ingest` fills it with `""` when an
upload lacks it, and an all-blank column is noise.

Measured on the real downloaded file, AT110 FACS1/MACS: 4,067 rows, Rank contiguous 1..4,067,
every sequence 126 wide, CDR3 on every row, zero cells containing `+N more`.

#### Rank, and why it is computed before the filter

`pred_enr` saturates. At four decimal places the top rows read 0.9998, 0.9997, 0.9997, 0.9996,
0.9996, and the ties are real rather than a rounding artefact — the model has no per-variant
parameters, so two variants with the same substitution set get byte-identical scores. Measured on
AT110:

| | |
|---|---|
| Distinct values at 4dp in the top 50 | 25 of 50 |
| Rows sharing a score with an earlier row | 2,842 of 4,067 |
| Largest single 4dp collision | 1,103 rows |

Rank is the row's place in the **whole** ranking, computed before the split filter is applied, so
narrowing to out-of-sample rows shows how far down they sit instead of renumbering from 1.
Filtering AT110 to held-out plus not-labelled gives ranks 1, 2, 5, 9, 10 — information that
renumbering would destroy.

**Rank asserts an ordering the scores do not support inside a tie block.** Rank 1,200 against
rank 1,800 can be an arbitrary tiebreak within a 1,103-row group. Nothing on the tab says so.

#### Every row is scored, including the rows the model was fitted on

`pred_enr` comes from encoding every row and scoring it, training rows included. Of 4,067 AT110
candidates: 3,082 seen in training (76%), 771 held out (19%), 214 never labelled (5%). The
ranking is not wrong — 133 substitution weights and no per-variant parameters, and the
train/held-out gap is 0.017 (0.938 against 0.922) — but the tab gave no way to tell which scores
were out of sample.

`s.SPLIT` is carried through into `ranked` and shown as **Data split**: `train` -> seen in
training, `test` -> held out, and unset -> not labelled. The unset value is `NaN`, not a literal
`"unset"`, so the mapping fills it. A multiselect filters the table by it, defaulting to all
three; multiselect rather than a hide-training checkbox because it can isolate the never-labelled
group on its own.

**The last two categories are not the same thing and the caption must not blur them.** Held-out
rows were not used to fit the model but were used to compute the accuracy figures on the Model
tab. Not-labelled rows were used for neither: enrichment could not be called from their counts at
all, so they are the candidates only the model can rank.

That last clause is advocacy by the section 4 rule and was left out on the first pass; it is in
the caption because it was asked for explicitly, as an exception. The rule still holds elsewhere.

Worth it for the demo, not just for correctness: of the top 50 AT110 candidates, 20 were seen in
training, 5 were held out and **25 were never labelled**. Filtering to out-of-sample leaves 985
rows (771 + 214), confirmed from the downloaded CSV.

#### Known, not fixed

- Rank states a strict order inside tie blocks that the scores do not support, and the tab
  does not say so.
- The Candidates tab never states how many candidates match the split filter. Narrowing
  4,067 to 985 is invisible: the reader sees the first 50 and the slider caps at 500.
- The Onboard tab keeps its own "Run the pipeline on this upload" button, a second route to the
  same result as the sidebar. Same figures, confusing seam.

### Stretch, in order, each independently droppable

1. ~~**Upload path.**~~ **Done.** The public format `{clone id, sequence, count per round}`
   goes down the same `enrich → encode → model` path and into every tab. The equal-length
   guardrail in `chk()` refuses ragged input by name rather than padding it. This is why
   recomputing from counts mattered: the upload path is the code that already runs.
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
3. **Alignment layer.** **Active next priority (post-hackathon).** Two approaches planned:
   ANARCI (antibody-specific, HMM-based IMGT numbering, ~3.7 min per 10k sequences) and
   MAFFT (general-purpose MSA, works on non-antibody proteins). ANARCI first for the primary
   nanobody use case; MAFFT as a fallback for broader applicability.
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
   readings: **(a)** MatuRate is enrichment prediction as proposed and the flag is one feature
   inside it, most likely an extension of the developability filter or a confound-aware model
   term; **(b)** MatuRate *is* the flag and the proposal is a separate third idea. **(a) is
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
   the offending rows. **Resolved post-hackathon: building an alignment layer is the active next
   priority.** Plan is ANARCI for antibody-specific IMGT numbering, plus MAFFT as a
   general-purpose fallback for non-antibody datasets.
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
14. ~~Who owns integration between workstreams.~~ **Resolved: moot.** The repo is now a
    single-project repo (MatuRate only); `true-hit-qc` was removed during the post-hackathon
    restructure on 2026-10-27.

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
- MatuRate repo: https://github.com/MatuRate-Biotech/MatuRate (pending repo rename from `true-hit-atg`)
