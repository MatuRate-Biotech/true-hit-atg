# Plan: two AI features for TrueHit ATG

Scope: the inner `true-hit-atg/` directory only. `claude-docs/context.md` in this directory
stays the source of truth for everything else; this file covers only the two additions and how
they attach to what is already built. Nothing here touches `true-hit-qc/`.

Both features are **additive**. The app, the 43 tests and the measured AT110 numbers must still
work with both features absent or failed. If either runs out of time, delete its tab and ship.

---

## Why these two

The repo currently has a working, reproduced implementation of Paul et al. and two honest
weaknesses:

1. **Generality is claimed but unproven.** `context.md` section 3 says the standard input format
   is the keystone and that onboarding a new dataset means writing a small converter. Nobody has
   written a second converter, so the claim rests on future human effort. **Feature A** makes the
   converter write itself.
2. **The distinctive demo beat is hollow.** Section 9 question 11 calls this the highest-value
   unanswered question: scoring outside clones works, but it is demonstrated on the parent
   sequence, which was in the campaign. **Feature B** supplies candidates worth pasting in.

Both are AI doing a job nothing else in the pipeline can do. Neither is a chatbot bolted on.

Background that makes Feature A worth claiming out loud: a 2026 benchmark of LLM agents across
76 protein design tasks found agents choose tools competently but never filter. Across 836
task-condition observations no agent discarded a single generated candidate, and they invoked
scoring tools at roughly 14% of expert intensity. The intervention that fixed it was forcing
generate, score, rank, submit-top. That is this pipeline's job description.

---

## Feature A: agent-written dataset converters

### The gap in code terms

`schema.chk()` enforces the internal contract: `CORE` plus `cnt_/norm_/has_` per round, equal
`aa_aln` width, unique `var_id`. That contract is the post-enrichment table. A user upload will
not have `enr`, `enr_bin`, `split`, or the normalized columns, because `enrich.py` computes
those. So the converter's target is **upstream of enrichment**, not `chk()`'s current shape.

First piece of work is therefore to name the raw contract that does not yet exist.

### A1. Add the raw contract to `schema.py`

```python
RAW_CORE = (VAR, AA_ALN)

def chk_raw(df, rnds, n_pos=None):
    """validate an upload before enrichment. rnds is the round names in order, earliest first."""
```

Rules `chk_raw` enforces, mirroring `chk()`'s style of naming offenders rather than coercing:

- `VAR` and `AA_ALN` present, `VAR` unique
- at least two `cnt(r)` columns
- `aa_aln` equal width across all rows; when ragged, list the first five offending `var_id`s and
  the distinct widths, and refuse. Reuse the existing message about padding scrambling positions.
- sequence alphabet is a subset of `encode.ALPH`; report unknown symbols with their rows
- counts are non-negative integers; report the first offenders

`AA` and `CDR3` are optional on upload. Derive `AA` as `aa_aln` with gaps stripped.

### A2. `truehit_atg/ingest.py` — deterministic half

No LLM in this file. It is the thing the agent targets and the thing tests exercise.

```python
@dataclass(frozen=True)
class Spec:
    var: str                      # source column -> var_id
    seq: str                      # source column -> aa_aln
    rnds: dict                    # {round name: source column}, insertion order = round order
    drop_gaps: bool = False       # source sequences already ungapped
    note: str = ""                # agent's one-line rationale, shown in the UI

def probe(path, n=20):
    """compact description of an unknown csv for the agent: columns, dtypes, n rows, head."""

def apply_spec(path, spec):
    """read csv, rename per spec, build the raw frame. no inference, no guessing."""

def to_internal(raw, pair, cfg=None):
    """raw frame -> internal table via the existing enrich path. returns the same shape data.load() does."""
```

`to_internal` must reuse `enrich.py` rather than reimplementing the recipe. The CPM floor, log10,
GMM at 0.8 posterior and the 42-seed split are verified against upstream and must not be forked.

`Spec` being a frozen dataclass rather than generated code is deliberate: it is serializable,
diffable, testable, and carries no execution risk. It covers renames and round ordering, which is
what most real files need.

**Escape hatch, build only if time allows.** Some files cannot be expressed as renames. A MaveDB
score set gives `hgvs_pro` mutation notation instead of full sequences, so the sequence has to be
reconstructed against the parent. If that case is in scope, add an optional
`Spec.prep: str | None` holding a generated function body, executed in a namespace containing
only `pd`, `np` and the input frame, with no `__import__`, no file and no network access. Default
it off. Say in the UI when a spec carries generated code.

### A3. `truehit_atg/agent.py` — the loop

```python
@dataclass
class Attempt:
    spec: Spec | None
    ok: bool
    err: str
    n_rows: int

def propose(desc, wt_len, rnds_hint=None, client=None):
    """one LLM call: probe output in, Spec out. client injectable so tests never hit the network."""

def onboard(path, camp_wt, max_tries=3, client=None):
    """probe -> propose -> apply -> chk_raw -> on failure feed the error back and retry.
    returns (raw_frame, Spec, [Attempt]) so the UI can show the agent's working."""
```

The loop is the feature. Spell it out:

1. `probe(path)` builds the description
2. `propose()` asks for a `Spec` as strict JSON, with the target contract and the campaign's
   aligned width in the prompt
3. `apply_spec()` builds the frame
4. `chk_raw()` validates
5. on `ValueError`, append the exact message to the conversation and go to 2
6. after `max_tries`, return the attempt history and let the user fix it by hand

Record every attempt. The failed ones are the demo: watching it read an error and correct itself
is more convincing than a clean first pass.

Prompt should include: the `Spec` field list, the raw contract rules, `wt_len`, the probe output,
and a hard instruction to return only JSON. Keep it in one module-level constant so it is
reviewable.

### A4. Config

Add to `Cfg` in `config.py`:

```python
ai_model: str = "claude-sonnet-4-5"
ai_key_env: str = "ANTHROPIC_API_KEY"
ai_max_tries: int = 3
ai_timeout: int = 60
```

No key in the repo. Missing key must degrade to a clear message in the UI, not a traceback.

### A5. App

New tab, "Onboard". File uploader, then:

- the probe output as a table
- each attempt as an expander: the proposed `Spec` as JSON, pass or fail, the error if it failed
- on success, the first rows of the converted frame and a Run button that feeds it through
  `enrich -> encode -> model` exactly as the vendored path does

The existing five tabs must be untouched.

### A6. Tests, `tests/test_ingest.py` and `tests/test_agent.py`

No test may make a network call. Use a `FakeClient` returning canned JSON.

- `chk_raw` accepts a good frame
- `chk_raw` names offenders on ragged widths, duplicate ids, missing counts, bad alphabet,
  negative counts
- `apply_spec` renames correctly and preserves row count
- `to_internal` on a fixture reproduces `data.load()` output on the same rows, since this is the
  only proof the upload path and the vendored path are the same code
- `onboard` with a FakeClient that fails once then succeeds returns two attempts and a valid frame
- `onboard` exhausting `max_tries` returns history and raises or returns `ok=False`, not a crash

Fixtures live in `tests/fixtures/`: a clean file, a ragged one, and one with renamed columns.

### A7. Proving generality

The point of the feature is a dataset that is not Paul et al. Per `context.md` section 7, tier A
is ProteinGym, whose DMS substitution files carry `mutant`, `mutated_sequence`, `DMS_score` and
`DMS_score_bin`, fixed-length and position-matched by construction with a binary label already
provided.

Note the shape difference: ProteinGym has a score and label but **no per-round counts**. So it
exercises `encode -> model` and not `enrich`. That is fine and worth stating plainly. Two paths:

- `Spec` gains optional `enr_col` and `enr_bin_col`, and `to_internal` skips enrichment when they
  are present
- or a second entry point, `from_scored()`, for datasets that arrive already labeled

Prefer the first. It keeps one path.

Pull the substitutions reference file and filter for an antibody, VHH or nanobody assay so the
molecule class matches. **Download tonight.** No venue wifi.

---

## Feature B: protein language model proposes, the fitted model ranks

### What already exists

`pipeline.score_seqs(res, seqs)` takes aligned sequences and returns a ranked frame with
`pred_enr`, `pred_cls` and `n_sub`, and it already raises a clear error on wrong length. So the
entire scoring half is done. **Feature B only has to produce `seqs`.** Keep it that way; do not
add a second scoring path.

### B1. The alignment problem, solve this first

ESM operates on ungapped sequence. The model operates on `aa_aln`, fixed width per campaign, gaps
as `-`. `wt_seq('AT110')` starts `QVQLQESGG-GLVQAGGSLRLSCAASGNIF----DADIMG`, so the gaps are real
and in the middle.

Build and test the index map before touching ESM:

```python
def gap_map(wt_aln):
    """-> (ungapped_seq, [aligned_index for each ungapped position])"""

def place(wt_aln, pos_ungapped, aa):
    """substitute at an ungapped position, return a full-width aligned sequence"""
```

Test that `place` round-trips every position and that output length always equals
`len(wt_aln)`. A bug here silently shifts every downstream position, which is exactly the failure
mode `chk()` exists to prevent. Write these tests first.

### B2. `truehit_atg/design.py`

```python
def esm_scores(seq, model_name=None):
    """masked-marginal log-likelihood per position per amino acid. torch imported lazily inside."""

def propose(wt_aln, top_k=3, min_llr=0.0, model_name=None):
    """-> DataFrame[pos_aln, pos_ungapped, wt_aa, sub_aa, esm_llr, seq_aln]
    single substitutions only. one row per candidate."""

def rank(res, cands):
    """pipeline.score_seqs on cands.seq_aln, joined back to keep esm_llr beside pred_enr."""
```

Method: for each ungapped position, mask it, run the model, read the distribution over the 20
amino acids, and take the log-likelihood ratio of each substitution against the wild-type
residue. Keep the top `top_k` per position above `min_llr`. For AT110 at 126 aligned positions
that is a few hundred candidates, which `score_seqs` handles instantly.

**Single substitutions only for the MVP.** It matches the "single-substitution picks" row in the
feature menu, it is interpretable, and it avoids the combinatorial explosion. Doubles are a
stretch and should be built by combining the top singles, not by enumerating.

### B3. Dependency handling

`facebook/esm2_t12_35M_UR50D` is the right size, roughly 140MB.

- `torch` and `transformers` go in a separate `requirements-design.txt`, not `requirements.txt`
- `design.py` imports them inside the function, not at module level
- the app's Design tab checks availability and shows an install hint instead of crashing
- `pytest` must pass with neither installed; mark the ESM tests `skipif`
- **set `HF_HOME` to a vendored path under `data/` and download the weights tonight.** Add that
  path to `.gitignore` unless the weights are small enough to commit, and note the decision in
  `data/DATASET.md`

### B4. The output that makes the demo

A table with both scores side by side:

| candidate | substitution | esm_llr | pred_enr | agree |
|---|---|---|---|---|

`esm_llr` is general protein plausibility, learned from millions of sequences, knowing nothing
about AT1R. `pred_enr` is this campaign's own learned preference, knowing nothing about proteins
in general. **They are independent, which is the whole point.** Candidates where both are high
are the interesting ones, and disagreements are worth a sentence in the pitch.

Cross-check worth running once: do the substitutions ESM likes correlate with the L1
coefficients in `res.cf`? Either answer is a result. High correlation says the campaign
rediscovered general protein preferences. Low correlation says the campaign learned something
target-specific that a general model misses, which is a better story.

### B5. Honesty constraints, non-negotiable

`context.md` section 4 applies in full. Specifically for this feature:

- these are **proposed** candidates, never "validated", "improved" or "better binders"
- the column is `pred_enr`, never anything containing `affinity` or `binding`
- nothing here is wet-lab tested and the UI says so on the tab, not in a footnote
- ESM proposing and the model ranking are independent; if that ever stops being true, for
  instance by filtering ESM candidates using model weights before scoring, the independence claim
  dies and must be removed from the pitch

### B6. The cheap fallback, and why it is worse

If torch cannot be installed in time, candidates can be built from the top-weighted substitutions
in `res.cf`. This needs no new dependency and takes under an hour.

But it is **circular**: candidates constructed from the model's own weights, then scored by that
model, will of course score well. This is the same failure as the counts baseline documented in
section 6. If this path is taken, label it in the UI as a sanity check rather than a design
feature, and do not present the resulting scores as evidence of anything.

---

## Sequence

Each step ends in a committable state. Stop anywhere.

| Step | Work | Rough time | Ends with |
|---|---|---|---|
| 1 | `gap_map` / `place` plus tests | 45 min | Alignment mapping proven safe |
| 2 | `chk_raw` plus tests | 45 min | Raw contract exists |
| 3 | `ingest.probe`, `Spec`, `apply_spec`, `to_internal` plus tests | 2 hr | Uploads work with a hand-written spec |
| 4 | `agent.propose` and `onboard` with FakeClient tests | 2 hr | Feature A complete headless |
| 5 | Onboard tab | 1 hr | Feature A demoable |
| 6 | ProteinGym converter and a real run | 1.5 hr | Generality proven on outside data |
| 7 | `design.esm_scores` and `propose` behind lazy import | 2 hr | Feature B headless |
| 8 | Design tab with the two-score table | 1 hr | Feature B demoable |
| 9 | ESM versus coefficient cross-check | 45 min | A finding for the pitch |

Steps 1 and 2 are cheap, carry no dependency risk, and both features need them. Do them first
regardless of which feature survives.

Feature A before Feature B because A has no heavy dependency and cannot be blocked by a failed
install. If torch is already working, swapping 7 and 8 earlier is reasonable.

---

## Risks

| Risk | Mitigation |
|---|---|
| Venue wifi dies, agent demo dies | Cache one recorded `onboard` run as a fixture and let the tab replay it |
| torch install eats hours | Install tonight, before anything else; the lazy import means failure costs only the Design tab |
| ESM weights not downloaded in time | Vendor under `data/`, verify offline, note it in `DATASET.md` |
| Gap mapping bug shifts positions | Step 1 first, tested, before any ESM work |
| Generated `prep` code executes something unwanted | Restricted namespace, default off, flagged in the UI |
| Existing 43 tests break | Both features add modules rather than editing `enrich.py`, `encode.py` or `model.py`; run the suite at every step |
| Demo now has too many tabs | Seven tabs is a lot for three minutes. Pick the beats in advance, do not tour the UI |

---

## What not to do

- Do not reimplement the enrichment recipe inside the upload path. It is verified at 99.4%
  against upstream and there is one copy of it.
- Do not swap one-hot for ESM embeddings. The model is at 0.960 against their CNN's 0.938; this
  is not the weak point and the dependency cost is not worth a marginal AUC change.
- Do not let the agent touch `enrich.py`, `encode.py` or `model.py`. It writes a `Spec`, nothing
  more.
- Do not add a CNN. It is cut, per 2026-10-23.
- Do not claim the designed candidates are better binders. They are untested proposals.

---

## Open questions these raise

Add to `context.md` section 9 once settled.

1. Does the ProteinGym path skipping enrichment weaken the "one pipeline" claim, or is an
   already-labeled dataset a legitimate second entry point?
2. If ESM substitution preferences correlate strongly with the fitted coefficients, does the
   two-score table still say anything, or does it collapse into one opinion?
3. Should the agent's `Spec` be shown to the user for approval before the run, or applied
   silently with the history available afterwards? Approval is more honest, one more click on
   stage.
4. Jessica's read on whether single-substitution ESM proposals are biologically sensible for a
   nanobody, in particular whether framework positions should be excluded from proposal and only
   CDR positions offered.
