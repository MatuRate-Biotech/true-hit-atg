# TrueHit — Cal Hacks 13.0 project context

This file is the full working context for the project. It is written for Claude Code to read
before setting up the repository. It is deliberately detailed rather than summarized.

Repo: https://github.com/heller1729/true-hit-atg
Event: Cal Hacks 13.0, Palace of Fine Arts, San Francisco. Hacking started Friday 2026-10-23
evening, judging Sunday. Roughly 36 hours of build time, 3000+ hackers, ~700 projects expected.

---

## 1. Who is on this

- **Pratham Jangra** (GitHub `heller1729`). MS CS thesis student at Virginia Tech, Graduate
  Research Assistant on the NSF-funded CyberSMART project. Strong on backend and software,
  close to zero hands-on hardware experience. Owns: the pipeline, streaming, backend, and the
  ML scoring work.
- **Jessica Pan** (Slack handle "Percabeth"). The domain expert. Antibody and molecular biology
  background. Owns: biology decisions, what counts as an artifact, germline references, QC
  logic correctness. She authored the original proposal and the TrueHit ATG proposal.
- Two more teammates being recruited as of Friday evening. Linero and Linlin both dropped out
  on Oct 23. Assume a 2-person team when planning scope; treat extra hands as a bonus.

Working style notes that matter for generated code:
- No comments in code that look AI-authored.
- Short variable names are preferred (`mtc` over `matched`).
- Pratham prefers practical, forward-moving output over exhaustive scaffolding.

---

## 2. Two ideas, two directories

The user explicitly asked that the repo hold **two separate directories**, one per idea. They
may be merged later; for now keep them independent so each can be built and demoed alone.

```
true-hit-atg/                 <- the repo
  true-hit-qc/                <- Idea 1: artifact QC on a naive antibody library (simulated data)
  true-hit-atg/               <- Idea 2: ML-guided nanobody affinity maturation (real dataset)
  claude-docs/context.md      <- this file
  README.md
```

The inner `true-hit-atg/` intentionally shares its name with the repo. Idea 2 is the headline
project, so the repo is named after it; Idea 1 sits alongside as `true-hit-qc/`. When a path is
ambiguous in conversation, "the repo root" means the outer one.

The two differ in a way that matters for setup: Idea 1 generates its own data and measures
against a planted answer key. Idea 2 runs on a published dataset with real validation labels.
They share almost no code, which is another reason to keep them apart for now.

Shared code (sequence utilities, simulators, plotting) may live in a top-level `common/` only
if both directories genuinely need it. Prefer duplication over premature coupling during a
hackathon.

---

## 3. Domain background (needed to write correct code)

Terms used throughout. Treat these as the project's vocabulary.

- **Sequence**: a string. DNA over `A C G T`. Protein over ~20 letters. An antibody variable
  domain is a few hundred letters.
- **Library**: a pool of tens of thousands to millions of designed antibody variants in one
  tube. Built by taking a starting antibody and mutating the region that grabs the target.
- **Variant / clone / library member**: one distinct sequence in that pool.
- **Format**: the molecular shape the library was built in.
  - Full antibody: the complete Y, two heavy chains and two light chains.
  - **Fab**: one arm of the Y.
  - **scFv**: the two tips of one arm (heavy and light variable domains) joined by a short
    peptide linker into a single chain. Roughly 250 protein letters, ~750 DNA letters.
  - **Nanobody / VHH**: a single variable domain, from camelids. Roughly 120 protein letters.
- **Framework**: the parts of an antibody that stay roughly constant across variants. They
  hold the fold together. Real frameworks align closely to a known germline.
- **CDR**: the hypervariable loops that do the binding. This is what gets mutated to make a
  library. CDR3 of the heavy chain is the most diverse.
- **Germline**: the natural antibody gene segments a sequence derives from (V, D, J segments).
  IMGT is the reference database. ANARCI and IgBLAST are the standard tools that number
  antibody sequences and assign germline.
- **Naive library**: the library as built, before any selection has been applied. This is the
  input for Idea 1.
- **Selection / panning / biopanning**: incubating the library with an immobilized target,
  washing away non-binders, keeping what stuck, then amplifying the survivors to have enough
  material for the next round. Rounds are conventionally labeled R1, R2, R3. The naive library
  is effectively R0.
- **Affinity maturation**: deliberately mutating selected binders into new variants to improve
  them. Produces variants that differ from each other by one or two letters by design.
- **Reads**: the sequencer outputs millions of strings; each read is one noisy observation of
  one variant. The same variant is read many times.
- **Reference**: a list of every sequence expected in the library. Available when you designed
  every member yourself. **Not available** for naive or affinity-matured libraries, which is
  precisely the gap this project targets.
- **Enrichment**: how a variant's read count changes round to round. Rising variants are the
  ones labs act on.

### Artifact types (what we are hunting)

| Artifact | What it is | Detectable by |
|---|---|---|
| Sequencing typo | Machine misreads a base; random, differs read to read | Consensus voting within a cluster |
| Truncation | A variant physically cut short during library construction | Length and coverage check |
| Chimera | Two different variants fused at a breakpoint during PCR; front half from parent A, back half from parent B | Population statistics (see below) |
| Contamination | A molecule from another sample or a different library format | Format check: domain count, length |
| Framework error | A badly assembled framework that matches no known germline, or breaks a conserved residue | Alignment to germline references |
| Index hopping | Reads misassigned between pooled samples during sequencing | Cross-sample abundance statistics |
| Propagation-advantage clone | A completely intact variant that rose because it grows faster, not because it binds | **Idea 2** — needs a learned model |

Published numbers worth knowing (from literature, useful for calibrating simulators and for
the pitch; do not state them as our own measurements):
- Inter-sample chimeras have been reported at roughly 15% of reads; intra-sample chimeras at
  roughly 20%, in deep antibody repertoire sequencing.
- One antibody-engineering study reduced sample cross-contamination (index hopping) from 14%
  to negligible by changing library prep method.
- Healing wounds / normal pH context is NOT relevant here — that was a different, dropped idea.
- A reference-free chimera checker (vsearch `uchime_denovo`) exists and achieved the highest
  precision among de novo chimera detectors on simulated long-read data. It is our baseline,
  not our contribution.

---

## 4. Idea 1 — `true-hit-qc/`

### Problem statement

A lab builds or receives an antibody library and sequences it. The resulting data is full of
molecules that look like real, novel candidates but are artifacts. If an artifact is mistaken
for a hit, someone orders it, expresses it, and tests it, burning weeks and real money on a
molecule that never existed. Reference-matching tools solve this when you designed every
sequence yourself. They cannot help with naive or affinity-matured libraries, where there is no
list to match against. Today that gap is closed by clustering and then eyeballing the output.

Two audiences:
1. Traditional discovery groups (pharma, biotech, university core labs) running large naive
   libraries.
2. AI protein-design companies, where artifacts silently poison training data, and which run
   affinity maturation — exactly the no-reference case.

### Pipeline

```
raw reads
  -> Step 1: clustering and dereplication          (existing tools)
  -> Step 2: certain checks                        (contamination, framework, truncation)
  -> Step 3: chimera suspicion score               (population statistics)
  -> Step 4: report                                (trusted variant list + extras)
```

**Step 1 — clustering.** Group near-identical reads (differing by a letter or two) so one
cluster is one true variant, then take a per-position consensus vote to recover its sequence.
Output: one row per variant with its read count. Use existing open source tools: Starcode,
MMseqs2, or VSEARCH. This step is not the contribution.

Known hazard, raised by Pratham and acknowledged by Jessica: in affinity-matured libraries real
variants can differ by a single letter, which is indistinguishable from a sequencing typo by
edit distance alone. Too loose a clustering threshold silently merges real hits. Jessica's
proposed fix is a toggle that changes clustering behavior for naive versus affinity-matured
input. The code should expose the identity threshold as a parameter, not bury it.

**Step 2 — the certain checks.** Three artifacts are unambiguous and get reliable rule-based
checks:
- *Contamination*: wrong format for this library. Check number of domains and total length.
  E.g. in an all-nanobody library (~120 aa, single domain), a 250 aa two-domain sequence with a
  linker is an scFv and does not belong.
- *Framework errors*: align to a germline reference set; flag sequences matching no germline,
  or missing conserved residues (e.g. the canonical cysteines that form the disulfide).
  Germline references are downloadable (IMGT). Jessica's plan also lets users upload their own.
  **Download these before the event; assume unreliable venue wifi.**
- *Truncations*: coverage and length check against the expected format's region layout. An
  scFv cluster that stops partway through the linker is missing its entire light domain.

Mixed-format libraries break the naive contamination check. Jessica's answer: report a
percentage per detected structure and let the scientist decide whether a mix is contamination
or intentional. Implement it as a distribution, not a binary flag.

**Step 3 — chimera suspicion score.** The genuinely hard artifact. In a diverse library a
fusion of two members can look exactly like a legitimate new variant; the sequence alone cannot
tell you. So score from population context instead. Three signals:
1. The variant is rare (low read count).
2. Its front segment matches one abundant variant (parent A) and its back segment matches a
   different abundant variant (parent B).
3. The switch point sits where fusions tend to occur during library prep.

Output a **suspicion score with named parents**, not a yes/no. Example: `X: 92% likely chimera,
parents A and B`. Some real variants will look like this by chance, and the honest output is a
probability.

Open question: how do we know where breakpoints tend to occur? Either learn it from the planted
data, or parameterize it and state the assumption.

**Step 4 — report.** The deliverable is a trustworthy variant list: every real sequence with
its count, every artifact flagged with a reason. Plus two extras:
- *2D diversity map.* Featurize each variant (k-mer counts, or pairwise edit distance), reduce
  to two dimensions with UMAP or t-SNE, plot every variant as a dot colored by artifact status.
  Real variants form tight groups; contaminants sit alone; chimeras land between the two groups
  they were fused from. This is the one picture a non-biologist judge can read instantly and is
  the single most important demo asset.
- *Developability flags and health metrics.* Rule-based flags for sequence motifs known to
  cause trouble downstream (deamidation sites, N-glycosylation sequons, oxidation-prone
  residues), plus overall diversity and health numbers for the library.

### Scope for the hackathon

Build on **simulated data generated by us**, with artifacts planted deliberately. That gives an
answer key, control over difficulty, and a real accuracy number.

MVP by the halfway mark:
- Clustering recovers most planted variants
- The three certain checks working
- End to end on one input with a basic results readout

Stretch, in order:
- Chimera suspicion scoring
- 2D diversity map
- Developability flags and diversity metrics
- Polished demo UI

### Data generation requirements

Ground truth library:
- Pull real antibody sequences from OAS (Observed Antibody Space),
  https://opig.stats.ox.ac.uk/webapps/oas — free download, filter by species and chain,
  pre-annotated with germline genes, over a billion unpaired sequences from 80+ studies. Use
  this for germline and species labels. **Download before the event.**
- OAS holds single chains, not formats, so construct formats ourselves:
  - scFv = heavy variable + linker (classically (G4S)x3) + light variable
  - Fab = heavy variable + CH1, light variable + CL
  - Nanobody = camelid single domain sequences
  - Junk = shuffled sequences and random non-antibody protein
- Have Jessica sanity-check that the constructs look realistic.

Read simulator produces, from the true library:
- random substitutions at a configurable error rate
- chimeras: join two variants at a random (or motif-biased) breakpoint
- truncations: cut a variant short
- contamination: inject sequences of a different format
- framework errors: corrupt a framework region
- multiple rounds where chosen variants grow in count
- every emitted read carries its ground-truth provenance so accuracy is measurable

Unresolved with Jessica: short reads (Illumina, 150-300 bp) versus long reads (PacBio,
nanopore). An scFv is ~750 DNA bases, so short reads cannot span a full variant without a
merging step. **Long reads are simpler for the demo; default to long reads unless Jessica says
otherwise.** Also unresolved: realistic error and chimera rates for the simulator, and the
library size and read count that keep the demo running on a laptop.

### Variant table schema (the handoff contract)

Agree on this first so both halves can be built in parallel.

| Column | Meaning |
|---|---|
| `var_id` | cluster id |
| `dna` | consensus DNA sequence |
| `aa` | translated protein sequence |
| `cnt_r1`, `cnt_r2`, ... | read count per round |
| `qc_flag` | clean / chimera / truncated / contaminant / framework_error |
| `qc_reason` | human-readable reason string |
| `fmt`, `fmt_conf` | detected format and confidence |
| `germ`, `species` | assigned germline and species |
| `chim_score` | chimera suspicion probability |
| `chim_parents` | parent var_ids when flagged |

---

## 5. Idea 2 — `true-hit-atg/` (TrueHit ATG)

Source: Jessica's proposal "Cal Hacks Proposal: ML-Guided Nanobody Affinity Maturation",
dated 20 Oct 2026. The content below follows that document.

### One-line statement

A tool that learns from antibody selection data to predict which binders are worth pursuing,
turning several rounds of lab screening into a model that ranks candidates after just one. You
feed it sequencing data from an affinity-maturation campaign, it measures how much each variant
enriched round over round, and it trains a model to predict that enrichment, so you find the
good variants earlier with less lab work.

### Background, as the proposal frames it

The whole thing rests on four ideas:
- A **binder** is a protein, here a **nanobody** (a small single-domain antibody), that sticks
  to a target. You have an okay one and want a better one.
- **Affinity maturation** is how you improve it: make a big library of mutated versions of the
  starting binder, then run several rounds of selection that keep the best binders and wash the
  rest away.
- **Enrichment is the signal.** Good binders survive each round, so their share of the pool
  rises round over round; bad ones fall. Sequence the pool after each round and you can measure
  this. Rising is good, falling is bad.
- **The data is one table per round**: each variant (a sequence) and how many times it was
  seen. Compare across rounds to get an enrichment value per variant.

### The problem

Affinity maturation works but is slow and expensive. Narrowing a huge mutant library down to a
few good binders takes multiple rounds of selection, meaning weeks to months of lab work and a
lot of pricey reagent. The readout is also noisy: raw enrichment counts correlate only loosely
with true binding, and later rounds can enrich the wrong things (sticky, over-expressed, or
non-specific binders) alongside the real ones.

The opportunity: a model trained on the sequencing data can learn which sequence features drive
enrichment and predict good variants from an early round instead of running the campaign to the
end. Recent work shows this works, with a model trained on a single sorting round predicting
what enriches several rounds later. Fewer rounds, less reagent, and a shot at variants the raw
counts would have missed.

### Architecture: two layers

```
per-round sequencing data in
  -> Layer 1: data + enrichment       (clean, count per variant per round, compute ratios)
  -> Layer 2: ML                      (train on enrichment, predict, rank)
  -> ranked binders out
```

**Layer 1, data and enrichment.** Take per-round sequencing data, clean and count each variant
per round, and compute enrichment ratios. This is also where an alignment front end lives if we
start from raw reads.

**Layer 2, ML.** Train a model on those enrichment values to predict which variants are
enriched or good, ideally from an early round, then rank or design candidates.

**The honest baseline is the enrichment ratios themselves** (rank by what rose). The ML layer is
what adds value beyond that. Both are useful, and the baseline can ship even if the ML turns out
shaky.

### Role split, as written in the proposal

- **Pratham, the ML layer**: encode the sequences (one-hot or simple features), train models to
  predict enrichment (logistic or linear regression as the honest baseline, a small CNN as a
  stretch), evaluate, and ideally wrap it in a small app that takes a dataset and returns
  ranked candidates.
- **Jessica, the data and enrichment layer**: assemble the per-round data, compute enrichment
  ratios, handle the noise (count thresholds, normalization), and if starting from raw reads,
  build the align and count front end. She hands over a clean table: variant, per-round counts,
  enrichment value.

Agreeing on that table format early is what lets both sides build in parallel. **This is the
first thing to lock down in code.**

### Feature menu with trade-offs (from the proposal, verbatim in substance)

A menu to pick from, not all of these, just the ones worth 36 hours.

| Feature | Pros | Cons |
|---|---|---|
| **Enrichment-ratio ranking** — rank variants by how much they rose across rounds | Dead simple, deterministic, trustworthy; works even if the ML fails; the honest core | Not novel; not "AI"; noisy for low-count variants |
| **ML enrichment prediction** — train on an early round, predict later-round enrichment | The real ML; genuinely useful (fewer rounds); fits the data we have | Reimplements a published method, not novel; needs enough data to train |
| **Antigen-specific models** — one model per target, learning that target's preferences | More tractable and honest than general prediction; 3 targets in the dataset to try | Less data per target; harder to generalize; more models to juggle |
| **Single-substitution picks** — score each point mutation, recommend the best | Very interpretable; directly actionable; linear models do this well | Misses combinations; "passenger" mutations can mislead |
| **Novel sequence design** — sample new variants from the model | Flashy demo ("we designed new binders"); shows generation, not just ranking | No wet-lab validation in a weekend; easy to overclaim |
| **Developability / polyreactivity filter** — flag sticky or liability-prone sequences | Real-world credibility; catches a known failure mode; mostly rule-based | A side feature; a pretrained model already exists, so limited novelty |
| **Align + count from raw FASTQ** — build enrichment from reads, not pre-made tables | Gives Layer 1 real work; the full end-to-end story; reusable skill | More plumbing; the main dataset is already pre-counted, so partly optional |
| **2D landscape map** — plot variants, colored by enrichment or prediction | Striking demo visual; shows clusters and where predictions land | Purely presentational; doesn't improve the model |
| **Validation vs. measured affinities** — check the ranking against real binding data | The honest, non-circular proof the model works; strongest credibility | Only where affinity labels exist (RX002 has some); limited scope |

**Jessica's stated priorities:** enrichment-ratio ranking and ML enrichment prediction are the
core, do these first, one each. Antigen-specific is a natural framing on top. Validation against
measured affinities is the highest-value credibility add if time allows. The align front end is
the piece she most wants for the experience, but it is also the part drifting furthest from the
ML core, so if it starts feeling like a separate project, drop it entirely and start from the
pre-counted tables. Novel design and the 2D map are demo candy, nice if ahead, skippable.
Developability is a tidy bonus.

### The data — this is real data, not simulated

**Important difference from Idea 1.** Idea 1 runs on data we simulate. Idea 2 runs on a
ready-made public dataset.

Source: the nanobody affinity-maturation dataset from **Paul et al. (2026)**, on GitHub at
**`debbiemarkslab/ML_affinity_maturation`**.

- Three campaigns, each a nanobody matured against a different GPCR target:
  - **AT110** against AT1R
  - **B7** against β2AR
  - **RX002** against RXFP1
- Per-round sequencing: one MACS round, then FACS1, FACS2, and FACS3 for B7. That is the
  round-by-round structure the method needs, already processed into enrichment.
- Ground truth for validation: AT110 and B7 have experimentally validated optimizing
  substitutions; RX002 has measured affinities for designed variants.
- B7 is the richest: three FACS rounds plus a polyreactivity counter-selection.

**Day-one to-do, called out in the proposal: confirm the repo ships the processed per-round
tables, not just code.** Clone and verify this before anything else, and vendor it locally
before relying on venue wifi.

### Scope for the weekend

Get the honest baseline working end to end first, then add the ML, then reach for extras.

Minimum viable:
- Load one campaign's per-round data, compute enrichment ratios, rank variants
- Train the baseline model (logistic or linear regression) on an early round, predict a later
  round, show it beats a naive baseline
- A basic readout of the ranked candidates

Stretch:
- Antigen-specific comparison across all three campaigns
- Validation against measured affinities and known substitutions
- A small app or UI, and the 2D landscape map
- The raw-FASTQ align and count front end
- Novel sequence design

If only the MVP lands, that is still a complete, honest tool with a real result.

### Honest notes the proposal says to agree on up front

These are constraints on how the project is described, not optional framing. Respect them in
code comments, README text, and anything that reaches a judge.

1. **This isn't novel.** It is an open implementation of a published method. That is a fine
   hackathon and learning project, but do not pitch it as new. Pitch it as a clean, usable,
   open version, and put any original energy into our own twist: the app, the cross-target
   comparison, our own validation.
2. **Enrichment is not affinity.** The model predicts enrichment, meaning what survives
   selection, which is a proxy for binding confounded by expression and stickiness. Claim
   "predicts enrichment" or "ranks candidates", **never** "predicts binding affinity". A sharp
   judge will know the difference.
3. **Validate across rounds, not within.** Testing on held-out variants from the same round
   risks learning the experiment's quirks. Train on an early round and test on a later one, or
   against measured affinities, for an honest result.

### The demo

1. "Here's sequencing data from a nanobody being improved over several rounds of selection."
2. Compute enrichment; show the good variants rising round over round.
3. "Now we train a model on just the first round, and it predicts which variants enrich later,
   without running the full campaign."
4. Show the ranked predictions, and if we got it, that the top picks match the known good
   binders or measured affinities.
5. Close: "This is how you cut an affinity-maturation campaign from many rounds to one. Less
   time, less reagent."

### Discrepancy to resolve with Jessica

On the call of 23 Oct, Pratham and Jessica agreed to add a **growth versus binding flag**:
identifying clones that rise across rounds because they replicate faster rather than because
they bind, which are structurally intact and so invisible to every check in Idea 1. Jessica
took on researching whether the two can be separated.

This proposal document predates that call (it is dated 20 Oct) and **does not contain that flag
as its core**. It touches the same confound in two places: the problem statement notes that
later rounds can enrich sticky, over-expressed or non-specific binders, and the feature menu
offers a developability and polyreactivity filter as a side feature.

So there are two readings, and the team needs to pick one:
- **(a)** TrueHit ATG is enrichment prediction as written here, and the growth-versus-binding
  flag is one feature inside it (most likely an extension of the developability filter, or a
  confound-aware term in the model).
- **(b)** TrueHit ATG is the growth-versus-binding flag, and this proposal is a separate third
  idea.

Reading (a) is the more likely intent and the safer default, since it has real data and a
defined MVP behind it. **Confirm with Jessica before building.** Build what this document
specifies; treat the growth-versus-binding flag as a planned extension, not the foundation.

---

## 6. Where the ultimate framing is heading

The strongest differentiator is not any single check. It is the output format: a **confidence
score per candidate with reasons attached**, turning a QC tool into a decision tool. Target
output shape:

```
Variant #4471 — 2,100 reads — enriched 40x across 3 rounds
Confidence: 31%
  Structure: clean
  80% of its rise is explained by growth rate, not selection
  Shares a breakpoint pattern with two abundant parents
  Not recommended for ordering
```

A second honest differentiator: **over-correction transparency.** Rather than silently merging
near-identical reads, report what was merged and how confident each merge was, let the user move
the threshold, and show the variant count respond. Existing tools hide this. The known tension
in the literature is that PCR and sequencing errors make true sequences hard to separate by read
count alone, while over-correcting destroys real diversity, and that matters because minimal
sequence differences can affect developability.

Note this framing applies to Idea 1. Idea 2 has its own stated framing in section 5, and the
"honest notes" there (not novel, enrichment is not affinity, validate across rounds) override
anything in this section for that directory.

Two further pain points researched but **not yet agreed with Jessica** — keep as future work
unless the team decides otherwise:
- Cross-sample leak detection (index hopping): run two simulated samples, find variants present
  in both that should not be shared, infer the true sample of origin from abundance. On the
  diversity map a leak appears as a faint echo of one sample inside another.
- Sequencing saturation curve: subsample reads at 10/25/50/100% and plot distinct variants
  recovered. A curve still climbing means the library is undersampled and the reported diversity
  is wrong. Cheap, visual, and answers a question labs currently guess at.

---

## 7. Competitive landscape (be honest about this)

The problem is real and documented, but the space is not empty. Do not claim novelty for the
analysis itself.

- **Geneious Biologics** analyzes millions of raw NGS antibody sequences in minutes, clusters
  sequences, shows diversity plots, and highlights sequence features, assets and liabilities
  across Fab, scFv and VHH. Our developability-flags extra is already standard there.
- **PipeBio** (YC W21) does annotation and liability identification, clustering and sample
  comparison for antibody NGS.
- **Genedata Selector** explicitly markets NGS-based antibody library quality and biopanning
  control, which is close to our exact pitch.
- **vsearch** already does reference-free chimera detection using abundance to distinguish
  chimera from parents.
- **Seq2scFv** is an open source toolkit for display libraries from long-read platforms doing
  V(D)J alignment, scFv delimitation, linker detection and scoring, flagging and counting. This
  overlaps our certain checks directly.

What is actually ours:
1. Reference-free artifact QC as the product, not a feature buried in a platform
2. A chimera suspicion score tuned to antibody libraries
3. The growth-versus-binding flag (Idea 2) — the least served gap
4. A measured benchmark on planted artifacts
5. Open source, against tools that cost thousands per seat

Jessica's own position, stated in Slack: the commercial sites cover what this project does, she
still wants to build it for the skills and understanding, and an open source version remains
useful because those tools are expensive. That framing is sound and also fits a YC-style
sponsor track.

---

## 8. Demo requirements

The demo decides the outcome. Historically, Cal Hacks overall winners were things judges could
watch working live; no sequencing, genomics or protein project has ever won, so the explanation
must be extremely simple.

Demo beats:
1. Setup: "here is a pile of antibody library reads, millions of them, full of errors, no
   answer key."
2. Run it: watch noisy data collapse into clean variants.
3. The key moment: "we planted known artifacts — contamination, truncations, chimeras, and fast
   growers. Here is how many the tool caught." One honest measured number.
4. The visual: the 2D diversity map, junk visibly separated from the real library.
5. Close: "this is the QC step every antibody team does by hand today. We automated it."

Pitch line to lead with, skipping the biology: drug companies spend weeks screening antibody
libraries to find new medicines; if the library is broken, all that work is wasted; we check it
first, in minutes, and show exactly what is wrong.

**Never quote an accuracy number that has not actually been measured.** A placeholder like
"48 of 50 chimeras" must not survive into the pitch unmeasured.

---

## 9. Stack

- Python for everything analytical: PyTorch, scikit-learn, Biopython, pandas, numpy
- UMAP or scikit-learn t-SNE for the diversity map
- Backend: FastAPI, websocket for streaming results to the dashboard
- Frontend: React or Streamlit, whichever ships faster. Streamlit is the safe choice given a
  2-person team
- Clustering: Starcode, MMseqs2 or VSEARCH as external binaries
- Germline assignment: ANARCI and/or IgBLAST as the rule-based baseline to beat
- Optional: AbLang or AntiBERTy antibody language model embeddings plus a small classifier.
  AbLang representations have been shown to cluster sequences by V-gene and mutation count
- Laptop CPU is sufficient for a few thousand variants. A GPU helps for language model
  embeddings but is not required

For `true-hit-atg/` specifically, the stack is smaller: pandas and numpy for the enrichment
tables, scikit-learn for logistic and linear regression, one-hot encoding of sequences, and
PyTorch only if the small CNN stretch happens. Scipy for rank correlation when comparing
predictions against measured affinities.

Vendor everything downloadable (the Paul et al. per-round tables, OAS subset, germline
references, model weights, binaries) before relying on venue wifi.

---

## 10. What Claude Code should do first

1. Initialize the repo structure in section 2 with both directories and a short root README.
2. In each directory: a Python package skeleton, `requirements.txt` (or `pyproject.toml`), a
   `data/` folder, and a `README.md` restating that directory's problem in a few lines.
3. **For `true-hit-atg/`, do this first**: clone `debbiemarkslab/ML_affinity_maturation` and
   verify it ships the processed per-round tables, not just code. The whole MVP depends on
   that. Vendor the tables into `true-hit-atg/data/` so the build does not need wifi. Then
   define the shared table format (variant, per-round counts, enrichment value) as a typed
   schema both layers import, since that is what unblocks parallel work.
4. Then in `true-hit-atg/`: enrichment ratio computation and ranking, then one-hot encoding plus
   logistic or linear regression trained on an early round and tested on a later one, then the
   comparison against a naive baseline, then the ranked-candidate readout.
5. For `true-hit-qc/`, build the data generator first. Nothing can be measured until there is an
   answer key. The simulator and the variant-table schema in section 4 are the two things that
   unblock parallel work there.
6. Then the certain checks, then end-to-end wiring, then the map, then chimera scoring.
7. Keep the clustering identity threshold, error rates, chimera rate, count thresholds,
   normalization method and library size as configuration, not constants.
8. Every artifact flag must carry a machine-readable code and a human-readable reason string.
   The reason strings are the product.
9. In `true-hit-atg/`, never name a model output "affinity" or "binding affinity" anywhere,
   including variable names, column headers and UI labels. Use "enrichment" or "predicted
   enrichment". This is an explicit constraint from the proposal.

## 11. Open questions still to settle with Jessica

Shared:
1. Which of the two ideas is the primary submission, and whether the second is being built at
   all given a 2-person team
2. Reading (a) versus (b) on the TrueHit ATG discrepancy in section 5
3. Exactly what counts as the MVP being done at the halfway mark
4. Live streaming run versus a pre-computed run replayed on screen
5. Who owns integration between the workstreams — currently nobody

Idea 1 (`true-hit-qc/`):
6. Short reads or long reads for the simulated data
7. Library format for the demo: scFv or nanobody
8. Library size and read count that keep the demo laptop-fast
9. Realistic error, chimera and contamination rates
10. Which clustering tool she already uses
11. Where chimera breakpoints tend to fall

Idea 2 (`true-hit-atg/`):
12. Does the Paul et al. repo actually ship processed per-round tables (day-one blocker)
13. Which campaign to start with. B7 is richest (three FACS rounds plus polyreactivity
    counter-selection), RX002 has measured affinities for validation
14. Which round pair to train and test on
15. Count threshold and normalization choices for the enrichment computation
16. Whether the align and count front end is in scope at all, or dropped per her own note
17. Whether a no-target or counter-selection arm can carry the growth-versus-binding extension
    (B7's polyreactivity counter-selection may already serve this)

## 12. Sources

- Observed Antibody Space: https://opig.stats.ox.ac.uk/webapps/oas
- AbLang: https://pmc.ncbi.nlm.nih.gov/articles/PMC9710568
- AntiBERTy: https://arxiv.org/abs/2112.07782v1
- vsearch: https://manpages.org/vsearch
- Phage display false positives and propagation advantage: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6259164/
- NGS reveals corruption of a phage display selection: https://pmc.ncbi.nlm.nih.gov/articles/PMC11431649
- Parasitic sequences and diversity collapse: https://pmc.ncbi.nlm.nih.gov/articles/PMC7759992
- PhD7Faster 2.0: https://peerj.com/articles/7131
- Dual UMIs remove chimeras and artifacts: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8727365/
- Index hopping in antibody library sequencing: https://www.utupub.fi/bitstreams/c372a0e4-7e8f-453d-b7a1-d952747c21a4/download
- Seq2scFv: https://www.biorxiv.org/content/10.1101/2024.07.04.602016.full.pdf
- Antibody library cloning and NGS QC (NAR 2025): https://pubmed.ncbi.nlm.nih.gov/41099698/
- Geneious Biologics: https://www.geneious.com/features/antibody-ngs-data-analysis
- PipeBio: https://pipebio.com/platform
- Genedata NGS antibody library QC: https://www.genedata.com/knowledge/details/deep-sequencing-of-antibody-libraries
- Whimsical board (pipeline and open problems): https://whimsical.com/cal-hacks-13-antibody-library-qc-pipeline-7Z4Y9jJZPLtFqbuyjyWkz6
- Idea 1 proposal doc: https://claude.ai/artifact/CoZqSjFq6vWpBKoT1sXcjQ
- Jessica's TrueHit ATG proposal, "ML-Guided Nanobody Affinity Maturation", 20 Oct 2026:
  https://claude.ai/code/artifact/002ddc1a-6e10-4a3e-99d4-aeaf38275550
- Idea 2 dataset and reference method, Paul et al. (2026):
  https://github.com/debbiemarkslab/ML_affinity_maturation
