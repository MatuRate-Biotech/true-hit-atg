# TrueHit

Antibody library work for Cal Hacks 13.0. Two independent projects, one directory each.

### `true-hit-atg/` — TrueHit ATG (headline)

ML-guided nanobody affinity maturation. Takes per-round sequencing data from an affinity
maturation campaign, computes how much each variant enriched round over round, and trains a
model to predict that enrichment so candidates can be ranked after one round instead of
running the full campaign.

Runs on the published dataset from Paul et al. (2026),
[`debbiemarkslab/ML_affinity_maturation`](https://github.com/debbiemarkslab/ML_affinity_maturation),
vendored under `true-hit-atg/data/paul2026/`. Three campaigns: AT110, NbB7, RX002.

This is an open implementation of a published method, not a new one. The model predicts
**enrichment** — what survives selection — which is a proxy for binding confounded by
expression and stickiness. It does not predict binding affinity, and nothing in this repo
should say that it does.

### `true-hit-qc/` — artifact QC

Reference-free quality control on a naive antibody library. Clusters raw reads into variants,
then flags the ones that are artifacts rather than real candidates — contamination,
truncations, framework errors, chimeras — each with a reason attached. Runs on data we
simulate ourselves with artifacts planted deliberately, so accuracy is measurable against an
answer key.

### Layout

```
true-hit-atg/WALKTHROUGH.md         start here: plain language, setup, and a tour of the app
claude-docs/context.md              full working context for both ideas
true-hit-atg/                       idea 2, real dataset
true-hit-atg/claude-docs/context.md the same context cut down to idea 2 alone
true-hit-qc/                        idea 1, simulated dataset
```

The two share almost no code and are kept apart on purpose. Each has its own
`requirements.txt` and its own `.venv`.

`claude-docs/` is context for reading before working in a directory, not documentation of it.
Each directory's own `README.md` stays the description of what is there.

### Setup

```bash
cd true-hit-atg && python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
cd true-hit-qc  && python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
```

On macOS or Linux use `.venv/bin/pip`.
