# TrueHit ATG — a walkthrough

Hi Jessica. This is the one document to read if you want to understand what got built and run
it yourself. It assumes you know antibodies and selection, and assumes nothing about Python.

Everything here is measured. Where a number appears, it comes from a real run and
`claude-docs/context.md` says where.

---

## 1. What it does

You give it the per-round sequencing counts from an affinity-maturation campaign. It works out
how much each variant rose from one round to the next, learns what it is about a variant's
**sequence** that makes it rise, and then ranks variants — including ones the campaign never
contained.

The point is to stop a campaign a round early. If a model trained on round 1 can tell you what
would have enriched by round 2, you do not have to run round 2.

This is an open implementation of **Paul et al. (2026)**, the nanobody affinity-maturation
paper. The method is theirs. What is ours is the app, the ability to run it on other people's
data, and our own checking.

---

## 2. The idea in four steps

| Step | What happens | In your language |
|---|---|---|
| 1 | Counts become enrichment | A variant's share of the pool in the later round divided by its share in the earlier one, as a log ratio. Normalised to counts per million with a 1 CPM floor. |
| 2 | Enrichment becomes a label | The log ratios are strongly bimodal. A two-component mixture splits them into **enriched** and **depleted**, and *abstains* where it is less than 80% sure — those variants are dropped rather than guessed. |
| 3 | Sequence becomes numbers | Each variant is one-hot encoded by position. The parent's own residue is dropped at every position, so each feature means "position 47 carries something the parent does not" — a substitution indicator. |
| 4 | Numbers become a ranking | An L1-penalised logistic regression is fitted on the early round and asked about a later one. L1 means most weights go to zero, so what survives is a short, readable list of substitutions. |

Steps 1 and 2 are the data-and-enrichment layer from your proposal. Steps 3 and 4 are the ML
layer. The table between them is a fixed contract, which is what let them be built separately.

A run takes a few seconds on a laptop. No GPU, nothing to wait for.

---

## 3. What is in the folder

### The pipeline, one file per step

| File | Does |
|---|---|
| `truehit_atg/enrich.py` | Step 1 and 2 — normalising, log ratios, the mixture labelling, the train/test split |
| `truehit_atg/encode.py` | Step 3 — turning sequences into numbers, and the parent-column drop |
| `truehit_atg/model.py` | Step 4 — fitting, scoring, and the metrics |
| `truehit_atg/pipeline.py` | Runs the four steps in order and hands back one result object |

### Getting data in

| File | Does |
|---|---|
| `truehit_atg/schema.py` | The contract: which columns must exist, and the checks that refuse bad input |
| `truehit_atg/data.py` | Loads the three campaigns from the paper |
| `truehit_atg/ingest.py` | Loads somebody else's CSV, once someone says which column is which |
| `truehit_atg/agent.py` | Asks a language model to work out which column is which, so nobody has to |

### Everything else

| File | Does |
|---|---|
| `app.py` | The web interface. The only file that knows a screen exists. |
| `data/paul2026/` | The paper's data for all three campaigns, committed, so no downloads |
| `tests/` | 142 automated checks, including that we reproduce the paper's published numbers |
| `truehit_atg/config.py` | Every tunable in one place — which campaign, which rounds, penalty strength |

One deliberate choice: **nothing inside `truehit_atg/` imports anything to do with the
interface.** The app only calls one function and displays what comes back. That is why a proper
website can be put on top later without touching the analysis.

---

## 4. Running it

You need **Python 3.10 or newer** and git. Nothing else — the data comes with the clone.

### macOS or Linux

```bash
git clone https://github.com/heller1729/true-hit-atg
cd true-hit-atg/true-hit-atg

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m pytest -q          # should say 142 passed
.venv/bin/python -m streamlit run app.py
```

### Windows

```bash
git clone https://github.com/heller1729/true-hit-atg
cd true-hit-atg\true-hit-atg

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m streamlit run app.py
```

A browser tab opens by itself. If it does not, the terminal prints a `localhost` address — open
that.

**Run the commands from inside the inner `true-hit-atg` folder.** The repository and the folder
share a name, so there are two levels with the same name; the one you want is the one containing
`app.py`.

**No API key is needed.** A file that already uses the expected column names is read directly,
with no model call. The model is only offered as a fallback, on the "Onboard a dataset" tab,
when a file's columns are named something else.

Once it is up: pick **AT110** in the sidebar and press **Run pipeline**.

---

## 5. What you are looking at

The app names the rounds **Input**, **Output 1**, **Output 2** and so on, by position rather
than by assay. One line on the Enrichment tab gives the mapping — "Input is the MACS round;
Output 1 is FACS1" — so you can still tell which sort produced which column. The numbering is
positional, so MACS is Input in AT110 but Output 1 in NbB7, which has a naive round before it.

| Tab | Shows | The thing to notice |
|---|---|---|
| **Enrichment** | Two expanders explaining the arithmetic in plain language, a histogram of enrichment coloured by label, and the biggest risers listed by their substitutions | Two clear humps — losers near −2.5, winners near +2. This picture is the whole project in one image, and it is why the task is a yes/no rather than a regression. The grey spike is the abstentions. |
| **Model** | How well it predicts | The headline is ROC-AUC. Plain version: pick one enriched and one depleted variant at random; this is how often the model scores the enriched one higher. 0.5 is a coin flip, 1.0 is perfect. The default trains and tests on the same round pair, which is what a real campaign gives you, and scores **0.922** on AT110. Pick a later round under "Test on" and the tab adds the full statistic set, including the comparison against the paper. |
| **Candidates** | Every variant ranked by predicted enrichment | Downloadable. Predicted enrichment is a probability from 0 to 1, not a predicted count and not an affinity; Model call is that probability thresholded at 0.5. The Data split column says whether the model was fitted on that variant, and filters the list down to the ones it was not. |
| **Substitution preferences** | A heatmap of position against substituted residue | This falls straight out of the fitted model. Because each feature is "this position carries something other than the parent", each weight reads directly as a preference for that substitution. Worth your eye — do the strong ones sit where you would expect? |
| **Score sequences** | A box to paste in sequences the campaign never contained | This is the part counts cannot do. An unsequenced clone has no enrichment to rank by, but it does have a sequence. |
| **Onboard a dataset** | Upload somebody else's CSV | A file with the expected column names is read directly. If the names are different, a language model can be asked to work out the mapping — that part, and only that part, needs an API key. Once a file is read, it appears in the sidebar's dataset list and the other five tabs render it. |

### Does it actually reproduce the paper?

Yes, on all four numbers the paper publishes for this model. The test suite asserts them, so
they cannot quietly drift:

| Campaign | our ROC | paper | our PR | paper |
|---|---|---|---|---|
| AT110 | 0.960 | 0.964 | 0.942 | 0.949 |
| RX002 | 0.952 | 0.935 | 0.952 | 0.947 |

All four land within 0.017. We also refitted their logistic model and compared the actual
coefficients: of the weights that survived, **every sign agreed** and the correlation was 0.983.

---

## 6. What we found that you should check

These are the honest problems. Several are biology questions more than code questions, which is
why they are addressed to you.

**"Enriched" largely means "absent from the input round."** 88% of the enriched AT110 test
variants have a MACS count of exactly **zero**, against 0% of the depleted ones. The mechanism is
plain: a variant missing from the input gets floored at 1 CPM, so any later appearance produces a
big positive ratio. Is that real selection of something rare, sampling noise at this depth, or
both? This is the single thing most worth your judgement, and it is the first question a sharp
judge would ask.

**The obvious baseline beats the model, and the comparison is rigged.** Ranking by the observed
counts gets 0.989 against the model's 0.960 — but the label is computed *from* those counts, so
it is circular. The fair comparison holds mutation load fixed, where counting substitutions has
no ability to separate anything (0.500 by construction) and the model gets **0.945**. That
comparison is the one that survives questioning, and it is on screen in the Model tab.

**Single substitutions cannot be tested with this data.** AT110 has 3 enriched single mutants
among 99 test rows; RX002 has 2 of 166. Both too few to measure anything. That is exactly the
regime your RX002 BLI measurements live in — which makes them not just a nice validation but the
**only** way to test that case at all. If you can get Supplementary Table 4 transcribed, it
becomes a real result.

**NbB7's FACS3 is a polyreactivity counter-selection.** So it screens stickiness, not growth
rate, and cannot act as the control arm for the growth-versus-binding idea from the 23 Oct call.
Nothing in this dataset isolates replication advantage. If we want that, it needs a different
dataset or an explicit term in the model.

**NbB7 is not a good demo campaign.** 85–96% of its FACS3 pairs are positive, so a model can
score well there by calling everything enriched. AT110 is the one to lead with.

---

## 7. What is not built, and not claimed

- **Enrichment has only ever run on this paper's data.** We ran the pipeline on two outside
  datasets, but both arrived already scored, so only the encoding and the model were exercised.
  If someone asks whether the enrichment code has been tested on anything but Paul et al., the
  answer is no.
- **Nothing here is wet-lab tested.** Candidates are proposals.
- **The method is not ours.** It is an open implementation of a published one.
- **It predicts enrichment, not affinity.** Enrichment is what survives selection, which is
  confounded by expression and stickiness. You know the difference better than anyone on this
  team; please flag it anywhere the wording slips.

One thing that did work outside the paper: a yeast enzyme assay with no connection to antibodies
ran through the same code at **0.959**. The pipeline is general. What it needs is variants
carrying several substitutions that recur across variants — a one-variant-per-substitution scan
cannot be learned at all, and we can show exactly why.

---

## 8. If you want more detail

| Document | For |
|---|---|
| `README.md` | The developer view — how the reproduction was verified, what the baselines mean |
| `claude-docs/context.md` | Everything, at length: decisions, measurements, open questions |
| `data/DATASET.md` | What the data is, what it does *not* contain, and which campaign to use for what |

Open questions still needing your call are in section 9 of `context.md`. The two that matter
most: where the sequences for the "score outside clones" demo should come from, and your read on
the input-round confound above.
