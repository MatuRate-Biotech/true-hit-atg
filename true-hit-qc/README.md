# TrueHit QC

Reference-free artifact QC on an antibody library.

A lab builds or receives a library of antibody variants and sequences it. The data comes back
full of molecules that look like real, novel candidates but are artifacts: machine typos,
variants cut short during construction, two variants fused into one by PCR, molecules from
another sample entirely, badly assembled frameworks. Mistake one for a hit and someone orders
it, expresses it and tests it — weeks and real money spent on a molecule that never existed.

Matching against a reference list solves this when you designed every sequence yourself. For a
naive or affinity-matured library there is no list, and today the gap is closed by clustering
and then eyeballing the output. This directory closes it with a measurable check instead.

### Pipeline

```
raw reads
  -> 1  cluster and dereplicate      existing tools, not the contribution
  -> 2  certain checks               contamination, framework, truncation
  -> 3  chimera suspicion score      population statistics
  -> 4  report                       trusted variant list, diversity map, health metrics
```

Step 2 is rule-based and unambiguous: wrong format for this library, no germline match or a
broken conserved residue, length and coverage short of the expected region layout.

Step 3 is the hard one. In a diverse library a fusion of two members can look exactly like a
legitimate new variant, and the sequence alone cannot tell you. So the score comes from
population context: the variant is rare, its front matches one abundant variant and its back
matches a different one, and the switch point sits where fusions tend to occur. The output is a
probability with named parents — `92% likely chimera, parents v12 and v88` — never a yes/no,
because some real variants will look like this by chance.

### Two things that make this honest

**Reasons, not verdicts.** Every flag carries a machine-readable code and a sentence naming the
numbers behind it. The sentences are the product; they are defined in one place, `RSNS` in
`truehit_qc/schema.py`, and `chk()` refuses a table with a flagged variant and no reason.

**Over-correction transparency.** Clustering near-identical reads into one variant is where
real diversity quietly dies — in an affinity-matured library two real variants can differ by a
single letter, which edit distance cannot tell from a sequencing typo. So report what was
merged and how confident each merge was, expose the identity threshold (`Qc.ident`), and let
the variant count respond when the user moves it. Existing tools hide this.

### Data

Simulated by us, with artifacts planted deliberately, so there is an answer key and a real
accuracy number. The ground-truth library is built from real antibody sequences — pull a subset
from [OAS](https://opig.stats.ox.ac.uk/webapps/oas) for germline and species labels, then
construct formats ourselves, since OAS holds single chains:

- nanobody — camelid single domain, ~120 aa
- scFv — heavy variable + `(G4S)x3` linker + light variable
- Fab — heavy variable + CH1, light variable + CL
- junk — shuffled sequence and random non-antibody protein

`data/raw/` holds generated reads, `data/ref/` germline references (IMGT), `data/out/` results.
All three are gitignored except for their `.gitkeep`. **Download the OAS subset and the germline
references before the venue**, and commit them under `data/ref/` — venue wifi is not a
dependency worth taking.

Long reads are the default: an scFv is ~750 DNA bases, so 150–300 bp short reads cannot span a
full variant without a merging step that buys nothing for the demo. Confirm with Jessica.

### The handoff contract

`truehit_qc/schema.py` defines the variant table — one row per cluster:

| Column | Meaning |
|---|---|
| `var_id` | cluster id |
| `dna` | consensus DNA sequence |
| `aa` | translated protein sequence |
| `cnt_r1`, `cnt_r2`, … | read count per round |
| `qc_flag` | `clean` / `chimera` / `truncated` / `contaminant` / `framework_error` |
| `qc_reason` | the sentence, built from `RSNS` |
| `fmt`, `fmt_conf` | detected format and confidence |
| `germ`, `species` | assigned germline and species |
| `chim_score` | chimera suspicion probability |
| `chim_parents` | parent `var_id`s when flagged |
| `true_src` | planted ground truth, simulated data only |

Format is reported as a distribution over `fmt` with a confidence, not a binary flag, because a
mixed-format library is a real thing and the scientist decides whether a mix is contamination
or intentional.

Every tunable lives in `truehit_qc/config.py`: clustering identity threshold, error and chimera
and contamination rates, library size, read count, count thresholds, germline identity floor.
None of them belong inline in the analysis code.

### Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python -m pytest tests -q
```

External binaries, not pip-installable, needed for step 1 and the germline baseline — install
them before the venue: one of Starcode, MMseqs2 or VSEARCH for clustering, and ANARCI or IgBLAST
for germline assignment. VSEARCH also gives us `uchime_denovo`, the reference-free chimera
detector that is our baseline to beat, not our contribution.

### Next

1. The data generator, first. Nothing is measurable until there is an answer key, and the
   simulator plus the schema above are what let the two halves be built in parallel.
2. The three certain checks.
3. End to end on one input with a basic readout.

Then, in order: chimera suspicion scoring, the 2D diversity map, developability flags and
diversity metrics, a demo UI.

The diversity map is the one picture a non-biologist judge reads instantly — featurize each
variant, reduce to two dimensions, plot every variant colored by artifact status. Real variants
form tight groups, contaminants sit alone, chimeras land between the two groups they were fused
from. Treat it as the most important demo asset after the measured number.

### Open, and not ours to guess

Library size and read count that keep the demo laptop-fast; realistic error, chimera and
contamination rates; where chimera breakpoints actually fall; scFv or nanobody for the demo;
which clustering tool Jessica already uses. The placeholder rates in `config.py` are marked as
placeholders. **Never quote an accuracy number that has not been measured.**
