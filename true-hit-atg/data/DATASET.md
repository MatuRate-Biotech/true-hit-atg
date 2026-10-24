# Vendored dataset — Paul et al. (2026)

Source: https://github.com/debbiemarkslab/ML_affinity_maturation
Pinned at upstream commit `96c5b0a23828e3b44498a5822b24e74ae4f88008`, fetched 2026-10-23.
Paper: *Machine learning enables efficient and effective affinity maturation of nanobodies*
(Paul, Harvey, Kollasch, Riesselman, Osei-Owusu, Hollingsworth, Pani, Harper, Kruse, Marks).

Only the `data/` folder is vendored here. The upstream model checkpoints, notebooks and
training scripts were left out; read them from the upstream clone if needed.

## Day-one blocker: resolved

**Yes, the repo ships processed per-round tables, not just code.** Each campaign folder has:

| File | Contents |
|---|---|
| `X.csv` | one row per variant: `seq_ID`, full protein sequence with and without gaps, CDR1/2/3 |
| `Y.csv` | one row per variant: raw count per round, normalized count, log count, log ratio per round pair, binary enriched label, and upstream's train/test split |
| `aligned_WT_sequence.csv` | the starting nanobody, ANARCI-numbered, with germline V and J assignment |

`FullSeqs_withgaps` is a fixed width per campaign, so it one-hot encodes position-wise with no
alignment work on our side.

## What is in each campaign

| Campaign | Target | Rounds | Variants | WT germline | Aligned width |
|---|---|---|---|---|---|
| AT110 | AT1R | MACS, FACS1, FACS2 | 5,257 | IGHV3-66*01 / IGHJ4*01 | 126 |
| NbB7 | β2AR | naive, MACS, FACS1, FACS2, FACS3 | 54,102 | IGHV3-21*01 / IGHJ4*01 | 125 |
| RX002 | RXFP1 | MACS, FACS1, FACS2 | 21,507 | IGHV3-66*01 / IGHJ4*01 | 122 |

## Labeled subset per round pair

`<late>_<early>_logratio` is defined for every row. `<late>_<early>_enriched` and
`<late>_<early>_split` are only set for the subset upstream considered callable for that pair,
so the usable training set is smaller than the variant count. Measured from the vendored files:

| Campaign | Pair | Labeled | Train | Test | Positive |
|---|---|---|---|---|---|
| AT110 | FACS1/MACS | 3,866 | 3,092 | 774 | 29% |
| AT110 | FACS2/MACS | 4,261 | 3,408 | 853 | 35% |
| AT110 | FACS2/FACS1 | 2,587 | 2,069 | 518 | 57% |
| NbB7 | MACS/naive | 4,723 | 3,778 | 945 | 28% |
| NbB7 | FACS1/MACS | 8,032 | 6,425 | 1,607 | 84% |
| NbB7 | FACS2/MACS | 4,280 | 3,424 | 856 | 68% |
| NbB7 | FACS3/MACS | 43,275 | 34,620 | 8,655 | 96% |
| NbB7 | FACS2/FACS1 | 8,647 | 6,917 | 1,730 | 30% |
| RX002 | FACS1/MACS | 16,858 | 13,486 | 3,372 | 52% |
| RX002 | FACS2/MACS | 13,871 | 11,096 | 2,775 | 41% |
| RX002 | FACS2/FACS1 | 14,159 | 11,327 | 2,832 | 40% |

Full table from `pairs()` in `truehit_atg/schema.py`; regenerate with the snippet in the README.

## Two things the dataset does NOT ship

1. **Raw FASTQ files.** Upstream says they were too large to include. The align-and-count front
   end (open question 16) therefore has no raw input here. Drop it, or point it at simulated
   reads from `true-hit-qc/`.
2. **Measured affinities.** No affinity or binding column appears in any vendored file. The
   upstream ESM2 pipeline script references an `AT1R_affinity` dataset that is not in the repo.
   So validating a ranking against measured binding data needs numbers from the paper's
   supplement, not from this download. Treat that feature as blocked until someone finds them.

## Which campaign and round pair to start with

AT110, train on `FACS1/MACS`, test on `FACS2/MACS`. Three reasons:

- It matches upstream's own experiment, whose directories are named
  `Train_FACS1_MACS_*/Test_FACS2_MACS_*` and which ship `Test_scores.csv` plus checkpoints, so
  our numbers can be checked against theirs.
- Train early, test later is the honest validation direction.
- Its labels are not degenerate. NbB7 looks richest by round count, but its FACS3 pairs are
  85–96% positive, so a model there can score well by predicting "enriched" for everything.

RX002 is the balanced alternative (52% positive, 16,858 labeled) and the campaign the paper
associates with designed-variant measurements, so it is the natural second campaign.
