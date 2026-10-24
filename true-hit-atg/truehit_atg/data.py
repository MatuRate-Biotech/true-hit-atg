from pathlib import Path

import pandas as pd

from . import schema as s
from .config import CFG

ROOT = Path(__file__).resolve().parent.parent / "data" / "paul2026"


def load_raw(camp):
    d = ROOT / camp
    x = pd.read_csv(d / "X.csv")
    y = pd.read_csv(d / "Y.csv", low_memory=False)
    return x.merge(y, on=["seq_ID", "index"], validate="one_to_one")


def load_wt(camp):
    return pd.read_csv(ROOT / camp / "aligned_WT_sequence.csv")


def wt_seq(camp):
    wt = load_wt(camp)
    pos = wt.columns[13:]
    return "".join(wt.loc[0, pos].astype(str))


def load(camp=None, pair=None, min_ct=None, min_ct_rnd=None):
    camp = camp or CFG.camp
    c = s.CAMPS[camp]
    late, early = pair or CFG.train
    if (late, early) not in s.pairs(camp):
        raise ValueError(f"{camp} has no round pair {late}/{early}, rounds are {c.rnds}")

    raw = load_raw(camp)
    out = pd.DataFrame(
        {
            s.VAR: raw["seq_ID"],
            s.AA: raw["FullSeq_nogaps"],
            s.AA_ALN: raw["FullSeqs_withgaps"],
            s.CDR3: raw["CDR3_nogaps"],
            s.ENR: raw[s.up_ratio(late, early)],
            s.ENR_BIN: raw[s.up_enr(late, early)],
            s.SPLIT: raw[s.up_split(late, early)],
        }
    )
    for r in c.rnds:
        out[s.cnt(r)] = raw[s.up_cnt(r)]
        out[s.norm(r)] = raw[s.up_norm(r)]
        out[s.has(r)] = raw[f"{r}_hasread"]

    mc = CFG.min_ct if min_ct is None else min_ct
    if mc:
        r = min_ct_rnd or CFG.min_ct_rnd or early
        out = out[out[s.cnt(r)] >= mc].reset_index(drop=True)

    return s.chk(out, c.rnds, c.n_pos)


# enr is defined for every row, but enr_bin and split are only set for the subset the
# upstream pipeline considered callable for that pair. train/test comes from upstream.
def labeled(df):
    return df[df[s.SPLIT].notna()].reset_index(drop=True)


def split(df, which):
    return df[df[s.SPLIT] == which].reset_index(drop=True)
