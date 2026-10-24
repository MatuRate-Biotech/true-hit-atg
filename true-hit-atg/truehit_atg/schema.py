from dataclasses import dataclass

VAR = "var_id"
AA = "aa"
AA_ALN = "aa_aln"
CDR3 = "cdr3"
ENR = "enr"
ENR_BIN = "enr_bin"
SPLIT = "split"

CORE = (VAR, AA, AA_ALN, CDR3, ENR, ENR_BIN, SPLIT)


def cnt(r):
    return f"cnt_{r}"


def norm(r):
    return f"norm_{r}"


def has(r):
    return f"has_{r}"


def up_cnt(r):
    return f"{r}_ct"


def up_norm(r):
    return f"{r}_normct"


def up_log(r):
    return f"{r}_logct"


def up_ratio(late, early):
    return f"{late}_{early}_logratio"


def up_enr(late, early):
    return f"{late}_{early}_enriched"


def up_split(late, early):
    return f"{late}_{early}_split"


@dataclass(frozen=True)
class Camp:
    name: str
    tgt: str
    rnds: tuple
    n_pos: int


# n_pos is the width of FullSeqs_withgaps, fixed per campaign, so aa_aln one-hots directly
CAMPS = {
    "AT110": Camp("AT110", "AT1R", ("MACS", "FACS1", "FACS2"), 126),
    "NbB7": Camp("NbB7", "beta2AR", ("naive", "MACS", "FACS1", "FACS2", "FACS3"), 125),
    "RX002": Camp("RX002", "RXFP1", ("MACS", "FACS1", "FACS2"), 122),
}


# pre-filter library sizes, recovered exactly from the shipped norm_* columns. upstream
# normalized to CPM against the library before count filtering, and that total is not in the
# tables. an upload has only its own column sum, which shifts every log ratio by a constant
# and so leaves the ranking untouched.
LIBS = {
    "AT110": {"MACS": 136075, "FACS1": 150367, "FACS2": 285531},
    "NbB7": {"naive": 226146, "MACS": 241225, "FACS1": 230677, "FACS2": 168989, "FACS3": 1524777},
    "RX002": {"MACS": 922141, "FACS1": 554585, "FACS2": 430342},
}


def pairs(camp):
    r = CAMPS[camp].rnds
    return [(r[j], r[i]) for i in range(len(r)) for j in range(i + 1, len(r))]


RAW_CORE = (VAR, AA_ALN)


# the contract for an upload, which sits upstream of enrichment: no enr, enr_bin, split or
# norm_* yet, because enrich.py computes those. names offenders rather than coercing.
def chk_raw(df, rnds, n_pos=None, alph=None):
    miss = [c for c in RAW_CORE if c not in df.columns]
    if miss:
        raise ValueError(f"upload missing: {miss}")
    have = [r for r in rnds if cnt(r) in df.columns]
    if len(have) < 2:
        raise ValueError(
            f"need counts for at least two rounds, found {len(have)}; "
            f"expected some of {[cnt(r) for r in rnds]}"
        )
    if df[VAR].duplicated().any():
        raise ValueError(f"duplicate var_id: {df.loc[df[VAR].duplicated(), VAR].head(5).tolist()}")

    w = df[AA_ALN].str.len()
    exp = n_pos if n_pos is not None else (int(w.iloc[0]) if len(w) else 0)
    if not w.eq(exp).all():
        bad = df.loc[w != exp, VAR].head(5).tolist()
        raise ValueError(
            f"aa_aln must be {exp} long for every row, found widths {sorted(w.unique())}; "
            f"first offenders: {bad}. one-hot columns only line up on equal-length input, "
            f"so padding or trimming here would scramble positions"
        )

    if alph:
        ok = set(alph)
        bad = [v for v, q in zip(df[VAR], df[AA_ALN]) if set(q) - ok][:5]
        if bad:
            sym = sorted({c for q in df[AA_ALN] for c in q} - ok)
            raise ValueError(f"unknown symbols {sym} in aa_aln; first offenders: {bad}")

    for r in have:
        col = df[cnt(r)]
        if col.dtype.kind not in "iu":
            raise ValueError(f"{cnt(r)} must be whole numbers, got dtype {col.dtype}")
        if (col < 0).any():
            raise ValueError(
                f"{cnt(r)} has negative counts; first offenders: "
                f"{df.loc[col < 0, VAR].head(5).tolist()}"
            )
    return df


def chk(df, rnds, n_pos=None):
    want = list(CORE) + [c for r in rnds for c in (cnt(r), norm(r), has(r))]
    miss = [c for c in want if c not in df.columns]
    if miss:
        raise ValueError(f"variant table missing: {miss}")
    if df[VAR].duplicated().any():
        raise ValueError("duplicate var_id")
    w = df[AA_ALN].str.len()
    if n_pos is not None and not w.eq(n_pos).all():
        bad = df.loc[w != n_pos, VAR].head(5).tolist()
        raise ValueError(
            f"aa_aln must be {n_pos} long for every row, found {sorted(w.unique())}; "
            f"first offenders: {bad}. one-hot columns only line up on equal-length input, "
            f"so padding or trimming here would scramble positions"
        )
    if n_pos is None and w.nunique() > 1:
        raise ValueError(f"aa_aln lengths are ragged: {sorted(w.unique())[:10]}")
    return df
