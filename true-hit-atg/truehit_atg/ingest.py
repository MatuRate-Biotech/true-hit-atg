from dataclasses import dataclass, field

import pandas as pd

from . import encode, enrich, schema as s


# what a converter produces: renames and round ordering, which is what most real files need.
# serializable, diffable, and carrying no executable code, which is what makes it safe for an
# agent to write in step 4.
@dataclass(frozen=True)
class Spec:
    var: str
    seq: str
    rnds: dict
    drop_gaps: bool = False
    note: str = ""
    # a dataset that arrives already scored has no counts, so enrichment does not run
    enr_col: str = ""
    enr_bin_col: str = ""


# the Spec for a file that already uses the contract's own column names. round order is
# column order, so the first cnt_* column is Input. apply_spec then validates it through
# chk_raw, which is what decides whether the file really conforms.
def identity_spec(cols):
    pre = s.cnt("")
    rnds = {c[len(pre):]: c for c in cols if c.startswith(pre)}
    return Spec(
        var=s.VAR,
        seq=s.AA_ALN,
        rnds=rnds,
        enr_col=s.ENR if s.ENR in cols else "",
        enr_bin_col=s.ENR_BIN if s.ENR_BIN in cols else "",
        note="column names already match the contract",
    )


def probe(path, n=20):
    df = pd.read_csv(path, nrows=n)
    full = pd.read_csv(path, usecols=[0])
    return {
        "path": str(path),
        "n_rows": int(len(full)),
        "cols": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head": df.head(min(n, 5)).to_dict("records"),
    }


def apply_spec(path, spec, n_pos=None):
    df = pd.read_csv(path)
    want = {spec.var: s.VAR, spec.seq: s.AA_ALN}
    want.update({c: s.cnt(r) for r, c in spec.rnds.items()})
    for c in (spec.enr_col, spec.enr_bin_col):
        if c:
            want[c] = c
    miss = [c for c in want if c not in df.columns]
    if miss:
        raise ValueError(f"source columns not in {path}: {miss}; file has {list(df.columns)}")

    out = df[list(want)].rename(columns=want).copy()
    out[s.AA_ALN] = out[s.AA_ALN].astype(str).str.strip().str.upper()
    if spec.drop_gaps:
        out[s.AA_ALN] = out[s.AA_ALN].str.replace(encode.GAP, "", regex=False)
    out[s.VAR] = out[s.VAR].astype(str)
    if spec.enr_col:
        out[s.ENR] = df[spec.enr_col].astype(float)
    if spec.enr_bin_col:
        out[s.ENR_BIN] = df[spec.enr_bin_col].astype(float)
    return s.chk_raw(out, list(spec.rnds), n_pos=n_pos, alph=encode.ALPH)


def rnds_of(raw):
    pre = len(s.cnt(""))
    return [c[pre:] for c in raw.columns if c.startswith(s.cnt(""))]


def scored(raw):
    return s.ENR in raw.columns


# a pre-scored upload skips enrichment entirely: there are no counts to normalize and no
# ratio to take. only the split and the labels are derived here.
def from_scored(raw):
    out = raw.copy()
    out[s.AA] = out[s.AA_ALN].str.replace(encode.GAP, "", regex=False)
    if s.CDR3 not in out.columns:
        out[s.CDR3] = ""
    src = "supplied"
    if s.ENR_BIN not in out.columns:
        out[s.ENR_BIN] = enrich.binarize(out[s.ENR], out[s.ENR].notna())
        src = "gmm"
    out[s.SPLIT] = enrich.make_split(out[s.ENR_BIN])
    return s.chk(out, (), n_pos=int(out[s.AA_ALN].str.len().iloc[0])), src


def to_internal(raw, pair, camp=None):
    rnds = rnds_of(raw)
    late, early = pair
    for r in (late, early):
        if r not in rnds:
            raise ValueError(f"round {r!r} has no counts; upload has {rnds}")

    out = raw.copy()
    # chk() wants the full internal contract. an upload carries neither of these, and the
    # vendored path takes them from upstream columns, so derive them here rather than
    # loosening the contract.
    out[s.AA] = out[s.AA_ALN].str.replace(encode.GAP, "", regex=False)
    if s.CDR3 not in out.columns:
        out[s.CDR3] = ""
    for r in rnds:
        out[s.norm(r)] = enrich.norm_ct(out, r, camp)
        out[s.has(r)] = enrich.has_read(out, r)

    out = enrich.pair(out, late, early, camp)
    return s.chk(out, rnds, n_pos=int(out[s.AA_ALN].str.len().iloc[0]))
