from dataclasses import dataclass

VAR = "var_id"
DNA = "dna"
AA = "aa"
FLAG = "qc_flag"
RSN = "qc_reason"
FMT = "fmt"
FMT_CONF = "fmt_conf"
GERM = "germ"
SPEC = "species"
CHIM = "chim_score"
PAR = "chim_parents"
TRUE = "true_src"

CORE = (VAR, DNA, AA, FLAG, RSN, FMT, FMT_CONF, GERM, SPEC, CHIM, PAR)

CLEAN = "clean"
CHIMERA = "chimera"
TRUNC = "truncated"
CONTAM = "contaminant"
FW_ERR = "framework_error"

FLAGS = (CLEAN, CHIMERA, TRUNC, CONTAM, FW_ERR)

FMT_SCFV = "scfv"
FMT_FAB = "fab"
FMT_NB = "nanobody"
FMT_JUNK = "junk"
FMT_UNK = "unknown"

FMTS = (FMT_SCFV, FMT_FAB, FMT_NB, FMT_JUNK, FMT_UNK)

# every flag carries a code and a sentence. the sentences are the product, so they name
# numbers and parents rather than restating the code.
RSNS = {
    "ok": "passed format, framework and length checks",
    "chim_parents": "first {bp} aa match {pa} ({ca} reads), remainder matches {pb} ({cb} reads); {cnt} reads of its own",
    "chim_bp": "breakpoint at {bp} sits in the region fusions favour during prep",
    "trunc_len": "{obs} aa against {exp} expected for {fmt}",
    "trunc_reg": "sequence stops inside {reg}, {miss} is absent",
    "contam_fmt": "{dom} domain(s) at {obs} aa reads as {got}, library is {fmt}",
    "contam_linker": "carries a {got} linker, absent from every {fmt} member",
    "fw_no_germ": "best germline hit {germ} at {idt:.0%} identity, below the {thr:.0%} floor",
    "fw_cys": "conserved cysteine missing at position {pos}",
    "fw_motif": "framework motif {want} replaced by {got} at position {pos}",
    "dev_deamid": "deamidation motif {mot} at position {pos}",
    "dev_glyc": "N-glycosylation sequon {mot} at position {pos}",
    "dev_oxid": "oxidation-prone {res} at position {pos}",
}


def rsn(key, **kw):
    return RSNS[key].format(**kw)


def cnt(rnd):
    return f"cnt_r{rnd}"


@dataclass(frozen=True)
class Layout:
    fmt: str
    regs: tuple
    aa_lo: int
    aa_hi: int
    dom: int
    linker: str


# region layouts the truncation and contamination checks measure against. lengths are
# the usual ranges, not hard limits; Jessica to confirm the bounds.
LAYOUTS = {
    FMT_NB: Layout(FMT_NB, ("vhh",), 110, 135, 1, ""),
    FMT_SCFV: Layout(FMT_SCFV, ("vh", "linker", "vl"), 230, 280, 2, "GGGGSGGGGSGGGGS"),
    FMT_FAB: Layout(FMT_FAB, ("vh", "ch1", "vl", "cl"), 420, 480, 4, ""),
}


def chk(df, rnds=()):
    want = list(CORE) + [cnt(r) for r in rnds]
    miss = [c for c in want if c not in df.columns]
    if miss:
        raise ValueError(f"variant table missing: {miss}")
    bad = set(df[FLAG].dropna()) - set(FLAGS)
    if bad:
        raise ValueError(f"unknown qc_flag: {sorted(bad)}")
    bad = set(df[FMT].dropna()) - set(FMTS)
    if bad:
        raise ValueError(f"unknown fmt: {sorted(bad)}")
    if df[VAR].duplicated().any():
        raise ValueError("duplicate var_id")
    flagged = df[FLAG] != CLEAN
    if (df.loc[flagged, RSN].fillna("") == "").any():
        raise ValueError("flagged variant with no reason string")
    return df
