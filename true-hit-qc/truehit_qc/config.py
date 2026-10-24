from dataclasses import dataclass, field

from .schema import FMT_NB


# rates below are placeholders chosen to make the demo visible, not measured values.
# published repertoire studies put intra-sample chimeras near 20% of reads and
# inter-sample near 15%; Jessica to confirm what is realistic here.
@dataclass
class Sim:
    fmt: str = FMT_NB
    n_var: int = 5000
    n_reads: int = 200_000
    rnds: int = 3
    read: str = "long"
    read_len: int = 0
    err: float = 0.01
    chim: float = 0.05
    trunc: float = 0.02
    contam: float = 0.02
    fw_err: float = 0.01
    contam_fmt: str = "scfv"
    grow: float = 4.0
    grow_frac: float = 0.02
    bp_bias: tuple = (0.35, 0.65)
    seed: int = 0


@dataclass
class Qc:
    ident: float = 0.98
    min_ct: int = 2
    germ_ident: float = 0.70
    len_tol: float = 0.10
    abund_ratio: float = 8.0
    chim_min_bp: int = 20
    map_method: str = "umap"
    kmer: int = 4


@dataclass
class Cfg:
    sim: Sim = field(default_factory=Sim)
    qc: Qc = field(default_factory=Qc)
    raw: str = "data/raw"
    ref: str = "data/ref"
    out: str = "data/out"


CFG = Cfg()
