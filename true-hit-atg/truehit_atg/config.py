from dataclasses import dataclass


@dataclass
class Cfg:
    camp: str = "AT110"
    train: tuple = ("FACS1", "MACS")
    test: tuple = ("FACS2", "MACS")
    min_ct: int = 0
    min_ct_rnd: str = ""
    norm: str = "cpm"
    seq_col: str = "aa_aln"
    seed: int = 0
    out: str = "data/out"

    fil_cnt: int = 5
    cpm_floor: float = 1.0
    gmm_tr: float = 0.8
    gmm_seed: int = 274
    test_size: float = 0.2
    split_seed: int = 42

    lam: float = 2.0
    max_iter: int = 2000
    top_n: int = 50


CFG = Cfg()
