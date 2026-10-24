import json
from dataclasses import asdict, dataclass, field

from . import encode, ingest, schema as s
from .config import CFG

# the shape the model must return. mirrors ingest.Spec, and the API enforces it through
# output_config rather than us hoping for clean JSON in prose.
SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "var": {"type": "string", "description": "source column holding the clone id"},
        "seq": {"type": "string", "description": "source column holding the aligned protein sequence"},
        "rnds": {
            "type": "array",
            "description": "selection rounds, earliest first, each mapping a round name to its count column",
            "items": {
                "type": "object",
                "properties": {
                    "round": {"type": "string"},
                    "col": {"type": "string"},
                },
                "required": ["round", "col"],
                "additionalProperties": False,
            },
        },
        "drop_gaps": {"type": "boolean", "description": "true only if the sequences carry gaps that should be removed"},
        "note": {"type": "string", "description": "one short sentence on why these columns"},
    },
    "required": ["var", "seq", "rnds", "drop_gaps", "note"],
    "additionalProperties": False,
}

SYS = """You map columns in an unfamiliar CSV onto a fixed contract for antibody selection data.

The contract, one row per clone:
  var_id   the clone identifier, unique
  aa_aln   the aligned protein sequence, one letter per residue, same length on every row
  a count column per selection round, ordered earliest round first

Rules:
- Choose only from the column names you are given. Never invent one.
- The sequence column holds protein, not DNA. A column of only A, C, G and T is DNA; do not pick it.
- Round order matters: the earliest round (input, naive, pre-selection, R0, MACS) comes first,
  then later rounds in order. Enrichment is computed late against early, so the order is the
  whole meaning.
- Pick at least two round columns. Counts are whole numbers, one per round.
- Set drop_gaps true only when the sequences contain gap characters that should be stripped.
- If an earlier attempt failed, read the error and change what it names. Do not repeat a
  rejected answer."""


@dataclass
class Attempt:
    spec: object = None
    ok: bool = False
    err: str = ""
    n_rows: int = 0

    def as_dict(self):
        return {
            "spec": asdict(self.spec) if self.spec else None,
            "ok": self.ok,
            "err": self.err,
            "n_rows": self.n_rows,
        }


def build_client():
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "the anthropic package is not installed; run "
            "`pip install -r requirements.txt` in true-hit-atg/"
        ) from e
    try:
        # zero-arg resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login`
        # profile, so an unset key does not necessarily mean no credentials
        return anthropic.Anthropic()
    except Exception as e:
        raise RuntimeError(
            f"could not build an Anthropic client: {e}. Set {CFG.ai_key_env} or run `ant auth login`."
        ) from e


def ask(desc, n_pos, hist=(), client=None, model=None, effort=None):
    cl = client or build_client()
    lines = [
        f"Columns: {desc['cols']}",
        f"Dtypes: {desc['dtypes']}",
        f"Rows: {desc['n_rows']}",
        f"Every aligned sequence must be exactly {n_pos} characters." if n_pos else "",
        f"First rows: {json.dumps(desc['head'], default=str)[:4000]}",
    ]
    msgs = [{"role": "user", "content": "\n".join(x for x in lines if x)}]
    for att in hist:
        msgs.append({"role": "assistant", "content": json.dumps(asdict(att.spec)) if att.spec else "{}"})
        msgs.append({"role": "user", "content": f"That was rejected: {att.err}\nTry again."})

    rsp = cl.messages.create(
        model=model or CFG.ai_model,
        max_tokens=CFG.ai_max_tokens,
        system=SYS,
        messages=msgs,
        output_config={
            "format": {"type": "json_schema", "schema": SPEC_SCHEMA},
            "effort": effort or CFG.ai_effort,
        },
    )
    txt = next((b.text for b in rsp.content if b.type == "text"), "")
    if not txt.strip():
        raise ValueError("the model returned no text")
    return json.loads(txt)


def to_spec(d):
    rnds = {}
    for e in d.get("rnds", []):
        rnds[e["round"]] = e["col"]
    if len(rnds) < 2:
        raise ValueError(f"need at least two rounds, the model gave {list(rnds)}")
    return ingest.Spec(
        var=d["var"], seq=d["seq"], rnds=rnds,
        drop_gaps=bool(d.get("drop_gaps", False)), note=d.get("note", ""),
    )


def onboard(path, n_pos=None, max_tries=None, client=None, model=None):
    tries = max_tries or CFG.ai_max_tries
    desc = ingest.probe(path)
    hist = []
    for _ in range(tries):
        att = Attempt()
        try:
            att.spec = to_spec(ask(desc, n_pos, hist, client=client, model=model))
            raw = ingest.apply_spec(path, att.spec, n_pos=n_pos)
        except (ValueError, KeyError) as e:
            att.err = str(e)
            hist.append(att)
            continue
        att.ok, att.n_rows = True, len(raw)
        hist.append(att)
        return raw, att.spec, hist
    return None, None, hist
