import json

import pandas as pd
import pytest

from truehit_atg import agent, data, ingest, schema as s


class Blk:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class Rsp:
    def __init__(self, text):
        self.content = [Blk(text)]


class Msgs:
    def __init__(self, outer):
        self.outer = outer

    def create(self, **kw):
        self.outer.calls.append(kw)
        out = self.outer.replies[min(len(self.outer.calls) - 1, len(self.outer.replies) - 1)]
        return Rsp(out if isinstance(out, str) else json.dumps(out))


# never touches the network
class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = Msgs(self)


GOOD = {
    "var": "clone", "seq": "protein", "drop_gaps": False, "note": "counts look like rounds",
    "rnds": [{"round": "MACS", "col": "input_ct"}, {"round": "FACS1", "col": "r1_ct"}],
}
BAD_COL = dict(GOOD, seq="nope")
ONE_ROUND = dict(GOOD, rnds=[{"round": "MACS", "col": "input_ct"}])


@pytest.fixture(scope="module")
def src(tmp_path_factory):
    df = data.load("AT110", pair=("FACS1", "MACS"))
    p = tmp_path_factory.mktemp("ag") / "unknown.csv"
    pd.DataFrame({
        "clone": df[s.VAR], "protein": df[s.AA_ALN],
        "input_ct": df[s.cnt("MACS")], "r1_ct": df[s.cnt("FACS1")],
    }).to_csv(p, index=False)
    return p


def test_to_spec_keeps_round_order():
    sp = agent.to_spec(GOOD)
    assert isinstance(sp, ingest.Spec)
    assert list(sp.rnds) == ["MACS", "FACS1"]
    assert sp.rnds["FACS1"] == "r1_ct"


def test_to_spec_rejects_one_round():
    with pytest.raises(ValueError, match="at least two rounds"):
        agent.to_spec(ONE_ROUND)


def test_ask_sends_schema_and_shape(src):
    cl = FakeClient(GOOD)
    agent.ask(ingest.probe(src), 126, client=cl)
    kw = cl.calls[0]
    assert kw["output_config"]["format"]["schema"] is agent.SPEC_SCHEMA
    assert kw["model"] == "claude-sonnet-5-5"
    assert "126 characters" in kw["messages"][0]["content"]
    assert "clone" in kw["messages"][0]["content"]
    assert "never invent" in kw["system"].lower()


def test_ask_raises_on_empty_text(src):
    with pytest.raises(ValueError, match="no text"):
        agent.ask(ingest.probe(src), 126, client=FakeClient(""))


def test_onboard_first_try(src):
    raw, spec, hist = agent.onboard(src, n_pos=126, client=FakeClient(GOOD))
    assert raw is not None and len(raw) == 5257
    assert spec.rnds == {"MACS": "input_ct", "FACS1": "r1_ct"}
    assert len(hist) == 1 and hist[0].ok and hist[0].n_rows == 5257


# the loop is the feature: it reads its own error and corrects
def test_onboard_recovers_from_a_bad_column(src):
    cl = FakeClient(BAD_COL, GOOD)
    raw, spec, hist = agent.onboard(src, n_pos=126, client=cl)
    assert raw is not None
    assert len(hist) == 2
    assert not hist[0].ok and "source columns not in" in hist[0].err
    assert hist[1].ok
    # the rejection was fed back
    second = cl.calls[1]["messages"]
    assert any("That was rejected" in m["content"] for m in second)
    assert any("nope" in m["content"] for m in second)


def test_onboard_exhausts_tries_without_crashing(src):
    raw, spec, hist = agent.onboard(src, n_pos=126, max_tries=3, client=FakeClient(BAD_COL))
    assert raw is None and spec is None
    assert len(hist) == 3
    assert all(not a.ok and a.err for a in hist)


def test_onboard_recovers_from_a_one_round_answer(src):
    raw, _, hist = agent.onboard(src, n_pos=126, client=FakeClient(ONE_ROUND, GOOD))
    assert raw is not None
    assert "at least two rounds" in hist[0].err


def test_onboard_rejects_a_dna_column_via_width(tmp_path):
    p = tmp_path / "dna.csv"
    pd.DataFrame({"clone": ["a", "b"], "dna": ["ACGT", "ACGA"],
                  "i": [5, 9], "r": [7, 2]}).to_csv(p, index=False)
    pick = {"var": "clone", "seq": "dna", "drop_gaps": False, "note": "x",
            "rnds": [{"round": "MACS", "col": "i"}, {"round": "FACS1", "col": "r"}]}
    raw, _, hist = agent.onboard(p, n_pos=126, max_tries=1, client=FakeClient(pick))
    assert raw is None
    assert "must be 126 long" in hist[0].err


def test_attempt_serializes_for_the_ui():
    a = agent.Attempt(spec=agent.to_spec(GOOD), ok=True, n_rows=10)
    d = a.as_dict()
    assert json.dumps(d)
    assert d["spec"]["rnds"] == {"MACS": "input_ct", "FACS1": "r1_ct"}
    assert agent.Attempt().as_dict()["spec"] is None


class Raising:
    def __init__(self, exc):
        self.messages = self
        self.exc = exc

    def create(self, **kw):
        raise self.exc


def test_unscoped_key_gets_a_sentence_not_a_traceback(src):
    err = Exception("400 ... must include the anthropic-workspace-id header ...")
    with pytest.raises(RuntimeError, match="not scoped to a workspace"):
        agent.ask(ingest.probe(src), 126, client=Raising(err))


def test_other_api_errors_are_not_swallowed(src):
    with pytest.raises(Exception, match="overloaded"):
        agent.ask(ingest.probe(src), 126, client=Raising(Exception("529 overloaded")))


def test_workspace_header_sent_when_configured(src, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_test")
    cl = FakeClient(GOOD)
    agent.ask(ingest.probe(src), 126, client=cl)
    assert cl.calls[0]["extra_headers"] == {"anthropic-workspace-id": "wrkspc_test"}


def test_no_workspace_header_when_unset(src, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    cl = FakeClient(GOOD)
    agent.ask(ingest.probe(src), 126, client=cl)
    assert "extra_headers" not in cl.calls[0]
