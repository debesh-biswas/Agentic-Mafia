from types import SimpleNamespace as NS

import json

import pytest
from mafia.cli import app
from mafia.config import ROOT, ModelCfg, load
from mafia.llm.client import ATTEMPTS, LLMClient, LLMError, LLMFormatError, ProviderMismatch, strip_reasoning
from mafia.llm.recorder import MockLLM, RecordingLLM, ReplayLLM
from mafia.llm.schemas import NightDecision, VoteDecision
from typer.testing import CliRunner

V = '{"vote": "Bo"}'


def test_strip():
    assert strip_reasoning(f"<think>hmm {{x}}</think>\n{V}") == V
    assert strip_reasoning(f"hidden</think>{V}") == V
    assert strip_reasoning(f"```json\n{V}\n```") == V
    h = f"<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>{V}<|return|>"
    assert strip_reasoning(h) == V
    assert strip_reasoning(f"<|channel|>analysis<|message|>think<|end|>{V}") == V


def test_one_retry_then_ok_and_message_text():
    m = MockLLM(["not json", V])
    out, rec = m.call("vote", "sys", "usr", VoteDecision, "a1")
    assert out.vote == "Bo" and rec["format_errors"] == 1
    retry = m.sent[1]
    assert retry[2] == {"role": "assistant", "content": "not json"}
    assert retry[3]["content"].startswith("Your last answer was invalid: ") and retry[3]["content"].endswith(
        ". Answer again with valid JSON.")


def test_two_failures_raise_after_exactly_one_retry():
    m = MockLLM(["nope", '{"vote": 3}', V])
    with pytest.raises(LLMFormatError) as e:
        m.call("vote", "s", "u", VoteDecision, "a1")
    assert len(m.sent) == 2 and m.script == [V] and e.value.record["format_errors"] == 2


def test_think_tags_stripped_in_mock():
    out, rec = MockLLM([f"<think>x</think>{V}"]).call("vote", "s", "u", VoteDecision, "a")
    assert out.vote == "Bo" and rec["format_errors"] == 0


def test_record_replay(tmp_path):
    p = tmp_path / "llm_cache" / "0.jsonl"
    nd = '{"notes": "n", "urgency": 2, "target": "Bo"}'
    rec = RecordingLLM(MockLLM([V, V, nd]), p)
    got = [rec.call("vote", "s", "u", VoteDecision, "a1"), rec.call("vote", "s", "u", VoteDecision, "a1"),
           rec.call("night", "s", "u", NightDecision, "a1")]
    rp = ReplayLLM(p)
    for (m, _), (c, s) in zip(got, [("vote", VoteDecision), ("vote", VoteDecision), ("night", NightDecision)]):
        assert rp.call(c, "", "", s, "a1")[0] == m
    with pytest.raises(KeyError, match="call_index=2"):
        rp.call("vote", "", "", VoteDecision, "a1")


class FakeSDK:
    """Shape of AsyncOpenAI: .chat.completions.create(**kw) -> response."""

    def __init__(self, *items):
        self.items, self.calls = list(items), []
        self.chat = NS(completions=NS(create=self.create))
        self.closed = 0

    async def close(self):
        self.closed += 1

    async def create(self, **kw):
        self.calls.append(kw)
        it = self.items.pop(0)
        if isinstance(it, Exception):
            raise it
        return it


class HTTPErr(Exception):
    def __init__(self, code):
        self.status_code = code


def resp(content=V, provider=None, finish="stop"):
    return NS(choices=[NS(message=NS(content=content, reasoning="because"), finish_reason=finish)], provider=provider,
              usage=NS(prompt_tokens=10, completion_tokens=5, cost=0.001,
                       completion_tokens_details=NS(reasoning_tokens=3)))


def cfg(name="model_openrouter", **kw):
    c = load(ModelCfg, ROOT / f"config/{name}.yaml")
    c.provider_pin = kw.pop("pin", None)
    return c


def client(c, *items):
    sdk = FakeSDK(*items)
    cl = LLMClient(c, sdk)
    cl.backoff_s = 0
    return cl, sdk


def test_request_shape_and_record():
    cl, sdk = client(cfg(pin={"order": ["P"]}), resp(provider="p"))
    out, rec = cl.call("vote", "sys", "usr", VoteDecision, "a1")
    kw = sdk.calls[0]
    assert out.vote == "Bo" and kw["extra_body"] == {"reasoning": {"effort": "medium"}, "provider": {"order": ["P"]}}
    assert kw["response_format"]["json_schema"]["name"] == "VoteDecision" and kw["max_tokens"] == 600
    assert rec | {"latency_s": 0, "prompt_hash": 0} == {
        "call_type": "vote", "agent": "a1", "tokens_in": 10, "tokens_out": 5, "tokens_reasoning": 3, "retries": 0,
        "format_errors": 0, "error": None, "provider": "p", "finish_reason": "stop", "cost": 0.001, "reasoning": "because",
        "latency_s": 0, "prompt_hash": 0}


def test_local_reasoning_param_and_token_budgets():
    cl, sdk = client(cfg("model_local"), resp(), resp(), resp())
    cl.call("speak", "s", "u", VoteDecision, "a")
    cl.call("deep_reflect", "s", "u", VoteDecision, "a")
    cl.call("summaries", "s", "u", VoteDecision, "a")
    assert [c["reasoning_effort"] for c in sdk.calls] == ["low", "medium", "low"]
    assert [c["max_tokens"] for c in sdk.calls] == [600, 1500, 1200] and "extra_body" not in sdk.calls[0]


def test_network_backoff_then_success_and_exhaustion():
    cl, _ = client(cfg(), HTTPErr(500), HTTPErr(429), resp())
    assert cl.call("vote", "s", "u", VoteDecision, "a")[1]["retries"] == 2
    cl, sdk = client(cfg(), *[HTTPErr(503)] * ATTEMPTS)
    with pytest.raises(LLMError) as e:
        cl.call("vote", "s", "u", VoteDecision, "a")
    assert len(sdk.calls) == ATTEMPTS and e.value.record["retries"] == 0


def test_4xx_not_retried():
    cl, sdk = client(cfg(), HTTPErr(400), resp())
    with pytest.raises(LLMError):
        cl.call("vote", "s", "u", VoteDecision, "a")
    assert len(sdk.calls) == 1


def test_provider_mismatch_aborts():
    cl, _ = client(cfg(pin={"order": ["DeepInfra"]}), resp(provider="Other"))
    with pytest.raises(ProviderMismatch):
        cl.call("vote", "s", "u", VoteDecision, "a")


def test_format_retry_through_sdk():
    cl, sdk = client(cfg("model_local"), resp("garbage"), resp(f"<think>x</think>{V}"))
    out, rec = cl.call("vote", "s", "u", VoteDecision, "a")
    assert out.vote == "Bo" and rec["format_errors"] == 1 and rec["tokens_in"] == 20
    assert len(sdk.calls[1]["messages"]) == 4


def test_llm_check_cli(monkeypatch):
    from mafia.llm import client as c
    async def fake(self, call_type, system, user, schema, agent_id):
        if schema.__name__ == "Compressed":
            raise LLMFormatError("bad", {"format_errors": 2})
        return schema.model_construct(), {"format_errors": 0}

    monkeypatch.setattr(c.LLMClient, "acall", fake)
    r = CliRunner().invoke(app, ["llm-check", "--model-config", str(ROOT / "config/model_local.yaml"), "--model", "x"])
    assert r.exit_code == 1 and r.output.count("PASS") == 9 and "Compressed" in r.output and "FAIL" in r.output


def test_unclosed_think_stripped():
    assert strip_reasoning(f"{V}<think>cut off mid") == V


def test_truncated_and_empty_fail_without_retry():
    for content, finish, msg in [(V, "length", "truncated at max_tokens"), ("<think>x</think>", "stop", "empty content"),
                                 (None, "stop", "empty content")]:
        cl, sdk = client(cfg(), resp(content, finish=finish), resp())
        with pytest.raises(LLMFormatError, match=msg) as e:
            cl.call("vote", "s", "u", VoteDecision, "a")
        assert len(sdk.calls) == 1 and e.value.record["finish_reason"] == finish


def test_record_replay_with_failure_in_middle(tmp_path):
    p = tmp_path / "g.jsonl"
    rec = RecordingLLM(MockLLM([V, "bad", "bad", V]), p)
    out = []
    for _ in range(3):
        try:
            out.append(rec.call("vote", "s", "u", VoteDecision, "a1")[0])
        except LLMFormatError:
            out.append(None)
    assert [o is None for o in out] == [False, True, False]
    rp = ReplayLLM(p)
    got = []
    for _ in range(3):
        try:
            got.append(rp.call("vote", "", "", VoteDecision, "a1")[0])
        except LLMFormatError as e:
            assert e.record["format_errors"] == 2
            got.append(None)
    assert got == out
    RecordingLLM(MockLLM([]), p)  # truncates
    assert p.read_text() == ""


def test_replay_duplicate_key(tmp_path):
    p = tmp_path / "d.jsonl"
    row = json.dumps({"key": ["a", "vote", 0], "response": {"vote": "Bo"}, "record": {}})
    p.write_text(row + "\n" + row + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        ReplayLLM(p)


def test_missing_api_key_and_pin_without_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        LLMClient(cfg())
    cl, _ = client(cfg(pin={"order": ["P"]}), resp(provider=None))
    with pytest.raises(ProviderMismatch):
        cl.call("vote", "s", "u", VoteDecision, "a")


def test_sync_call_closes_sdk():
    cl, sdk = client(cfg(), resp())
    cl.call("vote", "s", "u", VoteDecision, "a")
    assert sdk.closed == 1
