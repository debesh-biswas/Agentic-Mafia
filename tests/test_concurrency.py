import asyncio
import pytest
import random
import threading
import time
from types import SimpleNamespace as NS

from mafia.agents.llm_agent import LLMAgent
from mafia.config import ROOT, ModelCfg, Personas, Rules, Scoring, load
from mafia.engine.game import Game
from mafia.llm.client import LLMClient, LLMError, LLMFormatError
from mafia.llm.recorder import RecordingLLM, ScriptedLLM
from mafia.llm.schemas import PlanDecision, VoteDecision

R = load(Rules, ROOT / "config/rules.yaml")
S = load(Scoring, ROOT / "config/scoring_v0.yaml")
PC = load(Personas, ROOT / "config/personas.yaml")
P = sorted(PC.personas, key=lambda p: p.seat)
NAMES = {p.agent_id: p.name for p in P}
SYS = random.SystemRandom()  # unseeded: varies completion order between runs


class Sleepy:
    """Sleeps a random bit per call; optionally answers some plans with an over-long text."""

    def __init__(self, inner, once=(), always=()):
        self.inner, self.once, self.always, self.plans = inner, set(once), set(always), {}
        self._lock = threading.Lock()

    def call(self, call_type, system, user, schema, agent_id):
        time.sleep(SYS.random() * 0.01)
        if call_type == "plan":
            with self._lock:
                n = self.plans[agent_id] = self.plans.get(agent_id, 0) + 1
            if agent_id in self.always or agent_id in self.once and n == 1:
                return PlanDecision(plan="word " * 150), {"call_type": "plan", "latency_s": 0}
        return self.inner.call(call_type, system, user, schema, agent_id)


def _play(llm, concurrent=True):
    agents = {p.agent_id: LLMAgent(p, PC, llm, R, S, "control") for p in P}
    g = Game(R, P, agents, "t", 0, 5, "control", S.version, 1, concurrent=concurrent)
    g.play()
    return [(e.model_dump(mode="json", exclude={"ts"}) | {"payload": {k: v for k, v in e.payload.items() if k != "latency_s"}})
            for e in g.events]


def test_event_log_independent_of_completion_order():
    runs = [_play(Sleepy(ScriptedLLM(5, NAMES))) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    assert runs[0] == _play(ScriptedLLM(5, NAMES), concurrent=False)


def test_retry_and_fallback_are_deterministic():
    a1, a2 = P[0].agent_id, P[1].agent_id
    logs, llms = [], []
    for _ in range(3):
        llm = Sleepy(ScriptedLLM(5, NAMES), once=[a1], always=[a2])
        logs.append(_play(llm))
        llms.append(llm)
    assert logs[0] == logs[1] == logs[2]
    assert llms[0].plans[a1] == 2 and llms[0].plans[a2] == 2  # exactly one retry each
    assert [e for e in logs[0] if e["type"] == "fallback" and e["actor"] == a2 and e["payload"]["reason"] == "truncated"]
    assert not [e for e in logs[0] if e["type"] == "fallback" and e["actor"] == a1]


def test_recording_under_concurrency_matches_sequential(tmp_path):
    keys = []
    for conc in (True, False):
        path = tmp_path / f"{conc}.jsonl"
        _play(RecordingLLM(Sleepy(ScriptedLLM(5, NAMES)), path), concurrent=conc)
        keys.append(sorted(l.split('"record"')[0] for l in path.read_text().splitlines()))
    assert keys[0] == keys[1] and len(keys[0]) > 50


def test_client_concurrent_calls_respect_semaphore():
    state = NS(now=0, peak=0)

    async def create(**kw):
        state.now += 1
        state.peak = max(state.peak, state.now)
        await asyncio.sleep(0.02)
        state.now -= 1
        return NS(choices=[NS(message=NS(content='{"vote": "Bo"}'), finish_reason="stop")], provider=None, usage=None)

    async def close():
        pass

    sdk = NS(chat=NS(completions=NS(create=create)), close=close)
    cfg = load(ModelCfg, ROOT / "config/model_openrouter.yaml")
    cfg.provider_pin, cfg.max_concurrency = None, 3
    cl = LLMClient(cfg, sdk)
    outs = []
    ts = [threading.Thread(target=lambda i=i: outs.append(cl.call("vote", "s", "u", VoteDecision, f"a{i}"))) for i in range(10)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    cl.close()
    assert len(outs) == 10 and all(o[0].vote == "Bo" for o in outs)
    assert state.peak == 3


class Failing(Sleepy):
    """Agent `bad` raises `exc` on its night calls."""

    def __init__(self, inner, bad, exc):
        super().__init__(inner)
        self.bad, self.exc = bad, exc

    def call(self, call_type, system, user, schema, agent_id):
        if call_type == "night" and agent_id == self.bad:
            raise self.exc
        return super().call(call_type, system, user, schema, agent_id)


def test_format_error_in_concurrent_night_falls_back_identically():
    bad = P[2].agent_id
    mk = lambda: Failing(ScriptedLLM(5, NAMES), bad, LLMFormatError("garbage", {"call_type": "night", "latency_s": 0}))
    seq, conc = _play(mk(), concurrent=False), _play(mk(), concurrent=True)
    assert seq == conc
    fb = [e for e in conc if e["type"] == "fallback" and e["actor"] == bad and e["payload"]["call_type"] == "night"]
    assert fb and conc[-1]["type"] == "role_reveal"


def test_plain_llm_error_propagates_in_both_modes():
    for conc in (False, True):
        with pytest.raises(LLMError, match="net down"):
            _play(Failing(ScriptedLLM(5, NAMES), P[2].agent_id, LLMError("net down", {})), concurrent=conc)


def test_call_from_loop_thread_raises():
    cl = LLMClient(load(ModelCfg, ROOT / "config/model_local.yaml"))

    async def inside():
        cl.call("vote", "s", "u", VoteDecision, "a")

    with pytest.raises(RuntimeError, match="deadlock"):
        asyncio.run_coroutine_threadsafe(inside(), cl._loop_running()).result()
    cl.close()
