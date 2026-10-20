from mafia.agents.llm_agent import LLMAgent
from mafia.config import ROOT, ModelCfg, Personas, Rules, Scoring, load
from mafia.engine.game import Game
from mafia.llm.client import LLMClient
from mafia.llm.recorder import RecordingLLM, ReplayLLM, ScriptedLLM

R = load(Rules, ROOT / "config/rules.yaml")
S = load(Scoring, ROOT / "config/scoring_v0.yaml")
PC = load(Personas, ROOT / "config/personas.yaml")
P = sorted(PC.personas, key=lambda p: p.seat)
NAMES = {p.agent_id: p.name for p in P}


def _play(llm, seed=5):
    agents = {p.agent_id: LLMAgent(p, PC, llm, R, S, "control") for p in P}
    g = Game(R, P, agents, "t", 0, seed, "control", S.version, 1)
    g.play()
    return g, agents


def _norm(events):
    out = []
    for e in events:
        d = e.model_dump(mode="json", exclude={"ts"})
        d["payload"] = {k: v for k, v in d["payload"].items() if not (e.type == "llm_call" and k == "latency_s")}
        out.append(d)
    return out


def test_full_game_exercises_all_calls_and_prompts():
    llm = ScriptedLLM(5, NAMES)
    g, agents = _play(llm)
    assert {e.payload["call_type"] for e in g.events if e.type == "llm_call"} == {"plan", "night", "speak", "vote", "last_words"}
    assert not [e for e in g.events if e.type == "fallback"]
    systems = {m[0]["content"] for m in llm.sent}
    assert any("YOUR ROLE THIS GAME: Mafia" in s for s in systems)
    assert all(any(p.name in s for s in systems) for p in P)
    assert all(a.pending_records == [] and a.plan_text for a in agents.values())
    assert "(none)" in next(iter(systems))  # no memories for control


def test_replay_gives_identical_event_log(tmp_path):
    g1, _ = _play(RecordingLLM(ScriptedLLM(5, NAMES), tmp_path / "c.jsonl"))
    g2, _ = _play(ReplayLLM(tmp_path / "c.jsonl"))
    assert _norm(g1.events) == _norm(g2.events)


def test_play_real_client_builds_without_network():
    LLMClient(load(ModelCfg, ROOT / "config/model_local.yaml"))  # construction only
