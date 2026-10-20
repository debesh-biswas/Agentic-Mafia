import pytest
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


def test_agent_reuse_resets_per_game_and_control_has_no_leaderboard():
    llm = ScriptedLLM(5, NAMES)
    agents = {p.agent_id: LLMAgent(p, PC, llm, R, S, "control") for p in P}
    runs = []
    for i in (0, 1):
        llm.sent.clear()
        g = Game(R, P, agents, "t", i, 5, "control", S.version, 1)
        g.play()
        runs.append((g, [m[0]["content"] for m in llm.sent], {a: x.plan_text for a, x in agents.items()}))
    (g1, sys1, plans1), (g2, sys2, plans2) = runs
    roles2 = {e.actor: e.payload["role"] for e in g2.events if e.type == "role_assigned"}
    for a, x in agents.items():
        assert x.role_line.split(":")[1].strip().lower().startswith(roles2[a])
        assert set(x.notes_history) <= {1, 2, 3, 4, 5, 6, 7, 8}
    assert not any("LEADERBOARD" in s.upper() for s in sys1 + sys2)
    # a call-1 style check: game 2's plan-time system message carries no plan text yet
    plan_prompts = [m for m in sys2 if "YOUR PLAN FOR THIS GAME\n(none)" in m]
    assert len(plan_prompts) >= 7
    assert not any(p in s for s in plan_prompts for p in plans1.values() if p != "Stay observant and vote on evidence." and p)


def test_llm_call_events_never_reach_prompts():
    llm = ScriptedLLM(5, NAMES)
    g, _ = _play(llm)
    assert any(e.type == "llm_call" for e in g.events)
    assert not any("llm_call" in m["content"] or "prompt_hash" in m["content"] for ms in llm.sent for m in ms)


def test_memory_group_refused():
    with pytest.raises(NotImplementedError):
        LLMAgent(P[0], PC, None, R, S, "memory")


def test_cli_play_mock(tmp_path):
    import shutil

    import yaml
    from mafia.cli import app
    from typer.testing import CliRunner
    cfg = yaml.safe_load((ROOT / "config/run_local.yaml").read_text()) | {"run_id": "pytest-cli-tmp"}
    f = tmp_path / "run.yaml"
    f.write_text(yaml.safe_dump(cfg))
    try:
        res = CliRunner().invoke(app, ["play", "--run-config", str(f), "--games", "1", "--mock"])
        assert res.exit_code == 0 and "NIGHT 1" in res.output
        assert (ROOT / "runs/pytest-cli-tmp/games/0001.jsonl").exists()
        cfg["group"] = "memory"
        f.write_text(yaml.safe_dump(cfg))
        res = CliRunner().invoke(app, ["play", "--run-config", str(f), "--mock"])
        assert res.exit_code == 1 and "memory group not implemented" in res.output
    finally:
        shutil.rmtree(ROOT / "runs/pytest-cli-tmp", ignore_errors=True)
