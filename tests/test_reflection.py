import csv
import json

import pytest
import yaml
from mafia.agents.llm_agent import LLMAgent
from mafia.agents.persona import NEUTRAL_SELF, starting_impressions
from mafia.cli import app
from mafia.config import ROOT, Memory, Personas, Rules, Scoring, load, load_run
from mafia.engine.game import Game
from mafia.llm.recorder import MockLLM, ScriptedLLM
from mafia.logs.events import Event, EventLog
from mafia.memory.embed import FakeEmbedder
from mafia.memory.store import MemoryStore
from mafia.reflection.reflect import GameOutcomeForAgent, build_outcome, reflect
from mafia.reflection.summaries import summarize
from mafia.runner.season import run_season
from typer.testing import CliRunner

R = load(Rules, ROOT / "config/rules.yaml")
S = load(Scoring, ROOT / "config/scoring_v0.yaml")
PC = load(Personas, ROOT / "config/personas.yaml")
P = sorted(PC.personas, key=lambda p: p.seat)
NAMES = {p.agent_id: p.name for p in P}
MC = load(Memory, ROOT / "config/memory.yaml")
PTS = dict(points_win=13, points_votes=1, points_saves=0, points_mislynch=0, points_total=14)


def outcome(me="dev", game=2):
    return GameOutcomeForAgent(me, game, "town", "mafia_voted_out", 3, {n: "villager" for n in NAMES.values()},
                               "villager", {1: "n1"}, PTS, "LEADERBOARD (after 2 games)\n1. Dev 5")


def store(tmp_path):
    return MemoryStore(tmp_path / "m.sqlite", FakeEmbedder())


def caller(replies):
    """call(type, schema, user) using real parsing over scripted raw JSON replies."""
    llm = MockLLM(list(map(json.dumps, replies)))
    users = []

    def call(ct, schema, user):
        users.append(user)
        return llm.call(ct, "sys", user, schema, "dev")[0]
    call.users = users
    return call


def insights(*ins, keep=()):
    return {"insights": list(ins), "keep_ids": list(keep), "helped_or_hurt": "It helped.", "game_summary": "We won.",
            "summary_importance": 5}


def ins(text, **kw):
    return {"text": text, "kind": "belief", "about": ["Sana"], "role_context": None, "confidence": "fairly sure",
            "importance": 7, "cites": [], "action": "add", "target_id": None} | kw


def test_actions_applied_and_invalid_dropped(tmp_path):
    st = store(tmp_path)
    a = st.add("belief", "Sana is quiet when mafia", about=["sana"], game=1, importance=5, confidence="hunch")
    b = st.add("belief", "Mira lies a lot", about=["mira"], game=1, importance=5)
    old = st.add("belief", "Jonah is hidden mafia", about=["jonah"], game=1, importance=5)
    st.set_status([old], "faded")
    call = caller([{"questions": ["Who is quiet?"]},
                   insights(ins("Sana is quiet when mafia, confirmed", action="update", target_id=a, cites=[a, 999]),
                            ins("Mira actually tells the truth", action="supersede", target_id=f"m{b}", cites=[b]),
                            ins("Jonah thing", action="supersede", target_id=old),  # not active -> add
                            ins("Kai thing", action="update", target_id=12345, cites=[12345]),  # never shown -> add
                            ins("Plain new", cites=[a]), keep=[a])])
    specs = reflect(st, MC, NAMES, call, outcome())
    assert st.versions(a) == [(2, "Sana is quiet when mafia", "hunch")]
    assert st.get(a).text.endswith("confirmed") and st.get(a).confidence == "fairly sure" and st.get(a).cites == []
    assert st.get(b).status == "superseded" and st.get(st.get(b).superseded_by).text == "Mira actually tells the truth"
    assert st.get(st.get(b).superseded_by).cites == [b] and st.get(st.get(b).superseded_by).about == ["mira"]
    texts = {m.text: m for m in st.active()}
    assert texts["Jonah thing"].cites == [] and texts["Plain new"].cites == [a] and texts["Plain new"].about == ["sana"]
    types = [(t, p.get("action")) for t, v, p in specs if t in ("memory_write", "memory_supersede")]
    assert ("memory_supersede", None) in types
    assert [p["action"] for t, v, p in specs if t == "memory_write"][:2] == ["update", "supersede"]
    assert sum(1 for t, v, p in specs if t == "fallback" and "treated as add" in p["reason"]) == 2
    assert all(v in ("private:dev", "debug") for t, v, p in specs)
    assert next(m for m in st.active() if m.text == "It helped.").type == "reflection"
    assert next(m for m in st.active() if m.text == "It helped.").importance == 6
    epi = next(m for m in st.active() if m.type == "episodic")
    assert (epi.text, epi.importance, epi.game_created) == ("We won.", 5, 2)
    assert any("[m%d]" % a in u for u in call.users[1:])


def test_format_error_skips_step_not_season(tmp_path):
    st = store(tmp_path)
    call = caller(["nonsense"] * 5)
    specs = reflect(st, MC, NAMES, call, outcome())
    assert [t for t, v, p in specs] == ["fallback"] and st.active() == []
    specs = summarize(st, NAMES, "dev", caller(["nope"] * 5), 2, "", "")
    assert specs[0][0] == "fallback" and st.get_summaries() == {}


def test_null_role_context_lesson_becomes_any_and_related_excludes_retrieved(tmp_path):
    st = store(tmp_path)
    a = st.add("belief", "Sana is quiet", about=["sana"], game=1, importance=5)
    call = caller([{"questions": ["Is Sana quiet?"]}, insights(ins("Stay calm", kind="lesson", about=[]), ins("Mira is calm"))])
    reflect(st, MC, NAMES, call, outcome())
    assert next(m for m in st.active() if m.text == "Mira is calm").role_context is None
    assert next(m for m in st.active() if m.text == "Stay calm").role_context == "any"
    held = call.users[1].split("Beliefs and lessons you already hold")[1]
    assert f"[m{a}]" in call.users[1].split("Beliefs and lessons you already hold")[0] and f"[m{a}]" not in held


def test_summaries_stored_by_id_unknown_dropped(tmp_path):
    st = store(tmp_path)
    call = caller([{"friends": {"Sana": "sharp", "Nobody": "x", "Dev": "me?"}, "self": "I'm loud"}])
    specs = summarize(st, NAMES, "dev", call, 3, "", "")
    assert st.get_summaries() == {"sana": "sharp", "self": "I'm loud"}
    assert specs == [("summary_update", "private:dev", {"friends": {"sana": "sharp"}, "self": "I'm loud"})]


def test_outcome_has_no_others_notes_or_night_targets():
    def e(type, actor=None, targets=(), vis="truth", **payload):
        return Event(run="t", game=0, seq=0, ts="2026-01-01T00:00:00", day=1, phase="x", type=type, actor=actor,
                     targets=list(targets), visibility=vis, payload=payload, scoring_version=0, prompt_version=1)
    ev = [e("night_notes", "dev", notes="my note"), e("night_notes", "sana", notes="SECRET sana note"),
          e("night_action", "sana", ["dev"], action="target"), e("speech", "sana", message="TRANSCRIPT"),
          e("game_end", winner="town", end_reason="mafia_voted_out", days=3),
          e("role_reveal", roles={a: "villager" for a in NAMES})]
    o = build_outcome(ev, "dev", {"dev": PTS, "sana": {"points_total": 99}}, NAMES, "LB", 4)
    blob = json.dumps(o.__dict__)
    assert o.notes_history == {1: "my note"} and "SECRET" not in blob and "TRANSCRIPT" not in blob
    assert set(o.__dict__) >= {"winner", "end_reason", "days", "roles", "my_role", "notes_history", "points", "leaderboard_text"}


def mem_agents(tmp_path, llm, cfg=MC):
    emb = FakeEmbedder()
    return {p.agent_id: LLMAgent(p, PC, llm, R, S, "memory", MemoryStore(tmp_path / f"{p.agent_id}.sqlite", emb), cfg)
            for p in P}


def test_plan_prompt_has_leaderboard_and_memories_from_game2(tmp_path):
    llm = ScriptedLLM(5, NAMES)
    agents = mem_agents(tmp_path, llm)
    dev = agents["dev"]
    dev.memory.add("lesson", "As mafia, stay quiet", role_context="any", game=1, importance=9)
    for g, lb in ((1, ""), (2, "LEADERBOARD (after 1 games)\n1. Dev 5")):
        llm.sent.clear()
        for a in agents.values():
            a.begin_game(g, lb)
        Game(R, P, agents, "t", g - 1, 5, "memory", S.version, 1).play()
        plans = [m[0]["content"] for m in llm.sent if m[-1]["content"].startswith("A new game is starting")]
        assert len(plans) == 7
        assert all(("LEADERBOARD" in p) == (g == 2) for p in plans)
        assert any("[m1] (lesson, as any) As mafia, stay quiet" in p for p in plans)
        others = [m[0]["content"] for m in llm.sent if not m[-1]["content"].startswith("A new game is starting")]
        assert not any("LEADERBOARD" in o for o in others)  # in-game calls omit it
    assert json.loads(dev.memory.meta_get("applied_lessons")) == [1]
    leo = next(p for p in P if p.agent_id == "leo")
    assert agents["leo"].memory.get_summaries() == {**starting_impressions(leo, PC), "self": NEUTRAL_SELF}


def run_cfg(tmp_path, group="memory", games=3, rid="mem"):
    d = yaml.safe_load((ROOT / "config/run_memory.yaml").read_text())
    d |= {"run_id": rid, "group": group, "games": games, "scoring_config": "config/scoring_v0.yaml"}
    f = tmp_path / f"{rid}.yaml"
    f.write_text(yaml.safe_dump(d))
    return load_run(f)


def types(path):
    return [e.type for e in EventLog(path.parent.parent, int(path.stem) - 1).read()]


def test_memory_season_end_to_end_and_resume(tmp_path):
    run_season(run_cfg(tmp_path), mock=True, runs_dir=tmp_path)
    d = tmp_path / "mem"
    evs = EventLog(d, 2).read()
    ts = {e.type for e in evs}
    assert {"reflection", "memory_write", "summary_update", "leaderboard", "score"} <= ts
    lb = [e for e in evs if e.type == "leaderboard"]
    assert [e.visibility for e in lb] == [f"private:{p.agent_id}" for p in P]
    assert all("after 3 games" in e.payload["text"] for e in lb)
    first_post = next(i for i, e in enumerate(evs) if e.type == "leaderboard")
    assert evs[first_post - 1].type == "score" and not any(e.type == "score" for e in evs[first_post:])
    assert any(e.type == "llm_call" and e.payload["call_type"] == "reflect" for e in evs)
    rows = list(csv.DictReader(open(d / "agents.csv")))
    assert int(rows[-1]["memories_active"]) > 0 and sum(int(r["beliefs_active"]) + int(r["lessons_active"]) for r in rows) > 0
    assert len(list((d / "memory").glob("*.sqlite"))) == 7
    # resume: more games, leaderboard rebuilt from logs
    run_season(run_cfg(tmp_path, games=4), mock=True, resume=True, runs_dir=tmp_path)
    lb4 = [e for e in EventLog(d, 3).read() if e.type == "leaderboard"]
    assert "after 4 games" in lb4[0].payload["text"]
    tot = {a: 0 for a in NAMES}
    for g in range(4):
        pts = next(e.payload["points"] for e in EventLog(d, g).read() if e.type == "score")
        for a in tot:
            tot[a] += pts[a]["points_total"]
    assert lb4[0].payload["standings"] == tot


def test_control_group_creates_no_memory_and_never_reflects(tmp_path):
    llm = ScriptedLLM(5, NAMES)
    run_season(run_cfg(tmp_path, "control", 1, "ctl"), mock=True, runs_dir=tmp_path)
    d = tmp_path / "ctl"
    assert not (d / "memory").exists()
    evs = EventLog(d, 0).read()
    assert not {e.type for e in evs} & {"reflection", "memory_write", "summary_update", "leaderboard"}
    assert not {e.payload["call_type"] for e in evs if e.type == "llm_call"} & {"reflect", "summaries"}
    ctl = LLMAgent(P[0], PC, llm, R, S, "control")
    assert ctl.post_game(outcome()) == [] and llm.sent == []


def test_inspect_cli(tmp_path):
    run_season(run_cfg(tmp_path, rid="pytest-inspect-tmp"), mock=True, runs_dir=tmp_path)
    inv = lambda *a: CliRunner().invoke(app, ["inspect", "--runs-dir", str(tmp_path), *a])
    r = inv("--run", "pytest-inspect-tmp", "--agent", "dev")
    assert r.exit_code == 0 and "== " in r.output and "[m" in r.output
    r = inv("--run", "pytest-inspect-tmp", "--agent", "dev", "--summaries")
    assert r.exit_code == 0 and "Sana:" in r.output
    r = inv("--run", "pytest-inspect-tmp", "--agent", "dev", "--at-game", "2")
    assert r.exit_code == 1 and "snapshot" in r.output


def dump(d):
    out = {}
    for p in sorted((d / "memory").glob("*.sqlite")):
        st = MemoryStore(p, FakeEmbedder())
        out[p.stem] = (sorted((m.type, m.text) for m in st.active()), st.get_summaries())
        st.close()
    return out


def test_crash_in_post_game_resumes_to_same_memory(tmp_path, monkeypatch):
    run_season(run_cfg(tmp_path, rid="full"), mock=True, runs_dir=tmp_path)
    orig, n, lock = LLMAgent.post_game, [0], __import__("threading").Lock()

    def boom(self, outcome):
        r = orig(self, outcome)
        with lock:
            n[0] += 1
            if n[0] == 17:  # game 3, after several agents have written memories
                raise RuntimeError("boom")
        return r
    monkeypatch.setattr(LLMAgent, "post_game", boom)
    with pytest.raises(RuntimeError):
        run_season(run_cfg(tmp_path, rid="crash"), mock=True, runs_dir=tmp_path)
    monkeypatch.setattr(LLMAgent, "post_game", orig)
    run_season(run_cfg(tmp_path, rid="crash"), mock=True, resume=True, runs_dir=tmp_path)
    assert dump(tmp_path / "crash") == dump(tmp_path / "full")
    assert [r["game"] for r in csv.DictReader(open(tmp_path / "crash/games.csv"))] == ["1", "2", "3"]


def test_crash_after_checkpoint_before_state_restores_previous_generation(tmp_path, monkeypatch):
    import mafia.runner.season as SE
    run_season(run_cfg(tmp_path, rid="full"), mock=True, runs_dir=tmp_path)
    monkeypatch.setattr(SE, "_prune_ckpt", lambda *a: None)  # keep g0002 and g0003 around
    run_season(run_cfg(tmp_path, rid="crash"), mock=True, runs_dir=tmp_path)
    monkeypatch.undo()
    d = tmp_path / "crash"
    assert (d / "memory/ckpt/g0002").exists() and (d / "memory/ckpt/g0003").exists()
    (d / "state.json").write_text(json.dumps({"last_completed": 2}))  # crash between rename and state write
    (d / "memory/ckpt/g0004.tmp").mkdir()
    (d / "memory/dev.sqlite-journal").write_text("junk")
    run_season(run_cfg(tmp_path, rid="crash"), mock=True, resume=True, runs_dir=tmp_path)
    assert dump(d) == dump(tmp_path / "full")
    assert [p.name for p in (d / "memory/ckpt").iterdir()] == ["g0003"] and not (d / "memory/dev.sqlite-journal").exists()


def test_missing_checkpoint_generation_is_a_clear_error(tmp_path):
    import shutil

    from mafia.runner.season import CheckpointMissing
    run_season(run_cfg(tmp_path, games=2), mock=True, runs_dir=tmp_path)
    d = tmp_path / "mem"
    (d / "memory/ckpt/g0002/leo.sqlite").unlink()
    with pytest.raises(CheckpointMissing, match="leo"):
        run_season(run_cfg(tmp_path, games=3), mock=True, resume=True, runs_dir=tmp_path)
    shutil.rmtree(d / "memory/ckpt")
    with pytest.raises(CheckpointMissing):
        run_season(run_cfg(tmp_path, games=3), mock=True, resume=True, runs_dir=tmp_path)
