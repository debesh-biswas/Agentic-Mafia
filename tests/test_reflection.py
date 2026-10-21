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


# ---- Task 12: deep reflection, compression, fading, snapshots ----
from mafia.llm.client import LLMFormatError  # noqa: E402
from mafia.memory.maintenance import compress, fade  # noqa: E402
from mafia.memory.retrieval import retrieve  # noqa: E402
from mafia.reflection.deep import deep_reflect  # noqa: E402


def epi_store(tmp_path, games, imp=4):
    s = store(tmp_path)
    return s, [s.add("episodic", f"game {g} summary text", game=g, importance=imp + g % 3) for g in games]


def test_compress_merges_window_cites_max_importance_and_hides_originals(tmp_path):
    s, ids = epi_store(tmp_path, [1, 2, 3, 11, 25])  # at game 30: cutoff 20 -> windows 0 (1,2,3) and 1 (11 alone)
    call = caller([{"summary": "Merged.", "importance": 1}])
    out, merged = compress(s, MC, NAMES, call, "dev", 30)
    assert merged == 3 and len(call.users) == 1  # single-memory window and game 25 (too new) untouched
    new = [m for m in s.active(("episodic",)) if m.text == "Merged."]
    assert len(new) == 1 and new[0].cites == ids[:3] and new[0].game_created == 3 and new[0].game_updated == 30
    assert new[0].importance == max(s.get(i).importance for i in ids[:3])  # max of the group, not the model's 1
    assert all(s.get(i).status == "merged" for i in ids[:3]) and s.get(ids[3]).status == "active"
    assert out == [("memory_write", "private:dev", dict(memory_id=new[0].id, type="episodic", action="add", text="Merged."))]
    got = {m.id for m in retrieve(s, "game summary", 30, MC.retrieval, k=20)}
    assert not got & set(ids[:3]) and new[0].id in got


def test_summary_is_not_merged_again(tmp_path):
    s, ids = epi_store(tmp_path, [1, 2, 3])
    call = caller([{"summary": "Merged.", "importance": 1}])
    compress(s, MC, NAMES, call, "dev", 20)
    out, merged = compress(s, MC, NAMES, caller([]), "dev", 30)  # no call is made: the lone summary stays in window 0
    assert merged == 0 and out == [] and len(s.active(("episodic",))) == 1


def test_compress_skips_window_on_format_error(tmp_path):
    s, ids = epi_store(tmp_path, [1, 2, 11, 12])
    bad = {"nope": 1}
    call = caller([bad, bad, {"summary": "ok", "importance": 3}])  # first window exhausts its retry
    out, merged = compress(s, MC, NAMES, call, "dev", 40)
    assert merged == 2 and out[0][0] == "fallback"
    assert [s.get(i).status for i in ids] == ["active", "active", "merged", "merged"]


def test_fading_boundaries(tmp_path):
    s = store(tmp_path)
    f = lambda t, g, imp, last=None: (s.add(t, f"m{g}{imp}", game=g, importance=imp), last)
    cases = {"old_low": f("belief", 10, 3), "too_new": f("belief", 11, 3), "important": f("belief", 1, 4),
             "retrieved_recently": f("belief", 1, 2, 11), "retrieved_long_ago": f("belief", 1, 2, 10),
             "reinforced_recently": f("belief", 1, 2)}
    for i, last in cases.values():
        if last:
            s.touch([i], last)
    s.reinforce(cases["reinforced_recently"][0], 2, 25)  # game_updated counts as freshness
    assert fade(s, MC, 30) == 2  # created/retrieved <= 30-20=10 and importance <= 3
    st = {k: s.get(i).status for k, (i, _) in cases.items()}
    assert st == {"old_low": "faded", "too_new": "active", "important": "active",
                  "retrieved_recently": "active", "retrieved_long_ago": "faded", "reinforced_recently": "active"}


def test_deep_reflection_uses_shared_apply_and_shown_ids(tmp_path):
    s = store(tmp_path)
    e = [s.add("episodic", f"summary {g}", game=g, importance=5) for g in range(1, 8)]
    h = [s.add("reflection", f"hoh {g}", game=g, importance=6) for g in range(1, 8)]
    s.meta_set("hoh_ids", json.dumps(h))
    other = s.add("belief", "unseen belief", about=["sana"], game=1, importance=5)
    call = caller([{"insights": [ins("I trust too fast", kind="reflection", about=["self"], cites=[e[-1], other]),
                                 ins("lesson-ish", kind="lesson"),
                                 ins("rewrite", action="update", target_id=h[-1])]}])
    out = deep_reflect(s, NAMES, call, "dev", 5)
    u = call.users[0]
    assert all(f"[m{i}]" in u for i in e[-5:] + h[-5:]) and f"[m{e[0]}]" not in u and "[m%d]" % other not in u
    adds = [m for m in s.active(("reflection",)) if m.text == "I trust too fast"]
    assert adds[0].cites == [e[-1]] and adds[0].about == ["self"]  # unshown cite dropped
    assert not [m for m in s.active() if m.text == "lesson-ish"]  # lessons not allowed here
    assert any(t == "fallback" and "dropped 1 lesson" in p["reason"] for t, _, p in out)
    assert s.get(h[-1]).text == "rewrite" and s.get(other).text == "unseen belief"
    assert any(t == "memory_write" and p["action"] == "update" for t, _, p in out)


def test_deep_reflection_cap_is_three(tmp_path):
    s = store(tmp_path)
    s.add("episodic", "x", game=1, importance=5)
    call = caller([{"insights": [ins(f"i{k}", kind="reflection") for k in range(4)]}] * 2)
    assert deep_reflect(s, NAMES, call, "dev", 5)[0][0] == "fallback"  # schema rejects 4 -> format error, nothing written
    assert len(s.active(("reflection",))) == 0


def test_hoh_ids_recorded_by_reflect(tmp_path):
    s = store(tmp_path)
    call = caller([{"questions": ["q?"]}, insights(), ])
    out = reflect(s, MC, NAMES, call, outcome())
    ids = json.loads(s.meta_get("hoh_ids"))
    assert len(ids) == 1 and s.get(ids[0]).text == "It helped."


@pytest.mark.parametrize("g,deep,comp", [(1, 0, 0), (4, 0, 0), (5, 1, 0), (10, 1, 1), (15, 1, 0), (20, 1, 1)])
def test_cadence_deep_and_compression(tmp_path, g, deep, comp):
    llm = ScriptedLLM(5, NAMES)
    ag = mem_agents(tmp_path, llm)["dev"]
    for k in range(1, g + 1):  # build history so compression has something to merge
        ag.memory.add("episodic", f"old {k}", game=k, importance=4)
    specs = ag.post_game(outcome(game=g))
    cts = [p["call_type"] for t, v, p in specs if t == "llm_call"]
    assert cts.count("deep_reflect") == deep
    assert ("memory_maintenance" in [t for t, _, _ in specs]) == bool(comp)
    assert ("compress" in cts) == (comp and g >= 20)  # nothing older than 10 games before game 20


def test_snapshots_written_and_readable_by_inspect(tmp_path):
    cfg = run_cfg(tmp_path, games=4, rid="snap").model_copy(update={"snapshot_memory_every": 2})
    run_season(cfg, mock=True, runs_dir=tmp_path)
    snaps = sorted(p.name for p in (tmp_path / "snap" / "memory_snapshots").iterdir())
    assert snaps == ["g0002", "g0004"]
    assert len(list((tmp_path / "snap" / "memory_snapshots" / "g0002").glob("*.sqlite"))) == 7
    r = CliRunner().invoke(app, ["inspect", "--runs-dir", str(tmp_path), "--run", "snap", "--agent", "dev", "--at-game", "3"])
    assert r.exit_code == 0 and "[m" in r.output


def test_30_game_season_keeps_memory_bounded(tmp_path):
    cfg = run_cfg(tmp_path, games=30, rid="long").model_copy(update={"snapshot_memory_every": 10})
    run_season(cfg, mock=True, runs_dir=tmp_path)
    d = tmp_path / "long"
    assert sorted(p.name for p in (d / "memory_snapshots").iterdir()) == ["g0010", "g0020", "g0030"]
    maint = [e for g in range(30) for e in EventLog(d, g).read() if e.type == "memory_maintenance"]
    assert len(maint) == 3 * 7 and sum(e.payload["merged"] for e in maint) > 0
    # Derivation. Compression at games 10/20/30 merges episodic memories with game_created <= g-10, so after the
    # last one (g=30) the uncompressed ones are games 21..30 (10) plus one merged summary per 10-game window
    # touched. A summary keeps its window's latest game_created, so it is never re-merged: window 0 -> 1 summary,
    # window 1 -> 1 summary (window 2 = games 21..30 is still raw) -> 10 + 2 = 12.
    # Reflections (helped_or_hurt) are one per game and never faded (importance 6): 30. Insights: reflect adds <= 1
    # per game and deep reflection <= 1 per 5 games: <= 30 + 6.
    # Total active <= 12 + 30 + 36 = 78 against 30 games * (1 episodic + 1 reflection + 1 insight) = 90+ unbounded.
    for p in sorted((d / "memory").glob("*.sqlite")):
        st = MemoryStore(p, FakeEmbedder())
        act = st.active()
        assert len([m for m in act if m.type == "episodic"]) <= 12
        assert len(act) <= 78
        st.close()
