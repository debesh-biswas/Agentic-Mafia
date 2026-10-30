import csv
import json

import pytest
from mafia.agents.llm_agent import LLMAgent
from mafia.cli import app
from mafia.config import ROOT, Scoring, load, load_run
from mafia.llm.recorder import ScriptedLLM
from mafia.logs.events import Event
from mafia.metrics import compute
from mafia.runner.season import RunExists, run_season
from mafia.scoring import leaderboard
from mafia.scoring.score import score
from typer.testing import CliRunner

V0 = load(Scoring, ROOT / "config/scoring_v0.yaml")
IDS = "abcdefg"
ROLES = dict(zip(IDS, ["mafia", "doctor"] + ["villager"] * 5))


def ev(type, actor=None, targets=(), **payload):
    return Event(run="t", game=0, seq=0, ts="2026-01-01T00:00:00", day=1, phase="x", type=type, actor=actor,
                 targets=list(targets), visibility="truth", payload=payload, scoring_version=0, prompt_version=1)


def log(winner, extra):
    return [ev("role_assigned", a, role=r) for a, r in ROLES.items()] + extra + [ev("game_end", winner=winner)]


def test_breakdowns_exact():
    events = log("town", [
        ev("vote", "c", ["a"]), ev("vote", "d", ["a"]), ev("vote", "b", ["a"]), ev("vote", "a", ["c"]),  # mafia vote: nothing
        ev("vote", "e", ["f"]),
        ev("night_resolution", target="c", protect="c", saved=True, eliminated=None),
        ev("night_resolution", target="c", protect="d", saved=False, eliminated="c"),
        ev("vote_result", tally={}, eliminated="f", tie=False),
        ev("vote_result", tally={}, eliminated="a", tie=False),
        ev("vote_result", tally={}, eliminated=None, tie=True)])
    p = score(events, V0)
    assert p["a"] == dict(points_win=0, points_votes=0, points_saves=0, points_mislynch=1, points_total=1)
    assert p["b"] == dict(points_win=13, points_votes=1, points_saves=2, points_mislynch=0, points_total=16)
    assert p["c"]["points_total"] == 14 and p["d"]["points_votes"] == 1 and p["e"]["points_total"] == 13
    assert score(log("mafia", []), V0)["a"]["points_win"] == 8


def test_rescore_differs_only_in_win_points():
    events = log("town", [ev("vote", "c", ["a"]), ev("night_resolution", target="c", protect="c", saved=True, eliminated=None)])
    v1 = V0.model_copy(update={"version": 1, "win_points": {"town": 5, "mafia": 20}})
    a, b = score(events, V0), score(events, v1)
    for x in IDS:
        assert {k: v for k, v in a[x].items() if k not in ("points_win", "points_total")} == \
               {k: v for k, v in b[x].items() if k not in ("points_win", "points_total")}
    assert a["c"]["points_win"] == 13 and b["c"]["points_win"] == 5


def test_leaderboard_text():
    names = {"a": "Sana", "b": "Dev", "c": "Mira"}
    g = lambda *t: {k: {"points_total": v} for k, v in zip("abc", t)}
    assert leaderboard.render([g(5, 5, 0), g(13, 0, 14)], names) == \
        "LEADERBOARD (after 2 games)\n1. Sana 18 (+13 last game)  2. Mira 14 (+14)  3. Dev 5 (+0)"
    assert leaderboard.render([g(1, 1, 1)], names).endswith("1. Dev 1 (+1 last game)  2. Mira 1 (+1)  3. Sana 1 (+1)")


class Boom(LLMAgent):
    """Raises on its n-th vote across all instances, to crash a game mid-way."""
    count = 0
    fail_at = None

    def vote(self, view):
        Boom.count += 1
        if Boom.count == Boom.fail_at:
            raise RuntimeError("boom")
        return super().vote(view)


def factory(run, rules, scoring, pcfg, llm):
    return {p.agent_id: Boom(p, pcfg, llm, rules, scoring, run.group) for p in pcfg.personas}


def cfg(tmp_path, games=5, rid="s"):
    return load_run(ROOT / "config/run_local.yaml").model_copy(update={"run_id": rid, "games": games})


def rows(path):
    return [{k: v for k, v in r.items() if k not in ("seconds", "cost_usd")} for r in csv.DictReader(open(path))]


def test_crash_resume_matches_uninterrupted(tmp_path):
    Boom.count, Boom.fail_at = 0, None
    run_season(cfg(tmp_path, rid="full"), mock=True, agent_factory=factory, runs_dir=tmp_path)
    Boom.count, Boom.fail_at = 0, 28  # mid game 3 (games 1-2 cast 24 votes)
    with pytest.raises(RuntimeError):
        run_season(cfg(tmp_path, rid="crash"), mock=True, agent_factory=factory, runs_dir=tmp_path)
    d = tmp_path / "crash"
    done = json.loads((d / "state.json").read_text())["last_completed"]
    assert done == 2 and not (d / "games" / f"{done + 1:04d}.jsonl").exists()
    (d / "games" / f"{done + 1:04d}.jsonl").write_text("{}\n")  # crash after the log write, before state
    with pytest.raises(RunExists):
        run_season(cfg(tmp_path, rid="crash"), mock=True, agent_factory=factory, runs_dir=tmp_path)
    Boom.fail_at = None
    run_season(cfg(tmp_path, rid="crash"), mock=True, agent_factory=factory, resume=True, runs_dir=tmp_path)
    assert (d / "games" / f"{done + 1:04d}.partial").exists() and (d / "llm_cache" / f"{done + 1:04d}.partial").exists()
    for f in ("games.csv", "agents.csv"):
        a, b = rows(tmp_path / "full" / f), rows(d / f)
        assert [{k: v for k, v in r.items() if k != "run"} for r in a] == [{k: v for k, v in r.items() if k != "run"} for r in b]
    assert len(rows(d / "games.csv")) == 5
    assert json.loads((d / "manifest.json").read_text())["end"]


def test_csv_columns_and_play_mock(tmp_path):
    run_season(cfg(tmp_path, 1, "c"), mock=True, agent_factory=factory, runs_dir=tmp_path)
    assert next(csv.reader(open(tmp_path / "c/games.csv"))) == compute.GAMES_COLS
    assert ",".join(compute.GAMES_COLS) == ("run,group,game,seed,winner,end_reason,days,mafia,doctor,saves,mislynches,ties,"
        "format_errors,fallbacks,self_leaks,llm_calls,tokens_in,tokens_out,cost_usd,seconds,scoring_version,prompt_version,reflection_issues")
    assert ",".join(compute.AGENTS_COLS) == (
        "run,group,game,agent,role,team,team_won,alive_at_end,days_alive,points_total,points_win,points_votes,"
        "points_saves,points_mislynch,votes_cast,votes_on_mafia,speeches,passes,words_spoken,accusations_made,"
        "times_accused,priority_replies,format_errors,fallbacks,self_leak,memories_active,beliefs_active,lessons_active,reflection_issues")
    assert next(csv.reader(open(tmp_path / "c/agents.csv"))) == compute.AGENTS_COLS
    last = [json.loads(l) for l in (tmp_path / "c/games/0001.jsonl").read_text().splitlines()][-1]
    assert last["type"] == "score"


def test_play_mock_via_cli(monkeypatch, tmp_path):
    import mafia.runner.season as S
    monkeypatch.setattr(S, "ROOT", tmp_path)
    monkeypatch.setattr(S.run_season, "__kwdefaults__", {**S.run_season.__kwdefaults__, "runs_dir": tmp_path / "runs"})
    r = CliRunner().invoke(app, ["play", "--run-config", str(ROOT / "config/run_local.yaml"), "--games", "1", "--mock"])
    assert r.exit_code == 0 and ("Day" in r.output or "day" in r.output)
    assert (tmp_path / "runs/local-test/games.csv").exists()
    r = CliRunner().invoke(app, ["play", "--run-config", str(ROOT / "config/run_local.yaml"), "--mock"])
    assert r.exit_code == 1 and "--resume" in r.output


def test_resume_drops_csv_rows_ahead_of_state(tmp_path):
    Boom.count, Boom.fail_at = 0, None
    run_season(cfg(tmp_path, 2, "ahead"), mock=True, agent_factory=factory, runs_dir=tmp_path)
    d = tmp_path / "ahead"
    (d / "state.json").write_text(json.dumps({"last_completed": 1}))  # game 2 row is ahead of state
    with (d / "games.csv").open("a") as f:
        f.write("torn,row")  # crash mid-write
    run_season(cfg(tmp_path, 2, "ahead"), mock=True, agent_factory=factory, resume=True, runs_dir=tmp_path)
    assert [r["game"] for r in rows(d / "games.csv")] == ["1", "2"]
    assert [r["game"] for r in rows(d / "agents.csv")] == ["1"] * 7 + ["2"] * 7
    from mafia.engine.seeds import game_seed
    assert rows(d / "games.csv")[1]["seed"] == str(game_seed(cfg(tmp_path).seed, 1))


def test_preflight_missing_key_creates_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    import yaml
    m = tmp_path / "m.yaml"
    m.write_text(yaml.safe_dump(yaml.safe_load((ROOT / "config/model_local.yaml").read_text())
                                | {"api_key_env": "NO_SUCH_KEY_VAR_X"}))
    c = cfg(tmp_path, 1, "pre").model_copy(update={"model_file": m})
    with pytest.raises(RuntimeError, match="NO_SUCH_KEY_VAR_X"):
        run_season(c, agent_factory=factory, runs_dir=tmp_path)
    assert not (tmp_path / "pre").exists()


def test_manifest_provenance_and_resume_refusal(tmp_path, monkeypatch):
    from mafia.logs import manifest as M
    run_season(cfg(tmp_path, 1, "prov"), mock=True, agent_factory=factory, runs_dir=tmp_path)
    d = tmp_path / "prov"
    man = json.loads((d / "manifest.json").read_text())
    assert man["prompts_sha256"] == M.prompts_hash() and man["git_dirty"] in (True, False, None)
    for n in ("run", "model", "scoring", "rules", "personas", "memory"):
        assert (d / "config" / f"{n}.yaml").exists()
    assert (d / "config/rules.yaml").read_text() == (ROOT / "config/rules.yaml").read_text()
    assert not list(d.glob("*.tmp"))
    run_season(cfg(tmp_path, 2, "prov").model_copy(update={"games": 2}), mock=True, agent_factory=factory,
               resume=True, runs_dir=tmp_path)  # same inputs: fine
    (d / "config/rules.yaml").write_text("changed: 1\n")
    with pytest.raises(M.InputsChanged, match="rules.yaml"):
        run_season(cfg(tmp_path, 2, "prov"), mock=True, agent_factory=factory, resume=True, runs_dir=tmp_path)
    (d / "config/rules.yaml").write_text((ROOT / "config/rules.yaml").read_text())
    man["prompts_sha256"] = "0" * 64
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(M.InputsChanged, match="prompts"):
        run_season(cfg(tmp_path, 2, "prov"), mock=True, agent_factory=factory, resume=True, runs_dir=tmp_path)


def test_llm_call_events_drop_reasoning(tmp_path):
    class Rec(ScriptedLLM):
        async def acall(self, *a):
            out, rec = await super().acall(*a)
            return out, rec | {"reasoning": "secret thoughts"}
    run_season(cfg(tmp_path, 1, "rs"), mock=True, runs_dir=tmp_path,
               agent_factory=lambda run, rules, scoring, pcfg, llm: factory(run, rules, scoring, pcfg, Rec(run.seed, {p.agent_id: p.name for p in pcfg.personas})))
    text = (tmp_path / "rs/games/0001.jsonl").read_text()
    assert '"llm_call"' in text and "secret thoughts" not in text
