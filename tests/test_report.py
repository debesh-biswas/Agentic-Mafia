import json
import re

from mafia.cli import app
from mafia.config import ROOT, load_run
from mafia.logs.events import Event
from mafia.metrics import report as rep
from mafia.renderers.scene_script import beats
from mafia.runner.season import run_season
from typer.testing import CliRunner

HEADS = ["Role win rates", "Role-adjusted points", "Vote accuracy", "Per-agent table", "Reliability", "Memory stats",
         "Paired comparison"]


def make(tmp_path, groups=("control", "memory")):
    for g in groups:
        c = load_run(ROOT / "config/run_local.yaml").model_copy(update={"run_id": g, "games": 8, "group": g})
        run_season(c, mock=True, runs_dir=tmp_path)


def test_report_sections_and_numbers(tmp_path):
    make(tmp_path)
    r = CliRunner().invoke(app, ["report", "--runs", "control", "--runs", "memory", "--runs-dir", str(tmp_path)])
    assert r.exit_code == 0, r.output
    text = open(r.output.strip()).read()
    for h in HEADS:
        assert h in text
    assert "Skipped" not in text
    m = re.search(r"mean (-?[\d.]+), bootstrap 95% CI \[(-?[\d.]+), (-?[\d.]+)\]", text)
    mean, lo, hi = map(float, m.groups())
    assert lo <= mean <= hi
    row = next(l for l in text.splitlines() if l.startswith("| all") and "|" in l)
    assert all(0 <= float(x) <= 1 for x in re.findall(r"\b\d\.\d{3}\b", row))


def test_bootstrap_deterministic_and_control_only(tmp_path):
    make(tmp_path)
    runs = lambda *n: [rep.load(tmp_path / x) for x in n]
    a = rep.build(runs("control", "memory"), {})
    assert a == rep.build(runs("control", "memory"), {})
    only = rep.build(runs("control"), {})
    assert "## 7" in only and "Skipped" in only and "bootstrap" not in only
    no_ctl = rep.build(runs("memory"), {})
    assert "no control run" in no_ctl


def ev(seq, type, actor=None, targets=(), vis="public", **payload):
    return Event(run="t", game=0, seq=seq, ts="2026-01-01T00:00:00", day=2, phase="discussion", type=type, actor=actor,
                 targets=list(targets), visibility=vis, payload=payload, scoring_version=0, prompt_version=1)


def test_scene_beats():
    events = [
        ev(0, "role_assigned", "a", vis="truth", role="mafia"),
        ev(1, "night_notes", "a", vis="private:a", notes="secret"),
        ev(2, "morning_announcement", text="Morning of day 2: nobody was eliminated."),
        ev(3, "speech", "a", ["b"], message="b is acting very odd today friends", accuses=["b"], expression="nervous",
           urgency=3, self_leak=True),
        ev(4, "speech", "b", message="no", accuses=[], expression="calm"),
        ev(5, "last_words", "b", message="bye all", expression="sad"),
        ev(6, "vote", "a", ["b"]),
        ev(7, "llm_call", "a", vis="debug"),
    ]
    b = beats(events, {"a": 1, "b": 2})
    assert [x["kind"] for x in b] == ["morning_announcement", "speech", "speech", "last_words", "vote"]
    assert [x["beat"] for x in b] == [0, 1, 2, 3, 4] and [x["seq"] for x in b] == [2, 3, 4, 5, 6]
    assert b[1]["est_duration_s"] == 3.2 and b[1]["look_at"] == ["b"] and b[1]["seat"] == 1
    assert [x["camera_hint"] for x in b] == ["wide", "closeup", "closeup", "closeup", "wide"]
    assert "look_at" not in b[2] and b[0]["actor"] is None and b[4]["est_duration_s"] == 0.4
    blob = json.dumps(b)
    assert "secret" not in blob and "mafia" not in blob and "self_leak" not in blob and "urgency" not in blob
