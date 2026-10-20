import sqlite3

from mafia.cli import app
from mafia.config import ROOT, load_run
from mafia.renderers.dashboard import write_dashboard
from mafia.runner.season import run_season
from typer.testing import CliRunner


def season(tmp_path, cfg, games, **kw):
    run = load_run(ROOT / cfg).model_copy(update={"run_id": "d", "games": games})
    run_season(run, mock=True, runs_dir=tmp_path, **kw)
    return tmp_path / "d"


def test_memory_season_dashboard(tmp_path):
    seen = []
    d = season(tmp_path, "config/run_local_memory.yaml", 6, dashboard_every=5,
               on_game=lambda ev, g: seen.append((g, "Game 6 —" in (tmp_path / "d/dashboard.html").read_text())))
    assert seen[4] == (5, False)  # written at game 5, before game 6 exists
    html = (d / "dashboard.html").read_text()
    assert "Every 5 games" in html and "<td>1–5</td>" in html and "<td>6 (so far)</td>" in html
    assert "Game 6 of 6" in html and "Game 6 —" in html and "Full transcript" in html
    assert "What the players believe now" in html and "beliefs" in html


def test_control_has_no_memory_section(tmp_path):
    d = season(tmp_path, "config/run_local.yaml", 2, dashboard_every=5)
    assert "What the players believe now" not in (d / "dashboard.html").read_text()  # written at last game


def test_model_text_escaped_and_broken_db(tmp_path):
    d = season(tmp_path, "config/run_local_memory.yaml", 1)
    log = d / "games/0001.jsonl"
    log.write_text(log.read_text().replace("\"message\":\"", "\"message\":\"<script>x</script>", 1))
    for f in (d / "memory").glob("*.sqlite"):
        f.write_bytes(b"not a database")
    html = write_dashboard(d).read_text()
    assert "<script>" not in html and "&lt;script&gt;x" in html and "updating…" in html


def test_dashboard_cli(tmp_path):
    season(tmp_path, "config/run_local.yaml", 1)
    r = CliRunner().invoke(app, ["dashboard", "--run", "d", "--runs-dir", str(tmp_path)])
    assert r.exit_code == 0 and r.output.strip().endswith("dashboard.html")
