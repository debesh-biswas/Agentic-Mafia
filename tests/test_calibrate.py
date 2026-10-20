import csv
import json

import pytest
from mafia.cli import app
from mafia.config import Scoring, load
from mafia.runner.calibrate import propose, write_scoring
from typer.testing import CliRunner

BASE = Scoring(version=0, win_points={"town": 13, "mafia": 8}, bonus={"town_vote_on_mafia": 1, "doctor_successful_save": 2,
                                                                       "mafia_town_mislynch": 1})
WINNERS = ["town"] * 4 + ["mafia"] * 6  # n=10, town 4/10 = 0.4, mafia 6/10 = 0.6


def test_rates_and_points():
    p = propose(WINNERS, BASE)
    assert p["n"] == 10
    assert p["town"]["wins"] == 4 and p["town"]["rate"] == pytest.approx(0.4)
    assert p["mafia"]["wins"] == 6 and p["mafia"]["rate"] == pytest.approx(0.6)
    # Python's round() is banker's rounding: round(12.5) == 12 (5 / 0.4 = 12.5 exactly), round(8.33) == 8.
    assert p["town"]["win_points"] == 12
    assert p["mafia"]["win_points"] == 8


def test_wilson_interval_by_hand():
    # Wilson score interval for k successes in n trials, z = 1.96, p = k / n:
    #   center = (p + z^2 / 2n) / (1 + z^2 / n)
    #   half   = z * sqrt(p(1 - p) / n + z^2 / 4n^2) / (1 + z^2 / n)
    # n = 10: z^2 = 3.8416, z^2/n = 0.38416, z^2/2n = 0.19208, z^2/4n^2 = 0.009604, denominator = 1.38416.
    # k = 4 (p = 0.4): center = 0.59208 / 1.38416 = 0.42775; half = 1.96 * sqrt(0.033604) / 1.38416 = 0.25956
    #                  -> [0.16819, 0.68731]
    # k = 6 (p = 0.6): center = 0.79208 / 1.38416 = 0.57225; half = 0.25956 (p(1-p) is the same)
    #                  -> [0.31269, 0.83181]
    p = propose(WINNERS, BASE)
    lo, hi = p["town"]["ci"]
    assert lo == pytest.approx(0.168, abs=1e-3) and hi == pytest.approx(0.687, abs=1e-3)
    lo, hi = p["mafia"]["ci"]
    assert lo == pytest.approx(0.313, abs=1e-3) and hi == pytest.approx(0.832, abs=1e-3)


def test_zero_wins_raises():
    with pytest.raises(ValueError, match="town.*no wins"):
        propose(["mafia"] * 5, BASE)
    with pytest.raises(ValueError, match="mafia.*no wins"):
        propose(["town"] * 5, BASE)


def write_run(runs_dir, winners, run_id="calib-1"):
    d = runs_dir / run_id
    d.mkdir(parents=True)
    with (d / "games.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game", "group", "winner"])
        w.writeheader()
        for i, win in enumerate(winners):
            w.writerow({"game": i + 1, "group": "control", "winner": win})
    (d / "manifest.json").write_text(json.dumps({"run_config": {"scoring_config": "config/scoring_v0.yaml"}}))
    return d


def test_cli_writes_v1_and_refuses_overwrite(tmp_path):
    write_run(tmp_path, WINNERS)
    out = tmp_path / "scoring_v1.yaml"
    args = ["calibrate", "--run", "calib-1", "--runs-dir", str(tmp_path), "--out", str(out)]
    r = CliRunner().invoke(app, args)
    assert r.exit_code == 0, r.output
    assert "town" in r.output and "mafia" in r.output and "12" in r.output
    cfg = load(Scoring, out)
    assert cfg.version == 1
    assert cfg.win_points == {"town": 12, "mafia": 8}
    assert cfg.bonus == BASE.bonus
    assert out.read_text().startswith("#") and "calib-1" in out.read_text().splitlines()[0]

    r = CliRunner().invoke(app, args)
    assert r.exit_code == 1 and "--force" in r.output
    assert CliRunner().invoke(app, args + ["--force"]).exit_code == 0


def test_write_scoring_roundtrip(tmp_path):
    out = tmp_path / "v1.yaml"
    write_scoring(out, "calib-1", propose(WINNERS, BASE), BASE)
    assert load(Scoring, out).win_points == {"town": 12, "mafia": 8}
