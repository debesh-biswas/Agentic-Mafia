"""DRD 12: calibration from a control run's games.csv -> proposed scoring v1."""
import csv
from math import sqrt
from pathlib import Path

import yaml
from mafia.config import Scoring

TEAMS = ("town", "mafia")
Z = 1.96  # 95% two-sided
TARGET_POINTS = 5  # DRD 12: win_points = round(5 / win_rate)


def wilson(k: int, n: int, z: float = Z) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials."""
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def propose(winners: list[str], base: Scoring) -> dict:
    """Win counts, rates, Wilson intervals and proposed win_points per team. Raises if a team never won."""
    bad = sorted(set(winners) - set(TEAMS))
    if bad:
        raise ValueError(f"unexpected winner values {bad}; expected {list(TEAMS)}")
    n = len(winners)
    out = {"n": n}
    for team in TEAMS:
        wins = winners.count(team)
        if wins == 0:
            raise ValueError(f"{team} had no wins in {n} games; cannot propose win_points (rate would be 0)")
        rate = wins / n
        lo, hi = wilson(wins, n)
        # Python round() is banker's rounding: round(12.5) == 12
        out[team] = {"wins": wins, "rate": rate, "ci": [lo, hi], "win_points": round(TARGET_POINTS / rate)}
    return out


def write_scoring(path: Path, run_id: str, proposal: dict, base: Scoring) -> None:
    """Write scoring v1 YAML: proposed win_points, bonuses copied from the run's scoring config."""
    cfg = {"version": 1, "win_points": {t: proposal[t]["win_points"] for t in TEAMS}, "bonus": base.bonus}
    header = f"# Proposed from run {run_id}, {proposal['n']} games. Review before use.\n"
    Path(path).write_text(header + yaml.safe_dump(cfg, sort_keys=False))


def winners_from_run(run_dir: Path) -> list[str]:
    with (run_dir / "games.csv").open(newline="") as f:
        return [r["winner"] for r in csv.DictReader(f)]
