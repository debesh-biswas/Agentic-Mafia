"""DRD 12 / 9.4: cumulative standings from score events."""


def standings(games_points: list[dict[str, dict]]) -> dict[str, int]:
    """games_points: one `points` dict (agent_id -> breakdown) per game, oldest first."""
    tot: dict[str, int] = {}
    for pts in games_points:
        for a, b in pts.items():
            tot[a] = tot.get(a, 0) + b["points_total"]
    return tot


def render(games_points: list[dict[str, dict]], names: dict[str, str]) -> str:
    tot, last = standings(games_points), games_points[-1] if games_points else {}
    ranked = sorted(tot, key=lambda a: (-tot[a], names[a]))  # ties by name: deterministic
    parts = [f"{i}. {names[a]} {tot[a]} (+{last.get(a, {}).get('points_total', 0)}{' last game' if i == 1 else ''})"
             for i, a in enumerate(ranked, 1)]
    return f"LEADERBOARD (after {len(games_points)} games)\n" + "  ".join(parts)
