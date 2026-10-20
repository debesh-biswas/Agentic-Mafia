from collections import Counter

from mafia.engine.state import PlayerState
from mafia.types import Role, Team


def living(players: dict[str, PlayerState]) -> list[str]:
    return [p.agent_id for p in players.values() if p.alive]


def check_win(players: dict[str, PlayerState]) -> tuple[Team, str] | None:
    """DRD 5.4."""
    mafia = sum(p.alive and p.role is Role.MAFIA for p in players.values())
    town = sum(p.alive and p.role is not Role.MAFIA for p in players.values())
    if mafia == 0:
        return "town", "mafia_voted_out"
    if mafia >= town:
        return "mafia", "parity"
    return None


def mafia_targets(players: dict[str, PlayerState]) -> list[str]:
    return [p.agent_id for p in players.values() if p.alive and p.role is not Role.MAFIA]


def doctor_protects(players: dict[str, PlayerState], doctor: str, self_blocked: bool) -> list[str]:
    return [p for p in living(players) if not (self_blocked and p == doctor)]


def vote_targets(players: dict[str, PlayerState], voter: str) -> list[str]:
    return [p for p in living(players) if p != voter]


def tally(votes: list[tuple[str, str]]) -> tuple[dict[str, int], str | None]:
    """(counts, unique plurality winner or None on a tie)."""
    counts = Counter(t for _, t in votes)
    top = max(counts.values())
    leaders = [p for p, c in counts.items() if c == top]
    return dict(counts), leaders[0] if len(leaders) == 1 else None
