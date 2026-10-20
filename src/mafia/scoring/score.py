"""DRD 12: pure scoring from events. Never reads game state."""
from mafia.config import Scoring
from mafia.logs.events import Event

KEYS = ("points_win", "points_votes", "points_saves", "points_mislynch")


def team_of(role: str) -> str:
    return "mafia" if role == "mafia" else "town"


def score(events: list[Event], cfg: Scoring) -> dict[str, dict[str, int]]:
    roles = {e.actor: e.payload["role"] for e in events if e.type == "role_assigned"}
    out = {a: dict.fromkeys(KEYS, 0) for a in roles}
    winner = next((e.payload["winner"] for e in events if e.type == "game_end"), None)
    for a, r in roles.items():
        if team_of(r) == winner:
            out[a]["points_win"] = cfg.win_points[winner]
    for e in events:
        if e.type == "vote" and roles[e.actor] != "mafia" and roles[e.targets[0]] == "mafia":
            out[e.actor]["points_votes"] += cfg.bonus["town_vote_on_mafia"]
        elif e.type == "night_resolution" and e.payload["saved"]:
            for a, r in roles.items():
                if r == "doctor":
                    out[a]["points_saves"] += cfg.bonus["doctor_successful_save"]
        elif e.type == "vote_result" and e.payload["eliminated"] and roles[e.payload["eliminated"]] != "mafia":
            for a, r in roles.items():
                if r == "mafia":
                    out[a]["points_mislynch"] += cfg.bonus["mafia_town_mislynch"]
    for b in out.values():
        b["points_total"] = sum(b[k] for k in KEYS)
    return out
