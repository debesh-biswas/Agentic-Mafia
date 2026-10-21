"""DRD 11.4: per-game and per-agent CSV rows, derived from events only."""
import csv
from pathlib import Path

from mafia.logs.events import Event
from mafia.scoring.score import score, team_of

IN_GAME = {"plan", "night", "speak", "vote", "last_words"}  # other fallbacks are reflection bookkeeping

GAMES_COLS = "run group game seed winner end_reason days mafia doctor saves mislynches ties format_errors fallbacks " \
             "self_leaks llm_calls tokens_in tokens_out cost_usd seconds scoring_version prompt_version reflection_issues".split()
AGENTS_COLS = "run group game agent role team team_won alive_at_end days_alive points_total points_win points_votes " \
              "points_saves points_mislynch votes_cast votes_on_mafia speeches passes words_spoken accusations_made " \
              "times_accused priority_replies format_errors fallbacks self_leak memories_active beliefs_active " \
              "lessons_active reflection_issues".split()


def rows(events: list[Event], cfg, run: str, group: str, game: int, seed: int, seconds: float):
    """game is 1-based. Returns (games row, [agents rows])."""
    pts = score(events, cfg)
    roles = {e.actor: e.payload["role"] for e in events if e.type == "role_assigned"}
    end = next(e for e in events if e.type == "game_end").payload
    gone: dict[str, int] = {}  # agent -> day eliminated
    for e in events:
        if e.type == "night_resolution" and e.payload["eliminated"]:
            gone[e.payload["eliminated"]] = e.day
        elif e.type == "vote_result" and e.payload["eliminated"]:
            gone[e.payload["eliminated"]] = e.day
    calls = [e for e in events if e.type == "llm_call"]
    fb = [e for e in events if e.type == "fallback" and e.payload.get("call_type") in IN_GAME]
    issues = [e for e in events if e.type == "fallback" and e.payload.get("call_type") not in IN_GAME]
    speech = [e for e in events if e.type == "speech"]
    results = [e.payload for e in events if e.type == "vote_result"]
    fmt = lambda es, a=None: sum(e.payload.get("format_errors", 0) for e in es if a in (None, e.actor))
    g = dict(
        run=run, group=group, game=game, seed=seed, winner=end["winner"], end_reason=end["end_reason"], days=end["days"],
        mafia=next(a for a, r in roles.items() if r == "mafia"), doctor=next(a for a, r in roles.items() if r == "doctor"),
        saves=sum(e.payload["saved"] for e in events if e.type == "night_resolution"),
        mislynches=sum(1 for p in results if p["eliminated"] and roles[p["eliminated"]] != "mafia"),
        ties=sum(p["tie"] for p in results), format_errors=fmt(calls), fallbacks=len(fb),
        self_leaks=sum(e.payload["self_leak"] for e in speech), llm_calls=len(calls),
        tokens_in=sum(e.payload.get("tokens_in", 0) for e in calls), tokens_out=sum(e.payload.get("tokens_out", 0) for e in calls),
        cost_usd=round(sum(e.payload.get("cost") or 0 for e in calls), 6), seconds=round(seconds, 2),
        scoring_version=cfg.version, prompt_version=events[0].prompt_version, reflection_issues=len(issues))
    agents = []
    for a, role in roles.items():
        mine = [e for e in speech if e.actor == a]
        votes = [e for e in events if e.type == "vote" and e.actor == a]
        agents.append(dict(
            run=run, group=group, game=game, agent=a, role=role, team=team_of(role), team_won=team_of(role) == end["winner"],
            alive_at_end=a not in gone, days_alive=gone.get(a, end["days"]), **pts[a],
            votes_cast=len(votes), votes_on_mafia=sum(roles[v.targets[0]] == "mafia" for v in votes), speeches=len(mine),
            passes=sum(1 for e in events if e.type == "pass" and e.actor == a),
            words_spoken=sum(e.payload["word_count"] for e in mine),
            accusations_made=sum(len(e.payload["accuses"]) for e in mine),
            times_accused=sum(a in e.payload["accuses"] for e in speech),
            priority_replies=sum(e.payload["priority_reply"] for e in mine),
            format_errors=fmt(calls, a), fallbacks=sum(1 for e in fb if e.actor == a),
            self_leak=sum(e.payload["self_leak"] for e in mine), memories_active=0, beliefs_active=0, lessons_active=0,
            reflection_issues=sum(1 for e in issues if e.actor == a)))
    return g, agents


def append(path: Path, cols: list[str], new: list[dict]) -> None:
    fresh = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, cols)
        if fresh:
            w.writeheader()
        w.writerows(new)


def truncate(path: Path, last_game: int) -> None:
    """Drop rows past last_game (a crash between the CSV append and state.json)."""
    if not path.exists():
        return
    import os
    with path.open(newline="") as f:
        rd = csv.DictReader(f)
        cols, keep = rd.fieldnames, []
        for r in rd:
            try:
                if int(r["game"]) <= last_game and None not in r and None not in r.values():
                    keep.append(r)
            except (TypeError, ValueError):  # malformed (torn) row
                pass
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(keep)
    os.replace(tmp, path)
