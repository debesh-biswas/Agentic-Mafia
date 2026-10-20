from collections.abc import Iterable

from rich.console import Console

from mafia.logs.events import Event


def visible(events: Iterable[Event], view: str) -> list[Event]:
    """public | truth (all but debug) | <agent_id> (public + that agent's private)."""
    if view == "truth":
        return [e for e in events if e.visibility != "debug"]
    ok = {"public", f"private:{view}"} if view != "public" else {"public"}
    return [e for e in events if e.visibility in ok
            or (e.type == "role_assigned" and e.actor == view)]  # an agent knows its own role


class TextRenderer:
    def __init__(self, names: dict[str, str], console: Console | None = None):
        self.names, self.console = names, console or Console(highlight=False)

    def n(self, pid):
        return self.names.get(pid, pid)

    def line(self, e: Event) -> tuple[str, bool] | None:
        """(text, is_private_or_truth) or None to skip."""
        p, a = e.payload, self.n(e.actor)
        t = e.type
        if t == "game_start":
            return f"GAME {e.game + 1}  seed {p['seed']}", False
        if t == "role_assigned":
            return f"{a} is {p['role']}.", True
        if t == "plan":
            return f"{a} plans: {p['plan']}", True
        if t == "night_notes":
            return f"{a} notes: {p['notes']} (urgency {p['urgency']})", True
        if t == "night_action":
            return f"{a} chooses to {p['action']} {self.n(e.targets[0])}.", True
        if t == "night_resolution":
            return (f"Night result: target {self.n(p['target'])}, protected {self.n(p['protect'])}, "
                    f"saved={p['saved']}.", True)
        if t == "morning_announcement":
            return p["text"], False
        if t == "speech":
            return f"{a}: {p['message']}", False
        if t == "pass":
            return f"{a} passes.", False
        if t == "vote":
            return None  # collected into the vote_result line
        if t == "vote_result":
            return None
        if t == "last_words":
            return f"{a} (last words): {p['message']}", False
        if t == "game_end":
            return f"GAME OVER: {p['winner']} wins ({p['end_reason']}) on day {p['days']}.", False
        if t == "role_reveal":
            return "Roles: " + ", ".join(f"{self.n(k)} {v}" for k, v in p["roles"].items()), False
        return f"[{t}] {p}", True  # reflection, memory, score, etc.

    def render_game(self, events: Iterable[Event], view: str) -> None:
        out, votes, last = self.console.print, [], (None, None)
        for e in visible(events, view):
            if e.phase in ("night", "morning") and e.day != last[0]:
                out(f"\nNIGHT {e.day}")
            if e.phase == "discussion" and last[1] != "discussion":
                out(f"\nDAY {e.day}")
            last = (e.day if e.phase in ("night", "morning") else last[0], e.phase)
            if e.type == "vote":
                votes.append(f"{self.n(e.actor)}→{self.n(e.targets[0])}")
            elif e.type == "vote_result":
                res = "tie, nobody voted out" if e.payload["tie"] else f"{self.n(e.payload['eliminated'])} voted out"
                out(f"Votes: {', '.join(votes)}. {res}.", soft_wrap=True)
                votes = []
            elif (r := self.line(e)):
                text, private = r
                out(text, style="dim" if private else None, markup=False, soft_wrap=True)
