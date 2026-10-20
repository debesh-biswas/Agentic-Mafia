"""DRD 15 scene-script stub: public events -> beats for a future animator. Reads nothing but public events."""
from mafia.logs.events import Event

KINDS = {"speech": "closeup", "last_words": "closeup", "vote": "wide", "morning_announcement": "wide"}


def beats(events: list[Event], seats: dict[str, int]) -> list[dict]:
    out = []
    for e in events:
        if e.visibility != "public" or e.type not in KINDS:
            continue
        text = e.payload.get("message") or e.payload.get("text") or ""
        b = dict(beat=len(out), seq=e.seq, kind=e.type, actor=e.actor, seat=seats.get(e.actor), targets=list(e.targets),
                 text=text, expression=e.payload.get("expression"),
                 est_duration_s=round(0.4 + len(text.split()) / 2.5, 2), camera_hint=KINDS[e.type],
                 day=e.day, phase=e.phase)
        if e.type == "speech" and e.targets:
            b["look_at"] = list(e.targets)
        out.append(b)
    return out
