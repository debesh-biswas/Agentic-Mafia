"""Post-game reflection (DRD 8.1): questions, retrieval, insights, memory writes. Calls 1 and 2."""
from dataclasses import dataclass, field

from mafia.llm import schemas as S
from mafia.llm.client import LLMFormatError
from mafia.llm.prompts import template
from mafia.memory.retrieval import render, retrieve

HELPED_OR_HURT_IMPORTANCE = 6  # DRD 8.1 default


@dataclass
class GameOutcomeForAgent:
    """Everything one agent may know after the game: all roles, its own notes and points, never others' notes,
    night actions or the transcript."""
    agent_id: str
    game_no: int
    winner: str
    end_reason: str
    days: int
    roles: dict[str, str]  # name -> role, seat order
    my_role: str
    notes_history: dict[int, str]
    points: dict[str, int]
    leaderboard_text: str
    update_style_line: str = ""
    self_insight_line: str = ""

    @property
    def won(self) -> bool:
        return (self.my_role == "mafia") == (self.winner == "mafia")


def build_outcome(events, agent_id, points, names, leaderboard_text, game_no) -> GameOutcomeForAgent:
    """Built from the finished event log. Only role_reveal/game_end (public) and the agent's own night_notes are read."""
    end = next(e.payload for e in events if e.type == "game_end")
    reveal = next(e.payload["roles"] for e in events if e.type == "role_reveal")
    notes = {e.day: e.payload["notes"] for e in events if e.type == "night_notes" and e.actor == agent_id}
    return GameOutcomeForAgent(agent_id, game_no, end["winner"], end["end_reason"], end["days"],
                               {names[a]: r for a, r in reveal.items()}, reveal[agent_id], notes,
                               points[agent_id], leaderboard_text)


def outcome_fields(o: GameOutcomeForAgent) -> dict:
    pts = o.points
    return dict(
        outcome_line=f"The {o.winner.capitalize()} won ({o.end_reason.replace('_', ' ')}) on day {o.days}",
        all_roles=", ".join(f"{n} ({r})" for n, r in o.roles.items()), my_role=o.my_role,
        points_breakdown=f"win {pts['points_win']}, votes {pts['points_votes']}, saves {pts['points_saves']}, "
                         f"mislynch {pts['points_mislynch']}, total {pts['points_total']}",
        leaderboard_block=o.leaderboard_text,
        notes_history="\n".join(f"Night {d}: {t}" for d, t in sorted(o.notes_history.items())) or "(none)",
        update_style_line=o.update_style_line, self_insight_line=o.self_insight_line)


def _fallback(call_type, reason, chosen=None):
    return ("fallback", "debug", dict(call_type=call_type, reason=reason, chosen=chosen))


def reflect(store, mcfg, names: dict[str, str], call, o: GameOutcomeForAgent) -> list[tuple[str, str, dict]]:
    """call(call_type, schema, user) -> parsed; may raise LLMFormatError. Returns (type, visibility, payload) specs."""
    vis, game, out = f"private:{o.agent_id}", o.game_no, []
    f = outcome_fields(o)
    try:
        qs = call("reflect", S.Questions, template("reflect_questions").format(**f)).questions[:mcfg.reflection.questions]
    except LLMFormatError as e:
        return [_fallback("reflect", f"questions: {e}")]
    shown: dict[int, object] = {}
    per_q = []
    for q in qs:
        got = retrieve(store, q, game, mcfg.retrieval, k=mcfg.reflection.memories_per_question)
        per_q += [m for m in got if m.id not in {x.id for x in per_q}]
    joined = store.embedder.embed(" ".join(qs))
    cands = sorted(store.active(("belief", "lesson")), key=lambda m: (-float(m.embedding @ joined), m.id))
    related = cands[:mcfg.related_beliefs_for_reflection]
    for m in per_q + related:
        shown[m.id] = m
    try:
        res = call("reflect", S.ReflectInsights, template("reflect_insights").format(
            questions="\n".join(f"- {q}" for q in qs), retrieved_for_questions=render(per_q, names) or "(none)",
            related_beliefs=render(related, names) or "(none)", **f))
    except LLMFormatError as e:
        return [_fallback("reflect", f"insights: {e}")]
    by_name = {n.lower(): a for a, n in names.items()}
    out.append(("reflection", vis, dict(questions=qs, insights=[i.model_dump() for i in res.insights],
                                        helped_or_hurt=res.helped_or_hurt, game_summary=res.game_summary)))

    def wrote(i, action, text, type):
        out.append(("memory_write", vis, dict(memory_id=i, type=type, action=action, text=text)))

    for ins in res.insights[:mcfg.reflection.max_insights]:
        cites = [c for c in ins.cites if c in shown]
        about = [a for a in (by_name.get(n.lower(), "self" if n.lower() == "self" else None) for n in ins.about) if a]
        action, tid = ins.action, ins.target_id
        if action != "add" and not (tid in shown and store.get(tid).status == "active"):
            out.append(_fallback("reflect", f"invalid target_id {tid} for {action}; treated as add", "add"))
            action = "add"
        kw = dict(game=game, importance=ins.importance, confidence=ins.confidence)
        if action == "update":
            store.update(tid, ins.text, ins.confidence, ins.importance, game)
            wrote(tid, "update", ins.text, store.get(tid).type)
        elif action == "supersede":
            new = store.supersede(tid, ins.text, cites=cites, **kw)
            wrote(new, "supersede", ins.text, ins.kind)
            out.append(("memory_supersede", vis, dict(old_id=tid, new_id=new)))
        else:
            i = store.add(ins.kind, ins.text, about=about, role_context=ins.role_context, cites=cites, **kw)
            wrote(i, "add", ins.text, ins.kind)
    store.touch([k for k in res.keep_ids if k in shown], game)
    i = store.add("reflection", res.helped_or_hurt, game=game, importance=HELPED_OR_HURT_IMPORTANCE)
    wrote(i, "add", res.helped_or_hurt, "reflection")
    i = store.add("episodic", res.game_summary, game=game, importance=res.summary_importance)
    wrote(i, "add", res.game_summary, "episodic")
    return out
