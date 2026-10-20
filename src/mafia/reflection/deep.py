"""Deep reflection (DRD 8.2): every deep_reflection_every_games, look for patterns in one's own judgment."""
import json

from mafia.llm import schemas as S
from mafia.llm.client import LLMFormatError
from mafia.llm.prompts import template
from mafia.memory.retrieval import render
from mafia.reflection.reflect import apply_insights

LAST = 5


def deep_reflect(store, names, call, me: str, game: int) -> list[tuple[str, str, dict]]:
    """call(call_type, schema, user) -> parsed. Returns event specs. Shown ids = the episodic and helped_or_hurt
    memories in the prompt; `hoh_ids` meta is maintained by reflect()."""
    epi = sorted(store.active(("episodic",)), key=lambda m: (m.game_created, m.id))[-LAST:]
    hoh = [m for m in (store.get(i) for i in json.loads(store.meta_get("hoh_ids", "[]"))[-LAST:]) if m.status == "active"]
    if not epi and not hoh:
        return []
    try:
        res = call("deep_reflect", S.DeepReflect, template("deep_reflect").format(
            last_episodic=render(epi, names) or "(none)", last_helped_or_hurt=render(hoh, names) or "(none)",
            self_summary=store.get_summaries().get("self", "")))
    except LLMFormatError as e:
        return [("fallback", "debug", dict(call_type="deep_reflect", reason=str(e), chosen=None))]
    # kinds are reflection or belief only (DRD 8.2); lessons are dropped
    keep = [i for i in res.insights if i.kind != "lesson"]
    out = [("fallback", "debug", dict(call_type="deep_reflect", reason=f"dropped {len(res.insights) - len(keep)} lesson insight(s)",
                                      chosen=None))] if len(keep) < len(res.insights) else []
    apply_insights(store, names, f"private:{me}", game, keep, {m.id: m for m in epi + hoh}, out, "deep_reflect")
    return out
