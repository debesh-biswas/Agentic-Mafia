"""Summary regeneration (DRD 8.1 call 3)."""
from mafia.llm import schemas as S
from mafia.llm.client import LLMFormatError
from mafia.llm.prompts import template


def summarize(store, names: dict[str, str], me: str, call, game: int, update_line: str, insight_line: str) -> list[tuple[str, str, dict]]:
    vis = f"private:{me}"
    cur = store.get_summaries()
    text = "\n".join([f"- {n}: {cur[a]}" for a, n in names.items() if a != me and a in cur]
                     + [f"Yourself: {cur.get('self', '')}"])
    try:
        res = call("summaries", S.Summaries, template("summaries").format(
            current_summaries=text, update_style_line=update_line, self_insight_line=insight_line))
    except LLMFormatError as e:
        return [("fallback", "debug", dict(call_type="summaries", reason=str(e), chosen=None))]
    by_name = {n.lower(): a for a, n in names.items() if a != me}
    friends = {by_name[n.lower()]: t for n, t in res.friends.items() if n.lower() in by_name}  # unknown names dropped
    store.set_summaries(game, {**friends, "self": res.self})
    return [("summary_update", vis, dict(friends=friends, self=res.self))]
