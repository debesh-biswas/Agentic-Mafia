"""Compression and fading (DRD 7.5). Both run in the memory phase of every compression.every_games-th game."""
import json
from collections import defaultdict

from mafia.llm import schemas as S
from mafia.llm.client import LLMFormatError
from mafia.llm.prompts import template
from mafia.memory.retrieval import render
from mafia.reflection.reflect import wrote

WINDOW = 10  # games per compression group (DRD 7.5)


def compress(store, mcfg, names, call, me: str, game: int) -> tuple[list[tuple[str, str, dict]], int]:
    """Merge old episodic memories and old helped_or_hurt reflections (the ids in meta `hoh_ids`), one `compress`
    call per 10-game window with at least 2 memories. Returns (event specs, number of originals merged).
    A window whose call fails is skipped."""
    cutoff = game - mcfg.compression.merge_episodic_older_than_games
    hoh = set(json.loads(store.meta_get("hoh_ids", "[]")))
    pools = [("episodic", store.active(("episodic",))),
             ("reflection", [m for m in store.active(("reflection",)) if m.id in hoh])]
    out, merged, new_hoh = [], 0, []
    for type, pool in pools:
        windows = defaultdict(list)
        for m in pool:
            if m.game_created <= cutoff:
                windows[(m.game_created - 1) // WINDOW].append(m)
        for _, group in sorted(windows.items()):
            if len(group) < 2:
                continue
            try:
                res = call("compress", S.Compressed, template("compress").format(episodic_group=render(group, names)))
            except LLMFormatError as e:
                out.append(("fallback", "debug", dict(call_type="compress", reason=str(e), chosen=None)))
                continue
            i = store.add(type, res.summary, game=game, importance=max(m.importance for m in group),
                          cites=[m.id for m in group],
                          created=max(m.game_created for m in group))  # stays in its window: never re-merged
            store.set_status([m.id for m in group], "merged")
            merged += len(group)
            hoh -= {m.id for m in group}
            if type == "reflection":
                new_hoh.append(i)
            out.append(wrote(f"private:{me}", i, "add", res.summary, type))
    if new_hoh:  # merged summaries go first so deep reflection's "last 5" stay the recent unmerged ones
        keep = [i for i in json.loads(store.meta_get("hoh_ids", "[]")) if i in hoh]
        store.meta_set("hoh_ids", json.dumps(new_hoh + keep))
    return out, merged


def fade(store, mcfg, game: int) -> int:
    """Low-importance memories not retrieved, created or reinforced for a long time stop being candidates (no LLM)."""
    f = mcfg.fading
    ids = [m.id for m in store.active() if m.importance <= f.max_importance
           and max(m.last_retrieved_game or 0, m.game_updated, m.game_created) <= game - f.unretrieved_for_games]
    store.set_status(ids, "faded")
    return len(ids)
