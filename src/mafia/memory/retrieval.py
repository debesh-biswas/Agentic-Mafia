"""Memory retrieval (DRD 7.2) and lesson bookkeeping (DRD 7.3)."""
from __future__ import annotations

import json

import numpy as np

from mafia.memory.store import Memory, MemoryStore

MIN_APPLIED, FAIL_RATE = 3, 0.4


def _norm(x: np.ndarray) -> np.ndarray:
    """Min-max to [0,1]; if all values are equal the component carries no signal, so it is 0 for everyone."""
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x)


def retrieve(store: MemoryStore, query: str, game: int, cfg, *, k=None, types=None) -> list[Memory]:
    k = k or cfg.top_k
    cands = store.active(types)
    if not cands:
        return []
    q = np.asarray(store.embedder.embed(query), dtype=np.float32)
    rec = np.array([cfg.recency_decay_per_game ** (game - (m.last_retrieved_game if m.last_retrieved_game is not None
                                                          else m.game_updated)) for m in cands])
    imp = np.array([m.importance / 10 for m in cands])
    rel = np.array([float(m.embedding @ q) for m in cands])
    w = cfg.weights
    score = w["recency"] * _norm(rec) + w["importance"] * _norm(imp) + w["relevance"] * _norm(rel)
    order = sorted(range(len(cands)), key=lambda i: (-score[i], cands[i].id))
    chosen = order[:k]
    failed = sorted((i for i, m in enumerate(cands) if m.type == "lesson" and m.games_applied >= MIN_APPLIED
                     and m.wins_when_applied / m.games_applied < FAIL_RATE), key=lambda i: (-rel[i], cands[i].id))
    failed = failed[:cfg.include_failed_lessons]
    for i in failed:
        if i not in chosen:  # replaces the lowest-scored item that isn't itself a reserved failed lesson
            drop = [j for j in reversed(chosen) if j not in failed]
            if not drop:
                break
            chosen.remove(drop[0])
            chosen.append(i)
    chosen.sort(key=lambda i: (-score[i], cands[i].id))
    out = [cands[i] for i in chosen]
    store.touch([m.id for m in out], game)
    return out


def render(memories: list[Memory], names: dict[str, str]) -> str:
    lines = []
    for m in memories:
        if m.type == "belief":
            who = ", ".join(names.get(a, a) for a in m.about) or "?"
            tag = f"belief about {who}" + (f", {m.confidence}" if m.confidence else "") + f", game {m.game_updated}"
        elif m.type == "lesson":
            tag = "lesson" + (f", as {m.role_context}" if m.role_context else "")
            if m.games_applied:
                tag += f", won {m.wins_when_applied} of {m.games_applied} times used"
        elif m.type == "reflection":
            tag = f"reflection, game {m.game_updated}"
        else:
            tag = f"game {m.game_updated} summary"
        lines.append(f"[m{m.id}] ({tag}) {m.text}")
    return "\n".join(lines)


def mark_applied(store: MemoryStore, lesson_ids) -> None:
    store.meta_set("applied_lessons", json.dumps(sorted(set(lesson_ids))))


def settle_lessons(store: MemoryStore, won: bool) -> None:
    ids = json.loads(store.meta_get("applied_lessons", "[]"))
    store.bump_lessons(ids, won)
    store.meta_set("applied_lessons", "[]")
