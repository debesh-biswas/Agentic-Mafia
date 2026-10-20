import threading

import numpy as np
import pytest

from mafia.config import Retrieval
from mafia.memory.embed import FakeEmbedder
from mafia.memory.retrieval import mark_applied, render, retrieve, settle_lessons
from mafia.memory.store import MemoryStore

CFG = Retrieval(top_k=2, weights={"recency": 1.0, "importance": 1.0, "relevance": 1.0},
                recency_decay_per_game=0.5, include_failed_lessons=1)


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path / "m.sqlite", FakeEmbedder())


def add(store, text, imp=5, game=1, type="belief", **kw):
    return store.add(type, text, game=game, importance=imp, **kw)


def test_insert_get_roundtrip(store):
    i = add(store, "Dev is quiet", about=["dev"], confidence="hunch", cites=[3])
    m = store.get(i)
    assert (m.text, m.about, m.cites, m.status, m.game_created) == ("Dev is quiet", ["dev"], [3], "active", 1)
    assert m.embedding.shape == (384,) and m.embedding.dtype == np.float32


def test_ranking_matches_hand_computation(store):
    # query "alpha". game=3, decay 0.5, k=2.
    a = add(store, "alpha", imp=2, game=3)            # rec .5^0=1,   imp .2, rel 1
    b = add(store, "beta", imp=10, game=3)            # rec 1,        imp 1,  rel 0
    c = add(store, "gamma", imp=6, game=1)            # rec .25,      imp .6, rel 0
    # (hash collisions between words would break rel=0; assert the premise)
    assert store.get(a).embedding @ store.get(b).embedding == 0 == store.get(a).embedding @ store.get(c).embedding
    # normalized: rec: a=1,b=1,c=0 ; imp: a=0,b=1,c=(.6-.2)/.8=.5 ; rel: a=1,b=0,c=0
    # scores: a=2.0, b=2.0, c=0.5  -> tie a/b broken by id -> [a, b]
    assert [m.id for m in retrieve(store, "alpha", 3, CFG)] == [a, b]
    # with k=3 the third is c (score 0.5)
    assert [m.id for m in retrieve(store, "alpha", 3, CFG, k=3)] == [a, b, c]


def test_min_max_all_equal_is_zero(store):
    add(store, "x", imp=5), add(store, "y", imp=5)
    assert len(retrieve(store, "zzz", 1, CFG)) == 2  # no division by zero


def test_supersede_hides_old_and_links(store):
    old = add(store, "Dev is town", about=["dev"], confidence="hunch")
    new = store.supersede(old, "Dev is mafia", game=2, confidence="fairly sure")
    assert store.get(old).status == "superseded" and store.get(old).superseded_by == new
    assert store.get(new).about == ["dev"]
    assert [m.id for m in retrieve(store, "Dev", 2, CFG, k=5)] == [new]


def test_update_writes_version(store):
    i = add(store, "v1", confidence="hunch")
    store.update(i, "v2", "certain", 7, 4)
    m = store.get(i)
    assert (m.text, m.confidence, m.importance, m.game_updated) == ("v2", "certain", 7, 4)
    assert store.versions(i) == [(4, "v1", "hunch")]


def _lesson(store, text, applied, wins):
    i = add(store, text, type="lesson", role_context="mafia")
    for n in range(applied):
        store.bump_lessons([i], n < wins)
    return i


def test_failed_lesson_slot_present(store):
    top1 = add(store, "alpha", imp=10)
    top2 = add(store, "alpha two", imp=9)
    bad = _lesson(store, "unrelated words", applied=5, wins=1)   # 0.2 < 0.4
    got = [m.id for m in retrieve(store, "alpha", 5, CFG, k=2)]
    assert bad in got and len(got) == 2 and top1 in got and top2 not in got


def test_failed_lesson_slot_absent(store):
    add(store, "alpha", imp=10), add(store, "alpha two", imp=9)
    few = _lesson(store, "unrelated", applied=2, wins=0)    # too few uses
    ok = _lesson(store, "unrelated too", applied=5, wins=3)  # fine rate
    got = [m.id for m in retrieve(store, "alpha", 5, CFG, k=2)]
    assert few not in got and ok not in got


def test_retrieval_bookkeeping(store):
    i = add(store, "alpha")
    retrieve(store, "alpha", 7, CFG)
    retrieve(store, "alpha", 9, CFG)
    m = store.get(i)
    assert (m.last_retrieved_game, m.times_retrieved) == (9, 2)


def test_lesson_bookkeeping_and_render(store):
    i = add(store, "Don't accuse first", type="lesson", role_context="mafia")
    mark_applied(store, [i])
    settle_lessons(store, True)
    settle_lessons(store, False)  # cleared: no double count
    m = store.get(i)
    assert (m.games_applied, m.wins_when_applied) == (1, 1)
    names = {"dev": "Dev"}
    b = add(store, "goes quiet", about=["dev"], confidence="fairly sure", game=17)
    r = add(store, "insight", type="reflection", game=3)
    e = add(store, "I was doctor", type="episodic", game=31)
    assert render([store.get(x) for x in (b, i, r, e)], names).splitlines() == [
        f"[m{b}] (belief about Dev, fairly sure, game 17) goes quiet",
        f"[m{i}] (lesson, as mafia, won 1 of 1 times used) Don't accuse first",
        f"[m{r}] (reflection, game 3) insight",
        f"[m{e}] (game 31 summary) I was doctor",
    ]


def test_summaries_latest_per_subject_and_seed(store):
    store.seed({"dev": "friendly"}, "neutral")
    assert store.get_summaries() == {"dev": "friendly", "self": "neutral"}
    store.set_summaries(5, {"dev": "shifty"})
    assert store.get_summaries() == {"dev": "shifty", "self": "neutral"}
    assert store.meta_get("nope") is None
    store.meta_set("k", "v")
    assert store.meta_get("k") == "v"


def test_concurrent_access(store):
    errs = []

    def work(n):
        try:
            for j in range(30):
                add(store, f"t{n} item {j}")
                retrieve(store, "item", 1, CFG)
        except Exception as e:  # noqa: BLE001
            errs.append(e)

    ts = [threading.Thread(target=work, args=(n,)) for n in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and len(store.active()) == 60


def test_supersede_atomic_on_failure(store, monkeypatch):
    old = add(store, "Dev is town")
    n = len(store.active())
    real = store._db

    class Boom:
        def __getattr__(self, a):
            return getattr(real, a)

        def execute(self, sql, *args):
            if sql.startswith("UPDATE memories SET status='superseded'"):
                raise RuntimeError("boom")
            return real.execute(sql, *args)

    monkeypatch.setattr(store, "_db", Boom())
    with pytest.raises(RuntimeError):
        store.supersede(old, "Dev is mafia", game=2)
    monkeypatch.undo()
    assert store.get(old).status == "active" and len(store.active()) == n


def test_close(store):
    store.close()
