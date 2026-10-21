"""F16: near-duplicate reinforcement, reflection compression, in-game shown ids, split fallback counts."""
import csv
import json

from mafia.config import ROOT, load_run
from mafia.memory.embed import FakeEmbedder
from mafia.memory.maintenance import compress
from mafia.memory.store import MemoryStore
from mafia.metrics import compute, report as rep
from mafia.reflection.deep import deep_reflect
from mafia.reflection.reflect import reflect
from mafia.renderers.dashboard import write_dashboard
from mafia.runner.season import run_season
from test_reflection import MC, NAMES, caller, ins, insights, outcome, store
from test_season import IDS, V0, ev, log

BASE = "alpha beta gamma delta epsilon zeta eta theta iota kappa"  # + one word: cosine 10/sqrt(10*11) = 0.953
NEAR, FAR = BASE + " lambda", "totally unrelated words appear here instead"


def run(st, *insight, game=3, **kw):
    reflect(st, MC, NAMES, caller([{"questions": ["q?"]}, insights(*insight)]), outcome(game=game), **kw)


def test_near_duplicate_reinforces_instead_of_adding(tmp_path):
    st = store(tmp_path)
    a = st.add("belief", BASE, about=["sana"], game=1, importance=5, cites=[])
    run(st, ins(NEAR, importance=8, cites=[a]))
    assert len([m for m in st.active(("belief",))]) == 1
    m = st.get(a)
    assert (m.text, m.importance, m.game_updated, m.cites) == (BASE, 8, 3, [a])
    run(st, ins(NEAR, importance=2, cites=[a]), game=4)  # never lowers importance; cites not repeated
    assert (st.get(a).importance, st.get(a).cites, st.get(a).game_updated) == (8, [a], 4)


def test_below_threshold_adds_and_different_subject_or_type_do_not_merge(tmp_path):
    st = store(tmp_path)
    st.add("belief", BASE, about=["sana"], game=1, importance=5)
    run(st, ins(FAR), ins(NEAR, about=["Mira"]), ins(NEAR, kind="lesson", about=[]))
    assert len(st.active(("belief",))) == 3 and len(st.active(("lesson",))) == 1


def test_reinforce_event_and_supersede_dedupes(tmp_path):
    st = store(tmp_path)
    a = st.add("belief", BASE, about=["sana"], game=1, importance=5)
    b = st.add("belief", "other thing entirely", about=["sana"], game=1, importance=5)
    out = reflect(st, MC, NAMES, caller([{"questions": ["q?"]}, insights(ins(NEAR, action="supersede", target_id=b))]),
                  outcome(game=3))
    assert (st.get(b).status, st.get(b).superseded_by) == ("superseded", a)  # duplicate of a: reinforce a, retire b
    assert ("memory_supersede", "private:dev", dict(old_id=b, new_id=a)) in out
    assert [(p["memory_id"], p["action"]) for t, _, p in out if t == "memory_write"][0] == (a, "reinforce")
    assert not [m for m in st.active() if m.text == NEAR]


def test_hoh_compression_merges_window_and_keeps_hoh_ids_consistent(tmp_path):
    s = store(tmp_path)
    h = [s.add("reflection", f"hoh {g}", game=g, importance=6) for g in (1, 2, 3, 25)]
    other = s.add("reflection", "deep one", game=2, importance=7)  # not a helped_or_hurt: untouched
    s.meta_set("hoh_ids", json.dumps(h))
    out, merged = compress(s, MC, NAMES, caller([{"summary": "Merged hoh.", "importance": 1}]), "dev", 30)
    new = [m for m in s.active(("reflection",)) if m.text == "Merged hoh."]
    assert merged == 3 and new[0].cites == h[:3] and new[0].game_created == 3 and new[0].importance == 6
    assert all(s.get(i).status == "merged" for i in h[:3]) and s.get(other).status == "active"
    assert json.loads(s.meta_get("hoh_ids")) == [new[0].id, h[3]]
    assert out[0][2]["type"] == "reflection"
    # deep reflection shows the last 5 unmerged hoh, i.e. the recent ones, not the summary when 5 recent exist
    more = [s.add("reflection", f"recent {g}", game=g, importance=6) for g in range(26, 31)]
    s.meta_set("hoh_ids", json.dumps(json.loads(s.meta_get("hoh_ids")) + more))
    call = caller([{"insights": []}])
    s.add("episodic", "e", game=30, importance=5)
    deep_reflect(s, NAMES, call, "dev", 30)
    assert all(f"[m{i}]" in call.users[0] for i in more) and f"[m{new[0].id}]" not in call.users[0]


def test_update_of_memory_retrieved_in_game_is_applied(tmp_path):
    q = "completely different topic"
    for ingame in (False, True):
        st = MemoryStore(tmp_path / f"{ingame}.sqlite", FakeEmbedder())
        a = st.add("belief", "Zed never lies", about=["sana"], game=1, importance=5)
        for k in range(20):  # crowd the question's retrieval so `a` is not shown by reflection itself
            st.add("belief", f"{q} {k}", about=["mira"], game=1, importance=5)
        call = caller([{"questions": [q]}, insights(ins("Zed lies sometimes", action="update", target_id=a))])
        out = reflect(st, MC, NAMES, call, outcome(), ingame_ids=[a] if ingame else ())
        assert (st.get(a).text == "Zed lies sometimes") == ingame
        assert bool([1 for t, _, p in out if t == "fallback"]) == (not ingame)


def events(fallbacks):
    base = log("town", [ev(("fallback"), a, call_type=c, reason="r", chosen=None) for a, c in fallbacks])
    end = next(e for e in base if e.type == "game_end")
    end.payload = dict(winner="town", end_reason="mafia_voted_out", days=2)
    return base


def test_csv_splits_in_game_fallbacks_from_reflection_issues():
    fb = [("a", "vote"), ("a", "plan"), ("b", "speak"), ("a", "reflect"), ("a", "deep_reflect"), ("c", "compress"),
          ("c", "summaries")]
    g, agents = compute.rows(events(fb), V0, "r", "memory", 1, 0, 1.0)
    assert (g["fallbacks"], g["reflection_issues"]) == (3, 4)
    by = {r["agent"]: (r["fallbacks"], r["reflection_issues"]) for r in agents}
    assert by["a"] == (2, 2) and by["b"] == (1, 0) and by["c"] == (0, 2)
    assert compute.GAMES_COLS[-1] == compute.AGENTS_COLS[-1] == "reflection_issues"


def test_old_csv_without_column_still_loads(tmp_path):
    run_season(load_run(ROOT / "config/run_local_memory.yaml").model_copy(update={"run_id": "old", "games": 2}),
               mock=True, runs_dir=tmp_path)
    d = tmp_path / "old"
    for n in ("games.csv", "agents.csv"):
        rows = list(csv.DictReader((d / n).open(newline="")))
        cols = [c for c in rows[0] if c != "reflection_issues"]
        with (d / n).open("w", newline="") as f:
            w = csv.DictWriter(f, cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    assert all(r["reflection_issues"] == 0 for r in rep.load(d)["G"])
    assert "Every 5 games" in write_dashboard(d).read_text()
