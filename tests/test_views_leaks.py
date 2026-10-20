import json
import re

from mafia.agents.random_agent import random_agents
from mafia.config import ROOT, Personas, Rules, load
from mafia.engine.game import Game
from mafia.types import Phase, Role

R = load(Rules, ROOT / "config/rules.yaml")
P = load(Personas, ROOT / "config/personas.yaml").personas
IDS = [p.agent_id for p in sorted(P, key=lambda p: p.seat)]
ROLE_WORDS = re.compile(r"mafia|doctor|villager", re.I)
OWN_ONLY = {"my_role", "doctor_last_protect", "doctor_cannot_self_protect"}


def spied_game(seed):
    g = Game(R, P, {}, "t", 0, seed, "test", 0, 0)
    g.agents = random_agents(g.rng, IDS)
    seen = []  # (agent_id, view)
    for a in g.agents.values():
        for m in ("plan", "night", "speak", "vote", "last_words"):
            orig = getattr(a, m)
            setattr(a, m, lambda v, a=a, orig=orig: (seen.append((a.agent_id, v)), orig(v))[1])
    return g, seen


def test_views_leak_nothing_over_200_games():
    for seed in range(200):
        g, seen = spied_game(seed)
        g.play()
        roles = {p.agent_id: p.role for p in g.st.players.values()}
        night_acts = [e for e in g.events if e.type == "night_action"]
        assert seen and night_acts
        for pid, v in seen:
            assert v.my_role is roles[pid] and v.me == g.name(pid)
            doc = roles[pid] is Role.DOCTOR
            assert (v.doctor_cannot_self_protect is not None) == doc
            assert doc or v.doctor_last_protect is None
            text = json.dumps(v.model_dump(mode="json", exclude=OWN_ONLY | {"my_notes"}))
            assert not ROLE_WORDS.search(text), (seed, pid, text)  # no role word anywhere in public data
            assert not re.search(r"\b(?:urgency|expression)\b", text)  # others' urgency/expression stripped
            assert not re.search(r"protect|target|saved", text, re.I)  # night actions never shown
            for other in IDS:  # other agents' private notes never appear
                if other != pid:
                    assert f"note {other} d" not in text
            assert v.my_notes == "" or v.my_notes.startswith(f"note {pid} d")
            if v.today_transcript:
                assert all(set(t.model_dump()) == {"speaker", "message", "kind"} for t in v.today_transcript)
            if v.phase not in (Phase.DISCUSSION, Phase.VOTE):
                assert v.today_transcript is None
            assert (v.votes_so_far_today is not None) == (v.phase is Phase.VOTE)
            # alive and dead names partition the table, and nobody's role is revealed before game_end
            assert sorted(v.alive + v.dead) == sorted(p.name for p in P)


def test_view_is_frozen_and_copies_facts():
    g, seen = spied_game(1)
    g.play()
    v = seen[0][1]
    try:
        v.me = "x"
        raise AssertionError("view must be frozen")
    except Exception as e:
        assert not isinstance(e, AssertionError)
    assert v.public_facts is not g.st.public_facts
