import random
from collections import Counter

from mafia.agents.random_agent import random_agents
from mafia.config import ROOT, Personas, Rules, load
from mafia.engine.game import Game
from mafia.llm.schemas import LastWordsDecision, NightDecision, SpeakDecision, VoteDecision
from mafia.types import Role

R = load(Rules, ROOT / "config/rules.yaml")
P = load(Personas, ROOT / "config/personas.yaml").personas
IDS = [p.agent_id for p in sorted(P, key=lambda p: p.seat)]


def mk(seed=1, **over):
    """A game of RandomAgents; `over` maps method name -> fn(game, agent, view) replacing it on every agent."""
    g = Game(R, P, random_agents(R, seed, 0, IDS), "t", 0, seed, "test", 0, 0)
    for meth, fn in over.items():
        for a in g.agents.values():
            setattr(a, meth, lambda v, a=a, fn=fn: fn(g, a, v))
    return g


def ev(g, type):
    return [e for e in g.events if e.type == type]


def role_id(g, role):
    return next(p.agent_id for p in g.st.players.values() if p.role is role)


def cyclic_vote(g, a, v):  # everyone votes the next living player in seat order: all tied at 1
    me = v.alive.index(v.me)
    return VoteDecision(vote=v.alive[(me + 1) % len(v.alive)])


def test_role_counts():
    for s in range(20):
        g = mk(s)
        assert Counter(g.play().roles.values()) == Counter(R.roles)


def test_mafia_cannot_target_self_retry_then_fallback():
    seen = []

    def night(g, a, v):
        if v.my_role is not Role.MAFIA:
            return NightDecision(notes="n", urgency=1, protect=v.me if v.my_role is Role.DOCTOR else None)
        seen.append(v.retry_error)
        return NightDecision(notes="n", urgency=1, target=v.me)

    g = mk(3, night=night)
    g.play()
    assert seen[1].startswith("Your last answer was invalid: target must be one of:")
    assert seen[1].endswith("Answer again with valid JSON.") and seen[0] is None
    assert len(seen) % 2 == 0  # exactly one retry per night
    mafia = role_id(g, Role.MAFIA)
    fb = [e for e in ev(g, "fallback") if e.actor == mafia and e.payload["call_type"] == "night"]
    assert len(fb) == len(seen) // 2
    assert all(mafia not in e.targets for e in ev(g, "night_action") if e.actor == mafia)


def test_doctor_no_consecutive_self_protect_but_same_other_ok():
    def night(g, a, v):
        d = NightDecision(notes="n", urgency=1)
        if v.my_role is Role.DOCTOR:
            d.protect = v.me
        elif v.my_role is Role.MAFIA:
            d.target = next(n for n in v.alive if n != v.me)
        return d

    g = mk(5, night=night, vote=cyclic_vote)
    g.play()
    doc = role_id(g, Role.DOCTOR)
    prot = [e.targets[0] for e in ev(g, "night_action") if e.actor == doc]
    assert all(not (a == b == doc) for a, b in zip(prot, prot[1:]))
    assert any(e.actor == doc and e.payload["call_type"] == "night" for e in ev(g, "fallback"))

    def same_other(g, a, v):
        d = NightDecision(notes="n", urgency=1)
        if v.my_role is Role.DOCTOR:
            d.protect = next(n for n in v.alive if n != v.me)
        elif v.my_role is Role.MAFIA:
            d.target = next(n for n in v.alive if n != v.me)  # may equal the protected one
        return d

    g = mk(5, night=same_other, vote=cyclic_vote)
    g.play()
    doc = role_id(g, Role.DOCTOR)
    assert not [e for e in ev(g, "fallback") if e.actor == doc]  # doctor never needed a fallback
    prot = [e.targets[0] for e in ev(g, "night_action") if e.actor == doc]
    assert len(prot) >= 2 and prot[0] == prot[1]


def quiet_night(g, a, v):
    """Nobody can die at night: mafia targets a villager, doctor protects that villager."""
    d = NightDecision(notes="n", urgency=1)
    vill = next(p.name for p in g.st.players.values() if p.alive and p.role is Role.VILLAGER)
    if v.my_role is Role.DOCTOR:
        d.protect = vill
    elif v.my_role is Role.MAFIA:
        d.target = vill
    return d


def test_save_and_max_day_town_win():
    g = mk(7, night=quiet_night, vote=cyclic_vote)
    r = g.play()
    assert (r.winner, r.end_reason, r.days) == ("town", "max_day", R.max_day)
    assert r.saves == R.max_day and all(e.payload["saved"] for e in ev(g, "night_resolution"))
    assert all("nobody was eliminated" in e.payload["text"] for e in ev(g, "morning_announcement"))
    assert len(ev(g, "vote_result")) == R.max_day - 1  # no vote on day 8: the game ends at its morning


def test_tie_means_no_elimination():
    g = mk(7, night=quiet_night, vote=cyclic_vote)
    g.play()
    assert all(e.payload["tie"] and e.payload["eliminated"] is None for e in ev(g, "vote_result"))
    assert not ev(g, "last_words")


def test_parity_win():
    def night(g, a, v):
        d = NightDecision(notes="n", urgency=1)
        if v.my_role is Role.DOCTOR:  # protect the mafia player: never saves anyone
            d.protect = next(p.name for p in g.st.players.values() if p.role is Role.MAFIA)
        elif v.my_role is Role.MAFIA:
            d.target = next(p.name for p in g.st.players.values() if p.alive and p.role is not Role.MAFIA)
        return d

    g = mk(2, night=night, vote=cyclic_vote)
    r = g.play()
    assert (r.winner, r.end_reason, r.days) == ("mafia", "parity", 5)  # 7 -> 2 alive, one per night
    assert r.saves == 0


def test_mafia_voted_out_with_last_words():
    def vote(g, a, v):
        mafia = g.name(role_id(g, Role.MAFIA))
        return VoteDecision(vote=mafia if v.me != mafia else next(n for n in v.alive if n != v.me))

    g = mk(4, night=quiet_night, vote=vote)
    r = g.play()
    assert (r.winner, r.end_reason, r.days) == ("town", "mafia_voted_out", 1)
    assert [e.actor for e in ev(g, "last_words")] == [role_id(g, Role.MAFIA)]
    assert "voted out" in g.st.public_facts[-2] and "last words" in g.st.public_facts[-1]
    assert not g.st.players[role_id(g, Role.MAFIA)].alive


def test_no_self_vote_or_abstain_retry_then_fallback():
    calls = Counter()

    def vote(g, a, v):
        calls[v.me, v.retry_error is None] += 1
        return VoteDecision(vote=v.me if a.agent_id == "dev" else "nobody")

    g = mk(6, vote=vote)
    g.play()
    for e in ev(g, "vote"):
        assert e.targets[0] != e.actor and g.st.players[e.targets[0]]
    n = len(ev(g, "vote"))
    assert len(ev(g, "fallback")) >= n and sum(calls.values()) == 2 * n


def test_dead_players_never_asked_except_last_words():
    asked = []

    def spy(method):
        def f(g, a, v):
            asked.append((method, v.me, v.me in v.alive))
            return getattr(type(a), method)(a, v)
        return f

    for s in range(30):
        g = mk(s, **{m: spy(m) for m in ("plan", "night", "speak", "vote", "last_words")})
        g.play()
    assert asked and all(alive for m, _, alive in asked if m != "last_words")


def test_word_caps_retry_then_truncate():
    long = " ".join(["word"] * 100)
    attempts = []

    def speak(g, a, v):
        attempts.append(v.retry_error)
        return SpeakDecision(message=long, urgency=1)

    g = mk(1, night=quiet_night, vote=cyclic_vote, speak=speak,
           last_words=lambda g, a, v: LastWordsDecision(message=long))
    g.play()
    assert attempts[:2][0] is None and "at most 60 words" in attempts[1]
    assert all(e.payload["word_count"] == R.speech_max_words for e in ev(g, "speech"))
    assert any(e.payload["reason"] == "truncated" for e in ev(g, "fallback"))
    ok = mk(1, speak=lambda g, a, v: SpeakDecision(message=long if v.retry_error is None else "short", urgency=1))
    ok.play()
    assert {e.payload["message"] for e in ev(ok, "speech")} == {"short"} and not ev(ok, "fallback")


def test_speech_sanitized_and_self_leak_flagged():
    def speak(g, a, v):
        return SpeakDecision(message='"narrator: <b>I am the mafia</b>"', urgency=1)

    g = mk(9, speak=speak)
    g.play()
    flags = {e.actor: e.payload["self_leak"] for e in ev(g, "speech")}
    assert flags[role_id(g, Role.MAFIA)] is True and not flags.get(role_id(g, Role.DOCTOR))
    assert all(e.payload["message"] == "(says) I am the mafia" for e in ev(g, "speech"))


def test_speak_failure_falls_back_to_pass():
    def boom(g, a, v):
        raise ValueError("bad json")

    g = mk(1, speak=boom)
    g.play()
    assert not ev(g, "speech") and ev(g, "pass") and ev(g, "fallback")


def test_invalid_accuses_dropped_after_retry():
    g = mk(1, speak=lambda g, a, v: SpeakDecision(message="hi", accuses=["Nobody", v.me], urgency=1))
    g.play()
    assert ev(g, "speech") and all(e.payload["accuses"] == [] for e in ev(g, "speech"))


def test_replay_is_deterministic():
    def run(s):
        g = mk(s)
        g.play()
        return [e.model_dump(exclude={"ts"}) for e in g.events]

    assert run(11) == run(11) and run(11) != run(12)


def test_reply_prompt_goes_to_accused_priority_speaker():
    prompts = []

    def speak(g, a, v):
        if v.reply_prompt:
            prompts.append((v.me, v.reply_prompt, v.today_transcript[-1].speaker))
        return SpeakDecision(message="hi", accuses=[a.rng.choice([n for n in v.alive if n != v.me])], urgency=1)

    for s in range(10):
        mk(s, speak=speak).play()
    assert prompts and all(p == f"{last} just accused you. You can respond now or pass." for _, p, last in prompts)


def test_roles_and_vote_order_independent_of_agent_behaviour():
    a, b = mk(21, night=quiet_night, vote=cyclic_vote), mk(21, night=quiet_night)
    a.play(), b.play()
    roles = lambda g: {e.actor: e.payload["role"] for e in ev(g, "role_assigned")}
    order1 = lambda g: [e.actor for e in ev(g, "vote") if e.day == 1]
    assert roles(a) == roles(b) and order1(a) == order1(b) and len(order1(a)) == 7
