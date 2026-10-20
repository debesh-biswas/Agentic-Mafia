import random

from mafia.config import Rules, load, ROOT
from mafia.engine.scheduler import DiscussionScheduler
from mafia.llm.schemas import SpeakDecision

R = load(Rules, ROOT / "config/rules.yaml")
IDS = list("abcdefg")


def mk(ids=IDS, urg=None, seed=1, rules=R):
    return DiscussionScheduler(ids, urg or {p: 0 for p in ids}, rules, random.Random(seed))


def say(urg=0, accuses=()):
    return SpeakDecision(message="hi", accuses=list(accuses), urgency=urg)


PASS = SpeakDecision(message=None, urgency=0)


def test_budget_and_cap():
    s = mk()
    assert s.budget_remaining == 14
    n = 0
    while (nx := s.next_speaker()):
        s.record_turn(nx[0], say(4 if nx[0] == "a" else 0))
        n += 1
    assert n == 14 and s.next_speaker() is None


def test_cap_three():
    s = mk(urg={**{p: 0 for p in IDS}, "a": 4})
    seen = []
    while (nx := s.next_speaker()):
        s.record_turn(nx[0], say(4 if nx[0] == "a" else 0))
        seen.append(nx[0])
    assert seen.count("a") == 3 and len(seen) == 14
    assert max(seen.count(p) for p in IDS) == 3


def test_pass_consumes_turn():
    s = mk()
    p, _ = s.next_speaker()
    s.record_turn(p, PASS)
    assert s.budget_remaining == 13
    assert s.turns_left(p) == R.max_turns_per_player_per_day - 1


def test_priority_pass_not_consumed_when_tunable_off():
    s = mk(rules=R.model_copy(update={"priority_pass_consumes_turn": False}))
    s.record_turn("a", say(0, ["b"]))
    assert s.next_speaker() == ("b", True)
    s.record_turn("b", PASS, is_priority=True)
    assert s.budget_remaining == 13 and s.turns_left("b") == 3
    p, _ = s.next_speaker()
    s.record_turn(p, PASS)  # ordinary pass still consumes
    assert s.budget_remaining == 12


def test_priority_first_and_pair_once():
    s = mk()
    s.record_turn("a", say(0, ["b"]))
    assert s.next_speaker() == ("b", True)
    s.record_turn("b", say(0), is_priority=True)
    s.record_turn("a", say(0, ["b"]))  # (a,b) already granted today
    assert s.next_speaker()[1] is False


def test_counter_accusation_gets_no_priority():
    s = mk()
    s.record_turn("a", say(0, ["b"]))  # a accuses b: b gets priority
    assert s.next_speaker() == ("b", True)
    s.record_turn("b", say(0, ["a"]), is_priority=True)  # b accuses a back: no priority for a
    assert not s.priority_queue
    assert s.next_speaker()[1] is False


def test_counter_accusation_no_extra_priority_when_queued():
    s = mk()
    s.record_turn("a", say(0, ["b"]))
    s.record_turn("c", say(0, ["b"]))  # b already queued
    assert s.next_speaker() == ("b", True)
    assert s.next_speaker()[1] is False


def test_priority_skips_ineligible():
    s = mk()
    for _ in range(3):
        s.record_turn("b", say(0))
    s.record_turn("a", say(0, ["b"]))
    nx = s.next_speaker()
    assert nx[0] != "b" and nx[1] is False


def test_consecutive_passes_end():
    s = mk()
    while (nx := s.next_speaker()):
        s.record_turn(nx[0], PASS)
    assert 0 < s.budget_remaining and s.consecutive_passes >= 6


def test_speech_resets_pass_count():
    s = mk()
    for _ in range(6):
        s.record_turn(s.next_speaker()[0], PASS)
    s.record_turn("a", say())
    assert s.next_speaker() is not None


def test_urgency_picks_highest_and_seeded_ties():
    s = mk(urg={**{p: 1 for p in IDS}, "d": 3})
    assert s.next_speaker() == ("d", False)
    picks = lambda seed: [mk(seed=seed).next_speaker()[0] for _ in range(1)]
    assert picks(5) == picks(5)
    assert {mk(seed=i).next_speaker()[0] for i in range(40)} > {"a"}


def test_fewer_alive():
    s = mk(ids=list("abcde"))
    assert s.budget_remaining == 10
