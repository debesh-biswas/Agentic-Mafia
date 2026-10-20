import pytest
from mafia.llm.schemas import (Compressed, DeepReflect, Insight, LastWordsDecision, NightDecision, PlanDecision,
                               Questions, ReflectInsights, SpeakDecision, Summaries, VoteDecision)
from pydantic import ValidationError

INS = {"text": "t", "kind": "lesson", "importance": 5}


def rt(m):
    assert type(m).model_validate_json(m.model_dump_json()) == m


def test_round_trip():
    ins = Insight(**INS, cites=[1], action="update", target_id=3)
    for m in [PlanDecision(plan="p"), NightDecision(notes="n", urgency=2, target="Bo"),
              SpeakDecision(message=None, urgency=0), SpeakDecision(message="hi", accuses=["Al"], urgency=4, expression="angry"),
              VoteDecision(vote="Al"), LastWordsDecision(message="bye"), Questions(questions=["q"]), ins,
              ReflectInsights(insights=[ins], keep_ids=[2], helped_or_hurt="h", game_summary="s", summary_importance=3),
              Summaries(friends={"a": "b"}, self="me"), DeepReflect(insights=[ins]),
              Compressed(summary="s", importance=1)]:
        rt(m)


def test_memory_ids():
    i = Insight(**INS, cites=["m42", 7, "8"], target_id="m5")
    assert i.cites == [42, 7, 8] and i.target_id == 5
    r = ReflectInsights(insights=[], keep_ids=["m1", 2], helped_or_hurt="", game_summary="", summary_importance=1)
    assert r.keep_ids == [1, 2]
    with pytest.raises(ValidationError):
        Insight(**INS, cites=["x42"])


@pytest.mark.parametrize("u", [-1, 5])
def test_urgency_bounds(u):
    with pytest.raises(ValidationError):
        SpeakDecision(message="x", urgency=u)
    with pytest.raises(ValidationError):
        NightDecision(notes="", urgency=u)


def test_limits():
    with pytest.raises(ValidationError):
        Questions(questions=[])
    with pytest.raises(ValidationError):
        Questions(questions=list("abcd"))
    with pytest.raises(ValidationError):
        DeepReflect(insights=[Insight(**INS)] * 4)
    with pytest.raises(ValidationError):
        SpeakDecision(message="x", urgency=1, expression="bored")
