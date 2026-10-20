import pytest
from mafia.agents.random_agent import RandomAgent, random_agents
from mafia.config import ROOT, Personas, Rules, load
from mafia.engine.game import Game
from mafia.llm.client import LLMError
from mafia.llm.recorder import MockLLM
from mafia.llm.schemas import PlanDecision

R = load(Rules, ROOT / "config/rules.yaml")
P = load(Personas, ROOT / "config/personas.yaml").personas
IDS = [p.agent_id for p in sorted(P, key=lambda p: p.seat)]


class LLMPlanAgent(RandomAgent):
    def __init__(self, inner, llm):
        self.__dict__.update(inner.__dict__)
        self.llm = llm

    def plan(self, view):
        return self.llm.call("plan", "sys", "usr", PlanDecision, self.agent_id)[0]


def _game(llm, concurrent=False):
    agents = random_agents(R, 3, 0, IDS)
    agents[IDS[0]] = LLMPlanAgent(agents[IDS[0]], llm)
    return Game(R, P, agents, "t", 0, 3, "test", 0, 0, concurrent=concurrent)


def test_format_error_goes_straight_to_fallback():
    llm = MockLLM(["garbage", "still garbage"])
    g = _game(llm)
    assert g.play().winner
    assert len(llm.sent) == 2
    fb = [e for e in g.events if e.type == "fallback" and e.payload["call_type"] == "plan"]
    assert len(fb) == 1 and "LLMFormatError" in fb[0].payload["reason"]


def test_plain_llm_error_propagates():
    class Boom:
        def call(self, *a):
            raise LLMError("net down", {})
    with pytest.raises(LLMError):
        _game(Boom()).play()
