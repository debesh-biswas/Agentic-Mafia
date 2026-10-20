import random
from typing import get_args

from mafia.config import Rules
from mafia.engine.seeds import rng_for

from mafia.engine.views import AgentView
from mafia.llm.schemas import Expression, LastWordsDecision, NightDecision, PlanDecision, SpeakDecision, VoteDecision
from mafia.types import Role

_SAYS = ["I am not sure yet.", "Something feels off today.", "Let us think this through.", "I have no strong read."]
_EXPR = get_args(Expression)


class RandomAgent:
    """Legal uniform random choices (matches tools/balance_sim.py). Never calls an LLM."""

    def __init__(self, agent_id: str, rng: random.Random, rules: Rules):
        self.agent_id, self.rng, self.urg = agent_id, rng, (rules.urgency_min, rules.urgency_max)

    def plan(self, view: AgentView) -> PlanDecision:
        return PlanDecision(plan="Play it by ear.")

    def night(self, view: AgentView) -> NightDecision:
        r, others = self.rng, [n for n in view.alive if n != view.me]
        d = NightDecision(notes=f"note {self.agent_id} d{view.day}", urgency=r.randint(*self.urg))
        if view.my_role is Role.MAFIA:
            d.target = r.choice(others)  # the only other mafia would be a second mafia; roles has one
        elif view.my_role is Role.DOCTOR:
            d.protect = r.choice(others if view.doctor_cannot_self_protect else view.alive)
        return d

    def speak(self, view: AgentView) -> SpeakDecision:
        r = self.rng
        msg = None if r.random() < 0.3 else r.choice(_SAYS)
        acc = [r.choice([n for n in view.alive if n != view.me])] if msg and r.random() < 0.3 else []
        return SpeakDecision(message=msg, accuses=acc, urgency=r.randint(*self.urg), expression=r.choice(_EXPR))

    def vote(self, view: AgentView) -> VoteDecision:
        return VoteDecision(vote=self.rng.choice([n for n in view.alive if n != view.me]))

    def last_words(self, view: AgentView) -> LastWordsDecision:
        return LastWordsDecision(message="Good luck everyone.")


def random_agents(rules: Rules, run_seed: int, game_index: int, ids: list[str]) -> dict[str, RandomAgent]:
    """One RandomAgent per id, each on its own stream of the game seed."""
    return {i: RandomAgent(i, rng_for(run_seed, game_index, f"agent:{i}"), rules) for i in ids}
