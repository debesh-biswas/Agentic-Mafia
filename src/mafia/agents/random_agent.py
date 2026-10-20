import random

from mafia.engine.views import AgentView
from mafia.llm.schemas import LastWordsDecision, NightDecision, PlanDecision, SpeakDecision, VoteDecision
from mafia.types import Role

_SAYS = ["I am not sure yet.", "Something feels off today.", "Let us think this through.", "I have no strong read."]
_EXPR = ["neutral", "happy", "laughing", "suspicious", "defensive", "nervous", "angry", "surprised", "thinking"]


class RandomAgent:
    """Legal uniform random choices (matches tools/balance_sim.py). Never calls an LLM."""

    def __init__(self, agent_id: str, rng: random.Random):
        self.agent_id, self.rng = agent_id, rng

    def plan(self, view: AgentView) -> PlanDecision:
        return PlanDecision(plan="Play it by ear.")

    def night(self, view: AgentView) -> NightDecision:
        r, others = self.rng, [n for n in view.alive if n != view.me]
        d = NightDecision(notes=f"note {self.agent_id} d{view.day}", urgency=r.randint(0, 4))
        if view.my_role is Role.MAFIA:
            d.target = r.choice(others)  # the only other mafia would be a second mafia; roles has one
        elif view.my_role is Role.DOCTOR:
            d.protect = r.choice(others if view.doctor_cannot_self_protect else view.alive)
        return d

    def speak(self, view: AgentView) -> SpeakDecision:
        r = self.rng
        msg = None if r.random() < 0.3 else r.choice(_SAYS)
        acc = [r.choice([n for n in view.alive if n != view.me])] if msg and r.random() < 0.3 else []
        return SpeakDecision(message=msg, accuses=acc, urgency=r.randint(0, 4), expression=r.choice(_EXPR))

    def vote(self, view: AgentView) -> VoteDecision:
        return VoteDecision(vote=self.rng.choice([n for n in view.alive if n != view.me]))

    def last_words(self, view: AgentView) -> LastWordsDecision:
        return LastWordsDecision(message="Good luck everyone.")


def random_agents(rng: random.Random, ids: list[str]) -> dict[str, RandomAgent]:
    """One RandomAgent per id, each seeded from the game rng so the game is fixed by its seed."""
    return {i: RandomAgent(i, random.Random(rng.getrandbits(64))) for i in ids}
