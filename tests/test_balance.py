from mafia.agents.random_agent import random_agents
from mafia.config import ROOT, Personas, Rules, load
from mafia.engine.game import Game

R = load(Rules, ROOT / "config/rules.yaml")
P = load(Personas, ROOT / "config/personas.yaml").personas
IDS = [p.agent_id for p in sorted(P, key=lambda p: p.seat)]


def test_random_play_town_win_rate():
    """20k games: binomial SE = 0.0035, so 4 SE = 0.014 around the 0.393 reference."""
    n, town = 20_000, 0
    for i in range(n):
        g = Game(R, P, {}, "t", i, 7, "test", 0, 0, record=False)
        g.agents = random_agents(g.rng, IDS)
        town += g.play().winner == "town"
    assert abs(town / n - 0.393) < 0.014
