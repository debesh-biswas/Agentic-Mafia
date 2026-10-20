from mafia.agents.random_agent import random_agents
from mafia.config import ROOT, Personas, Rules, load
from mafia.engine.game import Game
from mafia.logs.events import EventLog
from mafia.renderers.text import TextRenderer, visible

RULES = load(Rules, ROOT / "config/rules.yaml")
PERSONAS = load(Personas, ROOT / "config/personas.yaml").personas
IDS = [p.agent_id for p in sorted(PERSONAS, key=lambda p: p.seat)]


def play(i=0):
    g = Game(RULES, PERSONAS, random_agents(RULES, 1, i, IDS), "t", i, 1, "sim", 0, 0)
    g.play()
    return g.events


def test_roundtrip(tmp_path):
    ev = play(2)
    log = EventLog(tmp_path, 2)
    log.write(ev)
    assert log.path.name == "0003.jsonl"
    assert log.read() == ev


def test_views():
    ev = play()
    assert not [e for e in visible(ev, "public") if e.visibility != "public"]
    me = IDS[0]
    got = {e.visibility for e in visible(ev, me)}
    assert got <= {"public", f"private:{me}"} and f"private:{me}" in got
    truth = visible(ev, "truth")
    assert all(e.visibility != "debug" for e in truth) and any(e.visibility == "truth" for e in truth)


def test_render_uses_names(capsys):
    ev = play()
    TextRenderer({p.agent_id: p.name for p in PERSONAS}).render_game(ev, "truth")
    out = capsys.readouterr().out
    assert "NIGHT 1" in out and "Roles:" in out and "eliminated" in out
