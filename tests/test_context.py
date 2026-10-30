import re

from mafia.agents.context import (
    rules_primer, role_line, scoring_summary, system_message, user_message_last_words,
    user_message_night, user_message_plan, user_message_speak, user_message_vote,
)
from mafia.agents.persona import persona_block, starting_summaries
from mafia.agents.random_agent import random_agents
from mafia.config import ROOT, Personas, Rules, Scoring, load
from mafia.engine.game import Game
from mafia.engine.views import AgentView
from mafia.llm.schemas import NightDecision, SpeakDecision, VoteDecision
from mafia.types import Phase, Role

R = load(Rules, ROOT / "config/rules.yaml")
PS = load(Personas, ROOT / "config/personas.yaml")
SC = load(Scoring, ROOT / "config/scoring_v0.yaml")
IDS = [p.agent_id for p in sorted(PS.personas, key=lambda p: p.seat)]
NAME = {p.agent_id: p.name for p in PS.personas}
ROLE_WORDS = r"mafia|doctor|villager"
LB = "LEADERBOARD (after 37 games)\n1. Sana 212 (+13 last game)  2. Dev 205 (+0)"


def system_for(p, role, leaderboard=None, memories=""):
    fs, me = starting_summaries(p, PS)
    return system_message(persona_block(p, PS), rules_primer(SC, R), role_line(role), fs, me,
                          "Play it by ear.", leaderboard, memories)


class PromptAgent:
    """Wraps RandomAgent, renders system+user prompts on every call, and answers invalidly on some calls."""

    def __init__(self, inner, persona):
        self.inner, self.persona, self.agent_id, self.n = inner, persona, inner.agent_id, 0
        self.sentinels, self.prompts = set(), []  # (method, view, system, user)

    def _render(self, method, view, fn):
        self.n += 1
        system = system_for(self.persona, view.my_role, LB if self.n % 2 else None)
        self.prompts.append((method, view, system, fn(view)))
        return self.n % 4 == 0 and view.retry_error is None  # misbehave on every 4th fresh call

    def plan(self, v):
        self._render("plan", v, lambda x: user_message_plan(x, R))
        return self.inner.plan(v)

    def night(self, v):
        bad = self._render("night", v, lambda x: user_message_night(x, R))
        d = self.inner.night(v)
        d.notes = f"zq{self.agent_id}x{self.inner.rng.randrange(10**9)}"  # unique sentinel per agent per night
        self.sentinels.add(d.notes)
        if bad and v.my_role is Role.MAFIA:
            return NightDecision(notes=d.notes, urgency=d.urgency, target=v.me)
        return d

    def speak(self, v):
        bad = self._render("speak", v, lambda x: user_message_speak(x, R))
        d = self.inner.speak(v)
        return SpeakDecision(message="word " * 80, urgency=1) if bad else d

    def vote(self, v):
        bad = self._render("vote", v, lambda x: user_message_vote(x, R))
        d = self.inner.vote(v)
        return VoteDecision(vote=v.me) if bad else d

    def last_words(self, v):
        self._render("last_words", v, lambda x: user_message_last_words(x, R))
        return self.inner.last_words(v)


def play(seed):
    inner = random_agents(R, seed, 0, IDS)
    agents = {i: PromptAgent(a, next(p for p in PS.personas if p.agent_id == i)) for i, a in inner.items()}
    g = Game(R, PS.personas, agents, "t", 0, seed, "test", 0, 0)
    g.play()
    return g, agents


def test_rendered_prompts_leak_nothing_over_100_games():
    retries = calls = 0
    for seed in range(100):
        g, agents = play(seed)
        roles = {p.agent_id: p.role for p in g.st.players.values()}
        for pid, a in agents.items():
            others = [NAME[o] for o in IDS if o != pid]
            for method, v, system, user in a.prompts:
                calls += 1
                retries += v.retry_error is not None
                text = system + "\n" + user
                assert not re.search(r"\{[a-z_0-9]+\}", text), text  # unfilled placeholder
                assert "kill" not in text.lower()
                for o in others:  # no other player's role word adjacent to their name
                    assert not re.search(rf"{o}\W{{0,4}}({ROLE_WORDS})|({ROLE_WORDS})\W{{0,4}}{o}", text, re.I), (seed, pid, o)
                for o in IDS:  # other agents' private notes
                    if o != pid:
                        assert not any(t in text for t in agents[o].sentinels), (seed, pid, o)
                if roles[pid] is not Role.DOCTOR:
                    assert "you protected" not in text.lower()
                if method != "plan":  # own role only ever named in role line / plan; user text has no role words
                    assert not re.search(ROLE_WORDS, user, re.I), user
                if method != "night" or roles[pid] is Role.VILLAGER:
                    assert not re.search(r"protect|target|saved", user, re.I), (method, user)
                if method == "night" and roles[pid] is Role.MAFIA:
                    assert "protect" not in user.lower()
                if method == "night" and roles[pid] is Role.DOCTOR:
                    assert "target" not in user.lower()
                assert not re.search(r"expression\W+\w+\s*\n|urgency: \d", user)
    assert calls and retries > 50


def test_word_caps_match_rules():
    g, agents = play(3)
    seen = {m: u for a in agents.values() for m, v, s, u in a.prompts}
    assert f"max {R.plan_max_words} words" in seen["plan"]
    assert f"max {R.notes_max_words} words" in seen["night"]
    assert f"max {R.speech_max_words} words" in seen["speak"]
    assert f"max {R.last_words_max_words} words" in seen["last_words"]
    assert "at most 60 words" in rules_primer(SC, R) and "reaches day 8" in rules_primer(SC, R)


def test_caps_come_from_rules():
    r = R.model_copy(update=dict(speech_max_words=33, last_words_max_words=7, notes_max_words=99,
                                 plan_max_words=55, max_day=5))
    assert "at most 33 words" in rules_primer(SC, r) and "(at most 7 words)" in rules_primer(SC, r)
    assert "day 5" in rules_primer(SC, r)
    v = view(phase=Phase.DISCUSSION, today_transcript=[], turns_left_today=1, my_turns_left=1)
    assert "max 33 words" in user_message_speak(v, r) and "max 99 words" in user_message_night(v, r)
    assert "max 55 words" in user_message_plan(v, r) and "max 7 words" in user_message_last_words(v, r)


def view(**kw):
    base = dict(me="Dev", my_role=Role.VILLAGER, phase=Phase.NIGHT, day=2, alive=["Dev", "Sana"], dead=["Kai"],
                public_facts=["Morning of day 1: Kai was eliminated during the night."], my_notes="hm")
    return AgentView(**{**base, **kw})


def test_scoring_summary_matches_drd_example():
    assert scoring_summary(SC) == (
        "winning team: Town 13 each, Mafia 8; +1 for each vote you cast on the Mafia (Town); "
        "+2 for each successful save (Doctor); +1 each time the Town votes out a non-Mafia player (Mafia). "
        "Eliminated players still get their team's points.")


def test_system_message_groups_and_persona():
    dev = PS.personas[0]
    mem = system_for(dev, Role.DOCTOR, LB, "m1: Tess is chaotic")
    ctl = system_for(dev, Role.DOCTOR, None, "")
    assert "LEADERBOARD (after 37 games)" in mem and "m1: Tess is chaotic" in mem
    assert "LEADERBOARD" not in ctl and "MEMORIES THAT COME TO MIND\n(none)" in ctl
    assert "YOUR ROLE THIS GAME: Doctor (Town)" in ctl
    friends = ctl.split("YOUR FRIENDS (your own impressions)\n")[1].split("YOURSELF")[0].strip().splitlines()
    assert [l.split(":")[0] for l in friends] == [f"- {NAME[i]}" for i in IDS if i != "dev"]
    assert "Leo would never lie to me." in "\n".join(friends)  # closest friend line, others only see impressions
    assert "You play to win" in ctl and "You change your mind slowly" in ctl and "You often misjudge" in ctl


def test_doctor_night_instruction():
    d = view(my_role=Role.DOCTOR, doctor_last_protect="Sana", doctor_cannot_self_protect=True)
    u = user_message_night(d, R)
    assert "Last night you protected Sana. You can't protect yourself tonight." in u and '"protect": name' in u
    first = user_message_night(view(my_role=Role.DOCTOR, day=1, doctor_cannot_self_protect=False), R)
    assert "protected" not in first and "can't" not in first


def test_speak_reply_line_and_retry_error():
    v = view(phase=Phase.DISCUSSION, today_transcript=[], turns_left_today=10, my_turns_left=2,
             reply_prompt="Tess just accused you. You can respond now or pass.", retry_error="Your last answer was invalid: x.")
    u = user_message_speak(v, R)
    assert "Tess just accused you." in u and u.endswith("Your last answer was invalid: x.")
    assert "Turns left today: 10 (yours: 2)" in u


def _speak_view(transcript):
    from mafia.engine.views import TranscriptLine
    return AgentView(me="Tess", my_role=Role.VILLAGER, day=2, phase=Phase.DISCUSSION, alive=["Tess", "Dev"], dead=[], public_facts=[], my_notes="",
                     today_transcript=[TranscriptLine(speaker=s, message=m, kind="speech") for s, m in transcript],
                     turns_left_today=4, my_turns_left=2)


def test_persona_block_plain_lines():
    b = persona_block(PS.personas[0], PS)
    assert "You're a real person, not a character" in b and "For tone only" in b


def test_speak_shows_own_messages_today():
    u = user_message_speak(_speak_view([("Tess", "why vote Dev?"), ("Dev", "no reason"), ("Tess", "still odd")]), R)
    assert "Your messages so far today: why vote Dev? | still odd" in u
    assert "Your messages so far today: (none yet)" in user_message_speak(_speak_view([("Dev", "hi")]), R)


def test_reflect_insights_plain_notes():
    from mafia.llm.prompts import template
    assert "Write like private notes to yourself" in template("reflect_insights")
