"""LLM-backed agent (DRD 6.1-6.2, 8.3). Control group: no memories, no leaderboard."""
import dataclasses

from mafia.agents import context as C
from mafia.agents.persona import (SELF_INSIGHT, UPDATE_STYLE, friend_lines, persona_block, starting_impressions,
                                  starting_summaries)
from mafia.llm import schemas as S
from mafia.llm.client import LLMError
from mafia.memory.retrieval import mark_applied, render, retrieve, settle_lessons
from mafia.memory.maintenance import compress, fade
from mafia.reflection.deep import deep_reflect
from mafia.reflection.reflect import reflect
from mafia.reflection.summaries import summarize


class LLMAgent:
    """One per season; per-game state resets in plan()."""

    def __init__(self, persona, personas, llm, rules, scoring, group, memory=None, memory_cfg=None):
        self.persona, self.personas, self.llm, self.rules, self.group = persona, personas, llm, rules, group
        self.memory, self.memory_cfg = memory, memory_cfg  # MemoryStore + Memory config; memory group only
        self.agent_id = persona.agent_id
        self.names = {p.agent_id: p.name for p in personas.personas}
        self.persona_text = persona_block(persona, personas)
        self.primer = C.rules_primer(scoring, rules)
        self.friends, self.self_summary = starting_summaries(persona, personas)
        if group == "memory":
            if memory is None or memory_cfg is None:
                raise ValueError("memory group needs a MemoryStore and a memory config")
            if not memory.get_summaries():
                memory.seed(starting_impressions(persona, personas), self.self_summary)
        self.pending_records: list[dict] = []
        self.plan_text, self.notes_history, self.role_line = "", {}, ""
        self.game_no, self.leaderboard_text, self.memories_text, self._lb_in_system = 0, "", "", False

    def begin_game(self, game_no: int, leaderboard_text: str = "") -> None:
        """Runner sets this before each game; only the memory group ever shows the leaderboard."""
        self.game_no = game_no
        self.leaderboard_text = leaderboard_text if self.group == "memory" else ""

    def _system(self, leaderboard: bool = False) -> str:
        lb = self.leaderboard_text if leaderboard and self.group == "memory" else None
        return C.system_message(self.persona_text, self.primer, self.role_line, self.friends, self.self_summary,
                                self.plan_text, lb, self.memories_text)

    def _call(self, call_type, schema, user, leaderboard=False):
        try:
            parsed, rec = self.llm.call(call_type, self._system(leaderboard), user, schema, self.agent_id)
        except LLMError as e:
            self.pending_records.append(e.record)
            raise
        self.pending_records.append(rec)
        return parsed

    def _recall(self, query: str, k=None, types=None):
        mems = retrieve(self.memory, query, self.game_no, self.memory_cfg.retrieval, k=k, types=types)
        self.memories_text = render(mems, self.names)
        return mems

    def drain_records(self) -> list[dict]:
        out, self.pending_records = self.pending_records, []
        return [{k: v for k, v in r.items() if k != "reasoning"} for r in out]  # reasoning stays in llm_cache only

    def plan_accepted(self, text: str) -> None:
        self.plan_text = text

    def notes_accepted(self, day: int, notes: str) -> None:
        self.notes_history[day] = notes

    def plan(self, view):
        self.role_line, self.plan_text, self.notes_history = C.role_line(view.my_role), "", {}
        self.memories_text = ""
        if self.group == "memory":
            sums = self.memory.get_summaries()
            self.friends = friend_lines(self.persona, self.personas, sums)  # seeded at construction, so complete
            self.self_summary = sums.get("self", self.self_summary)
            names = ", ".join(self.names[a] for a in self.names if a != self.agent_id)
            role = view.my_role.value
            mems = self._recall(f"{role} strategy; {names}")
            mark_applied(self.memory, [m.id for m in mems if m.type == "lesson" and m.role_context in (role, "any")])
        return self._call("plan", S.PlanDecision, C.user_message_plan(view, self.rules), leaderboard=True)

    def night(self, view):
        if self.group == "memory":
            self._recall(view.my_notes or self.plan_text)
        return self._call("night", S.NightDecision, C.user_message_night(view, self.rules))

    def speak(self, view):
        return self._call("speak", S.SpeakDecision, C.user_message_speak(view, self.rules))

    def vote(self, view):
        return self._call("vote", S.VoteDecision, C.user_message_vote(view, self.rules))

    def last_words(self, view):
        return self._call("last_words", S.LastWordsDecision, C.user_message_last_words(view, self.rules))

    def post_game(self, outcome) -> list[tuple[str, str, dict]]:
        """Memory group: reflect, update summaries, settle lessons. Returns (type, visibility, payload) event specs
        for the runner to emit; control group learns nothing."""
        if self.group != "memory":
            return []
        self.memories_text = ""
        outcome = dataclasses.replace(outcome, update_style_line=UPDATE_STYLE[self.persona.update_style],
                                      self_insight_line=SELF_INSIGHT[self.persona.self_insight])

        def call(call_type, schema, user):
            return self._call(call_type, schema, user, leaderboard=False)

        try:
            specs = reflect(self.memory, self.memory_cfg, self.names, call, outcome)
            specs += summarize(self.memory, self.names, self.agent_id, call, outcome.game_no,
                               outcome.update_style_line, outcome.self_insight_line)
            g, mc = outcome.game_no, self.memory_cfg
            if g % mc.deep_reflection_every_games == 0:
                specs += deep_reflect(self.memory, self.names, call, self.agent_id, g)
            if g % mc.compression.every_games == 0:
                more, merged = compress(self.memory, mc, self.names, call, self.agent_id, g)
                specs += more + [("memory_maintenance", f"private:{self.agent_id}",
                                  dict(merged=merged, faded=fade(self.memory, mc, g)))]
        finally:
            settle_lessons(self.memory, outcome.won)
        return specs + [("llm_call", "debug", rec) for rec in self.drain_records()]
