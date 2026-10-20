"""LLM-backed agent (DRD 6.1-6.2, 8.3). Control group: no memories, no leaderboard."""
from mafia.agents import context as C
from mafia.agents.persona import persona_block, starting_summaries
from mafia.llm import schemas as S
from mafia.llm.client import LLMError


class LLMAgent:
    """One per season; per-game state resets in plan()."""

    def __init__(self, persona, personas, llm, rules, scoring, group, memory=None):
        # ponytail: memory group (retrieval, leaderboard) arrives in build step 7; `memory` unused until then
        self.persona, self.llm, self.rules, self.group, self.memory = persona, llm, rules, group, memory
        self.agent_id = persona.agent_id
        self.persona_text = persona_block(persona, personas)
        self.primer = C.rules_primer(scoring, rules)
        self.friends, self.self_summary = starting_summaries(persona, personas)
        if group == "memory":
            raise NotImplementedError("memory group not implemented yet")
        self.pending_records: list[dict] = []
        self.plan_text, self.notes_history, self.role_line = "", {}, ""

    def _call(self, call_type, schema, user, view):
        system = C.system_message(self.persona_text, self.primer, self.role_line, self.friends,
                                  self.self_summary, self.plan_text, None, "")
        try:
            parsed, rec = self.llm.call(call_type, system, user, schema, self.agent_id)
        except LLMError as e:
            self.pending_records.append(e.record)
            raise
        self.pending_records.append(rec)
        return parsed

    def drain_records(self) -> list[dict]:
        out, self.pending_records = self.pending_records, []
        return out

    def plan_accepted(self, text: str) -> None:
        self.plan_text = text

    def notes_accepted(self, day: int, notes: str) -> None:
        self.notes_history[day] = notes

    def plan(self, view):
        self.role_line, self.plan_text, self.notes_history = C.role_line(view.my_role), "", {}
        return self._call("plan", S.PlanDecision, C.user_message_plan(view, self.rules), view)

    def night(self, view):
        return self._call("night", S.NightDecision, C.user_message_night(view, self.rules), view)

    def speak(self, view):
        return self._call("speak", S.SpeakDecision, C.user_message_speak(view, self.rules), view)

    def vote(self, view):
        return self._call("vote", S.VoteDecision, C.user_message_vote(view, self.rules), view)

    def last_words(self, view):
        return self._call("last_words", S.LastWordsDecision, C.user_message_last_words(view, self.rules), view)

    def post_game(self, outcome) -> None:
        """Control group: nothing to learn."""
