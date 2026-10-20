"""Prompt assembly (DRD 6.3, 9). Reads only AgentView plus the agent's own game context."""
from mafia.config import Scoring
from mafia.engine.views import AgentView, TranscriptLine
from mafia.llm.prompts import template
from mafia.types import Role

ROLE_LINES = {
    Role.MAFIA: "YOUR ROLE THIS GAME: Mafia. Each night you choose someone to eliminate. You win if you survive until only 2 players remain. Nobody else knows your role.",
    Role.DOCTOR: "YOUR ROLE THIS GAME: Doctor (Town). Each night you protect one player. Help the Town find and vote out the Mafia.",
    Role.VILLAGER: "YOUR ROLE THIS GAME: Villager (Town). You have no night action. Help the Town find and vote out the Mafia.",
}


def scoring_summary(s: Scoring) -> str:
    b = s.bonus
    return (f"winning team: Town {s.win_points['town']} each, Mafia {s.win_points['mafia']}; "
            f"+{b['town_vote_on_mafia']} for each vote you cast on the Mafia (Town); "
            f"+{b['doctor_successful_save']} for each successful save (Doctor); "
            f"+{b['mafia_town_mislynch']} each time the Town votes out a non-Mafia player (Mafia). "
            "Eliminated players still get their team's points.")


def rules_primer(s: Scoring) -> str:
    return template("rules_primer").format(scoring_summary=scoring_summary(s))


def role_line(role: Role) -> str:
    return ROLE_LINES[role]


def system_message(persona_block: str, rules_primer: str, role_line: str, friend_summaries: str,
                   self_summary: str, game_plan: str, leaderboard_block: str | None,
                   memories_text: str) -> str:
    """leaderboard_block None (control) drops the block and its header; memories_text "" reads "(none)"."""
    return template("system").format(
        persona_block=persona_block, rules_primer=rules_primer, role_line=role_line,
        friend_summaries=friend_summaries, self_summary=self_summary, game_plan=game_plan or "(none)",
        leaderboard_block=leaderboard_block if leaderboard_block is not None else "",
        retrieved_memories=memories_text or "(none)")


def _facts(v: AgentView) -> str:
    return "\n".join(v.public_facts) or "(none yet)"


def _names(xs: list[str]) -> str:
    return ", ".join(xs) or "nobody"


def _line(t: TranscriptLine) -> str:
    if t.kind == "pass":
        return f"{t.speaker} passed."
    return f"{t.speaker} (last words): {t.message}" if t.kind == "last_words" else f"{t.speaker}: {t.message}"


def _transcript(v: AgentView) -> str:
    return "\n".join(_line(t) for t in v.today_transcript or []) or "(nobody has spoken yet)"


def _with_retry(text: str, v: AgentView) -> str:
    return f"{text}\n\n{v.retry_error}" if v.retry_error else text


def user_message_plan(v: AgentView) -> str:
    return _with_retry(template("plan").format(role=v.my_role.value.capitalize()), v)


def user_message_night(v: AgentView) -> str:
    if v.my_role is Role.MAFIA:
        instr, field = "Choose one player to eliminate tonight (not yourself).", ', "target": name'
    elif v.my_role is Role.DOCTOR:
        instr = "Choose one player to protect tonight." + (
            f" Last night you protected {v.doctor_last_protect}." if v.doctor_last_protect else "")
        instr += " You can't protect yourself tonight." if v.doctor_cannot_self_protect else ""
        field = ', "protect": name'
    else:
        instr, field = "You have no night action.", ""
    return _with_retry(template("night").format(
        n=v.day, alive=_names(v.alive), dead=_names(v.dead), public_facts=_facts(v),
        notes=v.my_notes or "(none yet)", role_instruction=instr, target_field=field), v)


def user_message_speak(v: AgentView) -> str:
    return _with_retry(template("speak").format(
        n=v.day, alive=_names(v.alive), total=v.turns_left_today, mine=v.my_turns_left,
        public_facts=_facts(v), notes=v.my_notes or "(none yet)", today_transcript=_transcript(v),
        reply_line=v.reply_prompt or "", name=v.me), v)


def user_message_vote(v: AgentView) -> str:
    votes = "; ".join(f"{a} voted {b}" for a, b in v.votes_so_far_today or []) or "none yet"
    return _with_retry(template("vote").format(
        n=v.day, alive=_names(v.alive), public_facts=_facts(v), today_transcript=_transcript(v),
        votes_so_far=votes), v)


def user_message_last_words(v: AgentView) -> str:
    return _with_retry(template("last_words"), v)
