from mafia.engine.state import GameState
from mafia.types import Phase, Role
from pydantic import BaseModel


class TranscriptLine(BaseModel, frozen=True):
    speaker: str  # display name
    message: str  # "" for a pass
    kind: str  # speech | pass | last_words


class AgentView(BaseModel, frozen=True):
    """DRD 5.7. The only object prompt assembly may read from the game."""

    me: str
    my_role: Role
    phase: Phase
    day: int
    alive: list[str]
    dead: list[str]
    public_facts: list[str]
    my_notes: str
    today_transcript: list[TranscriptLine] | None = None
    votes_so_far_today: list[tuple[str, str]] | None = None
    turns_left_today: int | None = None
    my_turns_left: int | None = None
    reply_prompt: str | None = None
    doctor_last_protect: str | None = None
    doctor_cannot_self_protect: bool | None = None
    retry_error: str | None = None


def build_view(
    st: GameState, agent_id: str, phase: Phase, *, votes: list[tuple[str, str]] | None = None,
    turns_left: int | None = None, my_turns_left: int | None = None,
    reply_prompt: str | None = None, retry_error: str | None = None,
) -> AgentView:
    me = st.players[agent_id]
    talk = phase in (Phase.DISCUSSION, Phase.VOTE)
    doc = me.role is Role.DOCTOR
    # model_construct: the engine is the sole, trusted builder (speed for `mafia sim`)
    return AgentView.model_construct(
        me=me.name, my_role=me.role, phase=phase, day=st.day,
        alive=[p.name for p in st.players.values() if p.alive],
        dead=[p.name for p in st.players.values() if not p.alive],
        public_facts=list(st.public_facts), my_notes=me.notes,
        today_transcript=list(st.today_transcript) if talk else None,
        votes_so_far_today=list(votes or []) if phase is Phase.VOTE else None,
        turns_left_today=turns_left, my_turns_left=my_turns_left, reply_prompt=reply_prompt,
        doctor_last_protect=st.players[st.last_doctor_protect].name if doc and st.last_doctor_protect else None,
        doctor_cannot_self_protect=st.doctor_protected_self_last_night if doc else None,
        retry_error=retry_error,
    )
