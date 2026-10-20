from dataclasses import dataclass, field

from mafia.types import Phase, Role, Team


@dataclass
class PlayerState:
    agent_id: str
    name: str
    seat: int
    role: Role
    alive: bool = True
    eliminated_on: str | None = None  # "night2" | "day3"
    notes: str = ""
    urgency: int = 2


@dataclass
class GameState:
    run_id: str
    game_index: int
    seed: int
    players: dict[str, PlayerState]  # seat order
    day: int = 0  # night n precedes day n
    phase: Phase = Phase.SETUP
    last_doctor_protect: str | None = None  # agent_id
    doctor_protected_self_last_night: bool = False
    public_facts: list[str] = field(default_factory=list)
    vote_history: list[dict] = field(default_factory=list)  # {day, votes:[(voter,target)], result}
    today_transcript: list = field(default_factory=list)  # TranscriptLine
    winner: Team | None = None
    end_reason: str | None = None  # mafia_voted_out | parity | max_day
