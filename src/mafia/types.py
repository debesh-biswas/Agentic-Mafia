from enum import StrEnum
from typing import Literal


class Role(StrEnum):
    MAFIA = "mafia"
    DOCTOR = "doctor"
    VILLAGER = "villager"


class Phase(StrEnum):
    SETUP = "setup"
    PLAN = "plan"
    NIGHT = "night"
    MORNING = "morning"
    DISCUSSION = "discussion"
    VOTE = "vote"
    END = "end"
    REFLECT = "reflect"


Team = Literal["town", "mafia"]  # doctor and villagers are town
