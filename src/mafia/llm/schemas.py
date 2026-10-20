import re
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

Name = str  # validated against living names by the engine, not by pydantic
Expression = Literal["neutral", "happy", "laughing", "suspicious", "defensive", "nervous", "angry", "surprised", "thinking"]
Urgency = Annotated[int, Field(ge=0, le=4)]
Importance = Annotated[int, Field(ge=1, le=10)]


def _mem_id(v):
    """Accept 42, "42" or "m42"."""
    if isinstance(v, str):
        m = re.fullmatch(r"m?(\d+)", v.strip())
        if not m:
            raise ValueError(f"bad memory id: {v!r}")
        return int(m[1])
    if isinstance(v, bool) or (isinstance(v, int) and v < 0):
        raise ValueError(f"bad memory id: {v!r}")
    return v


MemId = Annotated[int, BeforeValidator(_mem_id)]


class PlanDecision(BaseModel):
    plan: str


class NightDecision(BaseModel):
    notes: str
    urgency: Urgency
    target: Name | None = None
    protect: Name | None = None


class SpeakDecision(BaseModel):
    message: str | None
    accuses: list[Name] = []
    urgency: Urgency
    expression: Expression | None = None


class VoteDecision(BaseModel):
    vote: Name


class LastWordsDecision(BaseModel):
    message: str
    expression: Expression | None = None


class Questions(BaseModel):
    questions: Annotated[list[str], Field(min_length=1, max_length=3)]


class Insight(BaseModel):
    text: str
    kind: Literal["belief", "lesson", "reflection"]
    about: list[str] = []
    role_context: Literal["mafia", "doctor", "villager", "any"] | None = None
    confidence: Literal["hunch", "fairly sure", "certain"] = "hunch"
    importance: Importance
    cites: list[MemId] = []
    action: Literal["add", "update", "supersede"] = "add"
    target_id: MemId | None = None


class ReflectInsights(BaseModel):
    insights: Annotated[list[Insight], Field(max_length=5)]
    keep_ids: list[MemId] = []
    helped_or_hurt: str
    game_summary: str
    summary_importance: Importance


class Summaries(BaseModel):
    friends: dict[str, str]
    self: str


class DeepReflect(BaseModel):
    insights: Annotated[list[Insight], Field(max_length=3)]


class Compressed(BaseModel):
    summary: str
    importance: Importance
