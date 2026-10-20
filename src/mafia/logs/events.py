from datetime import datetime

from pydantic import BaseModel


class Event(BaseModel):
    """DRD 11.1. Visibility: public | private:<agent_id> | truth | debug."""

    run: str
    game: int
    seq: int
    ts: datetime
    day: int
    phase: str
    type: str
    actor: str | None = None
    targets: list[str] = []
    visibility: str
    payload: dict = {}
    scoring_version: int
    prompt_version: int
