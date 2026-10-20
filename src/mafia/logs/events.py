from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

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


class EventLog:
    """DRD 11.3: one JSONL file per game, all visibilities."""

    def __init__(self, run_dir: Path, game_index: int):
        # game_index is 0-based inside the engine; files are 1-based, 4 digits
        self.path = Path(run_dir) / "games" / f"{game_index + 1:04d}.jsonl"

    def write(self, events: Iterable[Event]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("".join(e.model_dump_json() + "\n" for e in events))

    def read(self) -> list[Event]:
        return [Event.model_validate_json(line) for line in self.path.read_text().splitlines() if line]
