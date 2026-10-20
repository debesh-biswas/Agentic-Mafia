from collections.abc import Iterable
from typing import Protocol

from mafia.logs.events import Event


class Renderer(Protocol):
    def render_game(self, events: Iterable[Event], view: str) -> None: ...
