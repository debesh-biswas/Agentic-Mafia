import random
from collections import deque

from mafia.config import Rules
from mafia.llm.schemas import SpeakDecision


class DiscussionScheduler:
    """DRD 5.5. One instance per day."""

    def __init__(self, alive: list[str], urgency: dict[str, int], rules: Rules, rng: random.Random):
        self.alive = list(alive)
        self.urgency = {p: urgency[p] for p in alive}
        self.rules = rules
        self.rng = rng
        self.budget_remaining = rules.turns_per_player_per_day * len(alive)
        self.turns_used = {p: 0 for p in alive}
        self.priority_queue: deque[str] = deque()
        self.granted_pairs: set[tuple[str, str]] = set()
        self.consecutive_passes = 0

    def turns_left(self, pid: str) -> int:
        return self.rules.max_turns_per_player_per_day - self.turns_used[pid]

    def next_speaker(self) -> tuple[str, bool] | None:
        if self.budget_remaining <= 0:
            return None
        eligible = [p for p in self.alive if self.turns_left(p) > 0]
        if not eligible or self.consecutive_passes >= len(eligible):
            return None
        while self.priority_queue:
            pid = self.priority_queue.popleft()
            if pid in eligible:
                return pid, True
        top = max(self.urgency[p] for p in eligible)
        return self.rng.choice(sorted(p for p in eligible if self.urgency[p] == top)), False

    def record_turn(self, pid: str, decision: SpeakDecision, is_priority: bool = False) -> None:
        free = decision.message is None and is_priority and not self.rules.priority_pass_consumes_turn
        if not free:
            self.budget_remaining -= 1
            self.turns_used[pid] += 1
        self.urgency[pid] = decision.urgency
        if decision.message is None:
            self.consecutive_passes += 1
            return
        self.consecutive_passes = 0
        for accused in decision.accuses:
            # once per ordered pair; a counter-accusation (accused->pid already granted) gets no priority
            if (pid, accused) not in self.granted_pairs and (accused, pid) not in self.granted_pairs \
                    and accused not in self.priority_queue:
                self.granted_pairs.add((pid, accused))
                self.priority_queue.append(accused)
