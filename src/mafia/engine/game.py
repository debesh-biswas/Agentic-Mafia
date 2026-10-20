import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

from pydantic import ValidationError

from mafia.config import Persona, Rules
from mafia.engine import rules as R
from mafia.engine.sanitize import enforce_cap, sanitize, self_leak, word_count
from mafia.engine.scheduler import DiscussionScheduler
from mafia.engine.seeds import game_seed, rng_for
from mafia.engine.state import GameState, PlayerState
from mafia.engine.views import AgentView, TranscriptLine, build_view
from mafia.llm.client import LLMFormatError
from mafia.llm.schemas import SpeakDecision
from mafia.logs.events import Event
from mafia.types import Phase, Role, Team


class Invalid(Exception):
    """An agent answer that breaks a rule; the message is sent back on the retry."""


@dataclass
class GameResult:
    winner: Team
    end_reason: str
    days: int
    saves: int
    roles: dict[str, Role]


class Game:
    def __init__(self, rules: Rules, personas: list[Persona], agents: dict, run_id: str, game_index: int,
                 run_seed: int, group: str, scoring_version: int, prompt_version: int, record: bool = True, concurrent: bool = True):
        self.concurrent = concurrent  # plan/night agent calls in threads; results are still handled in seat order
        self.rules, self.agents, self.group = rules, agents, group
        self.run_seed = run_seed
        self.fb_rng = rng_for(run_seed, game_index, "fallback")
        self.versions = (scoring_version, prompt_version)
        self.events: list[Event] | None = [] if record else None
        self._seq = 0
        ps = sorted(personas, key=lambda p: p.seat)
        roles = [r for r, n in rules.roles.items() for _ in range(n)]
        self._roles = roles
        self.st = GameState(run_id, game_index, game_seed(run_seed, game_index),
                            {p.agent_id: PlayerState(p.agent_id, p.name, p.seat, Role.VILLAGER) for p in ps})
        self._by_name = {p.name.lower(): p.agent_id for p in ps}
        self.saves = 0
        self._sched: DiscussionScheduler | None = None
        self._reply: dict[str, str] = {}  # accused id -> accuser id, for reply_prompt
        self._prio = False
        self._votes_today: list[tuple[str, str]] = []

    # ---- plumbing -------------------------------------------------------------------------
    def emit(self, type: str, visibility: str, actor: str | None = None, targets: list[str] | None = None,
             **payload) -> None:
        """The single place events leave the engine."""
        if self.events is None:
            return
        self._seq += 1
        self.events.append(Event(
            run=self.st.run_id, game=self.st.game_index, seq=self._seq, ts=datetime.now(),
            day=self.st.day, phase=self.st.phase.value, type=type, actor=actor, targets=targets or [],
            visibility=visibility, payload=payload, scoring_version=self.versions[0],
            prompt_version=self.versions[1]))

    def stream(self, purpose: str):
        return rng_for(self.run_seed, self.st.game_index, purpose)

    def name(self, pid: str) -> str:
        return self.st.players[pid].name

    def view_for(self, agent_id: str, phase: Phase, retry_error: str | None = None) -> AgentView:
        s = self._sched
        speaking = phase is Phase.DISCUSSION and s is not None
        return build_view(
            self.st, agent_id, phase, votes=[(self.name(a), self.name(b)) for a, b in self._votes_today],
            turns_left=s.budget_remaining if speaking else None,
            my_turns_left=s.turns_left(agent_id) if speaking else None,
            reply_prompt=self._reply_prompt(agent_id) if speaking else None, retry_error=retry_error)

    def _reply_prompt(self, pid: str) -> str | None:
        return f"{self.name(self._reply[pid])} just accused you. You can respond now or pass." \
            if self._prio and pid in self._reply else None

    def _ask(self, pid: str, phase: Phase, method: str, validate, fallback):
        """DRD 5.9 for one agent, no threads."""
        return self._ask_all([(pid, validate, fallback)], phase, method, threads=False)[0]

    def _ask_all(self, items: list, phase: Phase, method: str, threads: bool = True) -> list:
        """DRD 5.9 for (pid, validate, fallback) items given in seat order: call, validate, one retry with the error
        in the view, then fallback. LLMFormatError (client already retried) goes straight to fallback; other
        LLMError/ProviderMismatch propagate. Only the agent calls run in threads (phases 1 and 3); validation,
        drains, fallbacks and emits happen here in item order, so the event log never depends on completion order."""
        parallel = threads and self.concurrent and len(items) > 1
        errs: dict[str, str | None] = {pid: None for pid, _, _ in items}
        whys: dict[str, str] = {}
        done: dict[str, object] = {}
        todo = list(items)
        for attempt in (0, 1):
            views = {pid: self.view_for(pid, phase, retry_error=errs[pid] and
                                        f"Your last answer was invalid: {errs[pid]}. Answer again with valid JSON.")
                     for pid, _, _ in todo}
            outs = self._run_agents(todo, method, views, parallel)
            nxt = []
            for (pid, validate, fallback), (dec, exc) in zip(todo, outs):
                self._drain(pid)
                try:
                    if exc is not None:
                        raise exc
                    done[pid] = validate(dec, attempt == 1)
                except (Invalid, ValidationError, ValueError, TypeError) as e:  # bad agent output; engine bugs propagate
                    errs[pid], whys[pid] = str(e) or type(e).__name__, f"{type(e).__name__}: {e}"
                    nxt.append((pid, validate, fallback))
                except LLMFormatError as e:
                    whys[pid] = f"{type(e).__name__}: {e}"
            todo = nxt
            if not todo:
                break
        # a pid that failed validation on the retry is in todo too; format errors are in neither list
        out = []
        for pid, _, fallback in items:
            if pid not in done:
                chosen = fallback()
                self.emit("fallback", "debug", pid, call_type=method, reason=whys[pid], chosen=str(chosen))
                done[pid] = chosen
            out.append(done[pid])
        return out

    def _run_agents(self, todo: list, method: str, views: dict, parallel: bool) -> list:
        """Agent calls only; returns (decision, exception) per item in item order."""
        def one(pid):
            try:
                return getattr(self.agents[pid], method)(views[pid]), None
            except Exception as e:  # re-raised or classified on the main thread
                return None, e
        pids = [pid for pid, _, _ in todo]
        if not parallel:
            return [one(p) for p in pids]
        with ThreadPoolExecutor(len(pids)) as ex:
            return list(ex.map(one, pids))

    def _drain(self, pid: str) -> None:
        """Optional agent hook: LLM call records become llm_call debug events."""
        for rec in getattr(self.agents[pid], "drain_records", lambda: [])():
            self.emit("llm_call", "debug", pid, **rec)

    def _cap(self, text: str, cap: int, final: bool, pid: str, method: str) -> str:
        """Sanitize, then enforce the word cap: error first time, truncate (logged) on the retry."""
        text = sanitize(text)
        if word_count(text) > cap:
            if not final:
                raise Invalid(f"at most {cap} words allowed, you wrote {word_count(text)}")
            text, _ = enforce_cap(text, cap)
            self.emit("fallback", "debug", pid, call_type=method, reason="truncated", chosen=None)
        return text

    def _id(self, name: str | None, legal: list[str], what: str) -> str:
        pid = self._by_name.get((name or "").strip().lower())
        if pid not in legal:
            raise Invalid(f"{what} must be one of: " + ", ".join(self.name(p) for p in legal))
        return pid

    # ---- phases ---------------------------------------------------------------------------
    def play(self) -> GameResult:
        st, rules = self.st, self.rules
        self.stream("roles").shuffle(self._roles)
        for p, r in zip(st.players.values(), self._roles):
            p.role = r
        self.emit("game_start", "truth", seed=st.seed, group=self.group,
                  seats={p.agent_id: p.seat for p in st.players.values()})
        for p in st.players.values():
            self.emit("role_assigned", "truth", p.agent_id, role=p.role.value)
        self.run_plans()
        while True:
            st.day += 1
            self.run_night()
            if self.check_win():
                break
            if st.day >= rules.max_day:
                self.end("town", "max_day")
                break
            self.run_discussion()
            self.run_vote()
            if self.check_win():
                break
        st.phase = Phase.END
        self.emit("game_end", "public", winner=st.winner, end_reason=st.end_reason, days=st.day)
        self.emit("role_reveal", "public", roles={p.agent_id: p.role.value for p in st.players.values()})
        return GameResult(st.winner, st.end_reason, st.day, self.saves,
                          {p.agent_id: p.role for p in st.players.values()})

    def run_plans(self) -> None:
        self.st.phase = Phase.PLAN
        items = [(pid, lambda d, f, pid=pid: self._cap(d.plan, self.rules.plan_max_words, f, pid, "plan"), lambda: "")
                 for pid in self.st.players]
        for pid, plan in zip(self.st.players, self._ask_all(items, Phase.PLAN, "plan")):
            if hasattr(self.agents[pid], "plan_accepted"):
                self.agents[pid].plan_accepted(plan)
            self.emit("plan", f"private:{pid}", pid, plan=plan)

    def run_night(self) -> None:
        st, rules = self.st, self.rules
        st.phase = Phase.NIGHT
        alive = R.living(st.players)
        doctor = next((p for p in alive if st.players[p].role is Role.DOCTOR), None)
        target = protect = None
        items, kinds = [], {}
        for pid in alive:
            p = st.players[pid]
            mafia, doc = p.role is Role.MAFIA, p.role is Role.DOCTOR
            legal = R.mafia_targets(st.players) if mafia else R.doctor_protects(
                st.players, pid, st.doctor_protected_self_last_night) if doc else []

            def validate(d, final, pid=pid, legal=legal, mafia=mafia, doc=doc):
                choice = self._id(d.target if mafia else d.protect, legal, "target" if mafia else "protect") \
                    if mafia or doc else None
                notes = self._cap(d.notes, rules.notes_max_words, final, pid, "night")
                return notes, min(max(d.urgency, rules.urgency_min), rules.urgency_max), choice

            def fallback(p=p, legal=legal, mafia=mafia, doc=doc):
                return p.notes, p.urgency, self.fb_rng.choice(legal) if mafia or doc else None

            items.append((pid, validate, fallback))
            kinds[pid] = mafia
        for pid, (notes, urgency, choice) in zip(alive, self._ask_all(items, Phase.NIGHT, "night")):
            p, mafia = st.players[pid], kinds[pid]
            p.notes, p.urgency = notes, urgency
            if hasattr(self.agents[pid], "notes_accepted"):
                self.agents[pid].notes_accepted(st.day, p.notes)
            self.emit("night_notes", f"private:{pid}", pid, notes=p.notes, urgency=p.urgency)
            if choice:
                self.emit("night_action", f"private:{pid}", pid, [choice], action="target" if mafia else "protect")
                if mafia:
                    target = choice
                else:
                    protect = choice
        if doctor:
            st.last_doctor_protect = protect
            st.doctor_protected_self_last_night = protect == doctor
        saved = target == protect
        self.saves += saved
        eliminated = None if saved else target
        self.emit("night_resolution", "truth", target=target, protect=protect, saved=saved, eliminated=eliminated)
        text = f"Morning of day {st.day}: nobody was eliminated." if saved else \
            f"Morning of day {st.day}: {self.name(target)} was eliminated during the night."
        if eliminated:
            st.players[eliminated].alive, st.players[eliminated].eliminated_on = False, f"night{st.day}"
        st.phase = Phase.MORNING
        self.emit("morning_announcement", "public", text=text, eliminated=eliminated)
        st.public_facts.append(text)

    def run_discussion(self) -> None:
        st, rules = self.st, self.rules
        st.phase = Phase.DISCUSSION
        st.today_transcript, self._reply = [], {}
        alive = R.living(st.players)
        self._sched = s = DiscussionScheduler(alive, {p: st.players[p].urgency for p in alive}, rules, self.stream(f"sched{st.day}"))
        while nxt := s.next_speaker():
            pid, prio = nxt
            self._prio = prio
            others = [p for p in alive if p != pid]

            def validate(d, final, pid=pid, others=others):
                msg = None if d.message is None else self._cap(d.message, rules.speech_max_words, final, pid, "speak")
                acc = []
                for n in d.accuses if msg else []:
                    try:
                        acc.append(self._id(n, others, "accuses"))
                    except Invalid:
                        if not final:
                            raise
                return SpeakDecision(message=msg or None, accuses=list(dict.fromkeys(acc)),
                                     urgency=min(max(d.urgency, rules.urgency_min), rules.urgency_max),
                                     expression=d.expression)

            dec = self._ask(pid, Phase.DISCUSSION, "speak", validate,
                            lambda pid=pid: SpeakDecision(message=None, urgency=st.players[pid].urgency))
            before = set(s.priority_queue)
            s.record_turn(pid, dec, prio)
            for a in set(s.priority_queue) - before:
                self._reply[a] = pid
            if prio:
                self._reply.pop(pid, None)
            if dec.message is None:
                st.today_transcript.append(TranscriptLine(speaker=self.name(pid), message="", kind="pass"))
                self.emit("pass", "public", pid, priority_reply=prio)
            else:
                st.today_transcript.append(TranscriptLine(speaker=self.name(pid), message=dec.message, kind="speech"))
                self.emit("speech", "public", pid, dec.accuses, message=dec.message, accuses=dec.accuses,
                          urgency=dec.urgency, expression=dec.expression, word_count=word_count(dec.message),
                          priority_reply=prio, self_leak=st.players[pid].role is Role.MAFIA and self_leak(dec.message))
            st.players[pid].urgency = dec.urgency
        self._sched = None

    def run_vote(self) -> None:
        st, rules = self.st, self.rules
        st.phase = Phase.VOTE
        order = R.living(st.players)
        self.stream(f"votes{st.day}").shuffle(order)
        self._votes_today = []
        for i, pid in enumerate(order):
            legal = R.vote_targets(st.players, pid)
            tgt = self._ask(pid, Phase.VOTE, "vote",
                            lambda d, f, legal=legal: self._id(d.vote, legal, "vote"),
                            lambda legal=legal: self.fb_rng.choice(legal))
            self._votes_today.append((pid, tgt))
            self.emit("vote", "public", pid, [tgt], order_index=i)
        counts, out = R.tally(self._votes_today)
        line = ", ".join(f"{self.name(a)}→{self.name(b)}" for a, b in self._votes_today)
        result = f"{self.name(out)} was voted out." if out else "tie, nobody was voted out."
        st.public_facts.append(f"Day {st.day} votes: {line}. Result: {result}")
        st.vote_history.append({"day": st.day, "votes": list(self._votes_today), "result": out})
        self.emit("vote_result", "public", tally=counts, eliminated=out, tie=out is None)
        if out:
            st.players[out].alive, st.players[out].eliminated_on = False, f"day{st.day}"
            lw = self._ask(out, Phase.VOTE, "last_words",
                           lambda d, f: (self._cap(d.message, rules.last_words_max_words, f, out, "last_words"), d.expression),
                           lambda: ("", None))
            st.today_transcript.append(TranscriptLine(speaker=self.name(out), message=lw[0], kind="last_words"))
            st.public_facts.append(f"{self.name(out)}'s last words: \"{lw[0]}\"")
            self.emit("last_words", "public", out, message=lw[0], expression=lw[1], word_count=word_count(lw[0]))
        self._votes_today = []

    def check_win(self) -> bool:
        res = R.check_win(self.st.players)
        if res:
            self.end(*res)
        return bool(res)

    def end(self, winner: Team, reason: str) -> None:
        self.st.winner, self.st.end_reason = winner, reason
