"""Record/replay of LLM responses (llm_cache/<game>.jsonl) and a scripted MockLLM for tests.
All expose the same acall/call signature as LLMClient (duck typing)."""
import asyncio
import json
import random
import re
import threading
import zlib
from collections import Counter
from pathlib import Path

from mafia.llm.client import LLMError, LLMFormatError, parse_with_retry
from pydantic import BaseModel


class _Sync:
    def call(self, call_type, system, user, schema, agent_id):
        return asyncio.run(self.acall(call_type, system, user, schema, agent_id))


class RecordingLLM:
    """Wraps a client for one game: appends one JSON line per call. call_index counts per (agent_id, call_type).
    Thread-safe: the index is taken under a lock before the inner call, so concurrent calls keep distinct keys."""

    def __init__(self, inner, path):
        self.inner, self.path, self.n, self._lock = inner, Path(path), Counter(), threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")  # one file per game: a rerun starts clean

    def _key(self, call_type, agent_id) -> dict:
        with self._lock:
            i = self.n[agent_id, call_type]
            self.n[agent_id, call_type] += 1
        return {"key": [agent_id, call_type, i]}

    def _ok(self, row, model, rec):
        self._write(row | {"response": model.model_dump(mode="json"), "record": rec})

    def _err(self, row, e):
        self._write(row | {"error": str(e), "format": isinstance(e, LLMFormatError), "record": e.record})

    def call(self, call_type, system, user, schema, agent_id):
        row = self._key(call_type, agent_id)
        try:
            model, rec = self.inner.call(call_type, system, user, schema, agent_id)
        except LLMError as e:
            self._err(row, e)
            raise
        self._ok(row, model, rec)
        return model, rec

    async def acall(self, call_type, system, user, schema, agent_id):
        row = self._key(call_type, agent_id)
        try:
            model, rec = await self.inner.acall(call_type, system, user, schema, agent_id)
        except LLMError as e:
            self._err(row, e)
            raise
        self._ok(row, model, rec)
        return model, rec

    def _write(self, row):
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


class ReplayLLM(_Sync):
    def __init__(self, path):
        self.rows = {}
        for r in map(json.loads, Path(path).read_text().splitlines()):
            if tuple(r["key"]) in self.rows:
                raise ValueError(f"duplicate recorded key {r['key']} in {path}")
            self.rows[tuple(r["key"])] = r
        self.n, self._lock = Counter(), threading.Lock()

    async def acall(self, call_type, system, user, schema, agent_id):
        with self._lock:
            i = self.n[agent_id, call_type]
            self.n[agent_id, call_type] += 1
        row = self.rows.get((agent_id, call_type, i))
        if row is None:
            raise KeyError(f"no recorded response for (agent={agent_id!r}, call_type={call_type!r}, call_index={i})")
        if "error" not in row and "response" not in row or "error" in row and not {"format", "record"} <= row.keys():
            raise LLMError(f"malformed recorded row for key {row['key']}: {sorted(row)}")
        if "error" in row:
            raise (LLMFormatError if row["format"] else LLMError)(row["error"], row["record"])
        return schema.model_validate(row["response"]), row["record"]


class MockLLM(_Sync):
    """Scripted raw replies (str, consumed in order, retries included), or a callable (call_type, user) -> str.
    Runs the real parse/corrective-retry path."""

    def __init__(self, script):
        self.script, self.sent, self._lock = script, [], threading.Lock()

    async def acall(self, call_type, system, user, schema: type[BaseModel], agent_id):
        async def send(messages):
            with self._lock:
                self.sent.append(list(messages))
                text = self.script(call_type, messages[-1]["content"]) if callable(self.script) else self.script.pop(0)
            return text, {}
        return await parse_with_retry(send, system, user, schema, call_type, agent_id)


class ScriptedLLM(MockLLM):
    """Deterministic valid JSON for every call type, from the prompt's Alive list (for --mock and tests)."""

    def __init__(self, seed: int, names: dict[str, str]):
        super().__init__(self._reply)
        self.seed, self.names, self._tl = seed, names, threading.local()  # agent id is per thread

    async def acall(self, call_type, system, user, schema, agent_id):
        self._tl.agent = agent_id
        return await super().acall(call_type, system, user, schema, agent_id)

    def _reply(self, call_type, user):
        r = random.Random(zlib.crc32(f"{self.seed}|{self._tl.agent}|{call_type}|{user}".encode()))
        me = self.names[self._tl.agent]
        m = re.search(r"Alive: ([^.]*)\.", user)
        others = [n for n in (m[1].split(", ") if m else []) if n != me]
        if call_type == "plan":
            out = {"plan": "Stay observant and vote on evidence."}
        elif call_type == "night":
            out = {"notes": f"Night thoughts {r.randint(0, 99)}.", "urgency": r.randint(0, 4)}
            if '"target"' in user:
                out["target"] = r.choice(others)
            elif '"protect"' in user:
                out["protect"] = r.choice(others)
        elif call_type == "speak":
            talk = r.random() > 0.3
            out = {"message": f"I think {others[0]} is worth watching." if talk else None,
                   "accuses": [r.choice(others)] if talk and r.random() < 0.4 else [],
                   "urgency": r.randint(0, 4), "expression": "thinking"}
        elif call_type == "vote":
            out = {"vote": r.choice(others)}
        else:
            out = {"message": "Good luck everyone.", "expression": "neutral"}
        return json.dumps(out)
