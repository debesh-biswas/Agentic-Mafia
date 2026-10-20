"""Record/replay of LLM responses (llm_cache/<game>.jsonl) and a scripted MockLLM for tests.
All expose the same acall/call signature as LLMClient (duck typing)."""
import asyncio
import json
from collections import Counter
from pathlib import Path

from mafia.llm.client import LLMError, LLMFormatError, parse_with_retry
from pydantic import BaseModel


class _Sync:
    def call(self, call_type, system, user, schema, agent_id):
        return asyncio.run(self.acall(call_type, system, user, schema, agent_id))


class RecordingLLM(_Sync):
    """Wraps a client for one game: appends one JSON line per call. call_index counts per (agent_id, call_type)."""

    def __init__(self, inner, path):
        self.inner, self.path, self.n = inner, Path(path), Counter()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")  # one file per game: a rerun starts clean

    async def acall(self, call_type, system, user, schema, agent_id):
        i = self.n[agent_id, call_type]  # taken before the await so concurrent calls keep distinct indices
        self.n[agent_id, call_type] += 1
        row: dict = {"key": [agent_id, call_type, i]}
        try:
            model, rec = await self.inner.acall(call_type, system, user, schema, agent_id)
            row |= {"response": model.model_dump(mode="json"), "record": rec}
            self._write(row)
            return model, rec
        except LLMError as e:
            row |= {"error": str(e), "format": isinstance(e, LLMFormatError), "record": e.record}
            self._write(row)
            raise

    def _write(self, row):
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")


class ReplayLLM(_Sync):
    def __init__(self, path):
        self.rows = {}
        for r in map(json.loads, Path(path).read_text().splitlines()):
            if tuple(r["key"]) in self.rows:
                raise ValueError(f"duplicate recorded key {r['key']} in {path}")
            self.rows[tuple(r["key"])] = r
        self.n = Counter()

    async def acall(self, call_type, system, user, schema, agent_id):
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
        self.script, self.sent = script, []

    async def acall(self, call_type, system, user, schema: type[BaseModel], agent_id):
        async def send(messages):
            self.sent.append(list(messages))
            text = self.script(call_type, messages[-1]["content"]) if callable(self.script) else self.script.pop(0)
            return text, {}
        return await parse_with_retry(send, system, user, schema, call_type, agent_id)
