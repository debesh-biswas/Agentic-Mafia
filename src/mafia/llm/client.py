"""LLM client over the openai SDK (DRD section 10). Engine-facing interface:
acall/call(call_type, system, user, schema, agent_id) -> (parsed model, call record)."""
import asyncio
import hashlib
import os
import re
import threading
import time

import openai
from mafia.config import ModelCfg
from pydantic import BaseModel, ValidationError

ATTEMPTS = 5  # network/5xx tries per request


class LLMError(Exception):
    def __init__(self, msg: str, record: dict):
        super().__init__(msg)
        self.record = record


class LLMFormatError(LLMError):
    """Still unparseable after the one corrective retry; the engine goes straight to its fallback (no second agent call)."""


class ProviderMismatch(RuntimeError):
    """OpenRouter served the call from a provider other than the pinned one: abort the run."""


def strip_reasoning(text: str) -> str:
    """Drop <think> blocks, Harmony analysis channels and ```json fences."""
    text = re.sub(r".*</think>", "", text, flags=re.S)
    text = re.sub(r"<think>.*$", "", text, flags=re.S)  # unclosed: cut off mid-thought
    final = re.findall(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|end\|>|<\|return\|>|$)", text, flags=re.S)
    if final:
        text = final[-1]
    else:
        text = re.sub(r"<\|channel\|>\w+<\|message\|>.*?<\|end\|>", "", text, flags=re.S)
    text = re.sub(r"<\|\w+\|>", "", text).strip()
    m = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    return m.group(1) if m else text


def _error_text(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(map(str, x['loc'])) or 'answer'}: {x['msg']}" for x in e.errors()[:5])


async def parse_with_retry(send, system: str, user: str, schema: type[BaseModel], call_type: str, agent_id: str):
    """send(messages) -> (text, meta). One corrective retry on parse error (DRD 5.9), counted as format_error."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    rec = {"call_type": call_type, "agent": agent_id, "tokens_in": 0, "tokens_out": 0, "tokens_reasoning": 0,
           "retries": 0, "format_errors": 0, "error": None, "provider": None, "cost": None,
           "prompt_hash": hashlib.sha256(f"{system}\n{user}\n{schema.__name__}".encode()).hexdigest()[:12]}
    t0 = time.monotonic()
    try:
        for attempt in range(2):
            text, meta = await send(messages)
            for k in ("tokens_in", "tokens_out", "tokens_reasoning", "retries"):
                rec[k] += meta.get(k, 0)
            rec["provider"] = meta.get("provider") or rec["provider"]
            rec["finish_reason"] = meta.get("finish_reason")
            if meta.get("cost") is not None:
                rec["cost"] = (rec["cost"] or 0) + meta["cost"]
            if meta.get("reasoning"):
                rec["reasoning"] = meta["reasoning"]  # debug only, never fed back to agents
            clean = strip_reasoning(text or "")
            if meta.get("finish_reason") == "length" or not clean:
                rec["format_errors"] += 1
                rec["error"] = ("truncated at max_tokens (reasoning may have consumed the budget)"
                                if meta.get("finish_reason") == "length" else "empty content")
                raise LLMFormatError(f"{call_type}/{agent_id}: {rec['error']}", rec)  # a retry would hit the same wall
            try:
                return schema.model_validate_json(clean), rec
            except ValidationError as e:
                rec["format_errors"] += 1
                err = _error_text(e)
                rec["error"] = err
                messages += [{"role": "assistant", "content": text or ""},
                             {"role": "user", "content": f"Your last answer was invalid: {err}. Answer again with valid JSON."}]
        raise LLMFormatError(f"{call_type}/{agent_id}: invalid after retry: {rec['error']}", rec)
    except LLMError:
        raise
    except ProviderMismatch:
        raise
    except Exception as e:  # network retries exhausted or non-retryable 4xx
        rec["error"] = f"{type(e).__name__}: {e}"
        raise LLMError(rec["error"], rec) from e
    finally:
        rec["latency_s"] = round(time.monotonic() - t0, 3)


def _retryable(e: Exception) -> bool:
    code = getattr(e, "status_code", None)
    return isinstance(e, openai.APIConnectionError) or code == 429 or (code is not None and code >= 500)


class LLMClient:
    backoff_s = 1.0  # first retry delay; doubles each attempt

    def __init__(self, cfg: ModelCfg, sdk=None):
        self.cfg = cfg
        self._sdk_override = sdk  # tests pass an object shaped like AsyncOpenAI
        if sdk is None and cfg.api_key_env and not os.environ.get(cfg.api_key_env):
            raise RuntimeError(f"environment variable {cfg.api_key_env} is not set or empty")
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._st = None  # (sdk, semaphore), created once and reused

    def _state(self):
        with self._lock:
            if self._st is None:
                key = os.environ.get(self.cfg.api_key_env) if self.cfg.api_key_env else "none"
                sdk = self._sdk_override or openai.AsyncOpenAI(
                    base_url=self.cfg.base_url, api_key=key or "none", timeout=self.cfg.timeout_s, max_retries=0)
                self._st = (sdk, asyncio.Semaphore(self.cfg.max_concurrency))  # binds to the loop on first use
            return self._st

    def _loop_running(self) -> asyncio.AbstractEventLoop:
        """One persistent loop on a daemon thread; every sync call() is submitted to it."""
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                self._thread = threading.Thread(target=loop.run_forever, name="llm-loop", daemon=True)
                self._thread.start()
                self._loop = loop
            return self._loop

    def close(self) -> None:
        with self._lock:
            loop, thread, st, self._loop, self._thread, self._st = self._loop, self._thread, self._st, None, None, None
        try:
            if st and loop is not None:
                asyncio.run_coroutine_threadsafe(st[0].close(), loop).result()
        finally:
            if loop is not None:
                loop.call_soon_threadsafe(loop.stop)
                thread.join()
                loop.close()

    def _kwargs(self, call_type: str, schema: type[BaseModel]) -> dict:
        c = self.cfg
        mt = c.max_output_tokens
        kw: dict = {"model": c.model, "temperature": c.temperature,
                    "max_tokens": mt.get(call_type) or mt.get("reflect" if call_type == "deep_reflect" else "default", mt["default"])}
        extra: dict = {}
        level = c.reasoning.get(call_type) or c.reasoning.get("reflect")
        if c.reasoning_param == "openrouter":
            extra["reasoning"] = {"effort": level}
        elif c.reasoning_param == "reasoning_effort":
            kw["reasoning_effort"] = level
        if c.provider_pin:
            extra["provider"] = c.provider_pin
        if extra:
            kw["extra_body"] = extra
        if c.structured_output == "json_schema":
            # strict=False: pydantic's schema isn't strict-compatible (optional fields, defaults); pydantic validates anyway.
            kw["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.__name__, "schema": schema.model_json_schema(), "strict": False}}
        elif c.structured_output == "json_object":
            kw["response_format"] = {"type": "json_object"}
        return kw

    async def _send(self, messages, kw):
        sdk, sem = self._state()
        retries = 0
        async with sem:
            while True:
                try:
                    r = await sdk.chat.completions.create(messages=messages, **kw)
                    break
                except Exception as e:
                    if not _retryable(e) or retries + 1 >= ATTEMPTS:
                        raise
                    await asyncio.sleep(self.backoff_s * 2 ** retries)
                    retries += 1
        msg, u = r.choices[0].message, getattr(r, "usage", None)
        provider = getattr(r, "provider", None)
        pin = (self.cfg.provider_pin or {}).get("order")
        if self.cfg.provider_pin and (not provider or provider.lower() not in {p.lower() for p in pin or []}):
            raise ProviderMismatch(f"served by {provider!r}, pinned to {pin}")
        details = getattr(u, "completion_tokens_details", None)
        return msg.content, {
            "tokens_in": getattr(u, "prompt_tokens", 0) or 0, "tokens_out": getattr(u, "completion_tokens", 0) or 0,
            "tokens_reasoning": getattr(details, "reasoning_tokens", 0) or 0, "retries": retries,
            "provider": provider, "finish_reason": r.choices[0].finish_reason, "cost": getattr(u, "cost", None),
            "reasoning": getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)}

    async def acall(self, call_type: str, system: str, user: str, schema: type[BaseModel], agent_id: str):
        """Must run on this client's loop (the semaphore binds to it); from other threads use call()."""
        kw = self._kwargs(call_type, schema)
        return await parse_with_retry(lambda m: self._send(m, kw), system, user, schema, call_type, agent_id)

    def call(self, call_type, system, user, schema, agent_id):
        """Thread-safe: any number of threads may call at once; max_concurrency bounds the requests in flight."""
        if threading.current_thread() is self._thread:
            raise RuntimeError("LLMClient.call() from the client's own loop thread would deadlock; await acall()")
        return asyncio.run_coroutine_threadsafe(
            self.acall(call_type, system, user, schema, agent_id), self._loop_running()).result()
