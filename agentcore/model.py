"""One model call, done safely: per-attempt timeout, retry with backoff, a trace span for every attempt."""

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from .trace import Span, Trace


@dataclass(frozen=True)
class CallConfig:
    timeout_s: float = 30.0  # one attempt
    overall_timeout_s: float = 90.0  # all attempts and waits together
    max_attempts: int = 3
    base_delay_s: float = 1.0
    max_delay_s: float = 20.0
    jitter: bool = True
    params: dict = field(default_factory=dict)  # extra request parameters, e.g. {"thinking": {"type": "disabled"}}
    sleep: Callable[[float], None] = time.sleep  # tests replace these two
    clock: Callable[[], float] = time.monotonic


class ModelCallError(Exception):
    """The call failed for good. Carries the exception type and status only, never the SDK's text."""

    def __init__(self, reason: str, attempts: int, status_code: int | None = None):
        super().__init__(f"model call failed after {attempts} attempt(s): {reason}")
        self.reason, self.attempts, self.status_code = reason, attempts, status_code


def response_text(response: Any) -> str:
    return "".join(b.text for b in response.content if b.type == "text")


def _summarize(content: Any) -> Any:
    """Readable form of a message's content for the trace. Thinking blocks are left out."""
    if isinstance(content, str):
        return content
    out = []
    for block in content:
        get = block.get if isinstance(block, dict) else lambda key, default=None: getattr(block, key, default)
        kind = get("type")
        if kind == "text":
            out.append(get("text"))
        elif kind == "tool_use":
            out.append({"tool_use": get("name"), "input": get("input")})
        elif kind == "tool_result":
            out.append({"tool_result": get("content"), "is_error": bool(get("is_error", False))})
        else:
            out.append({"type": kind})
    return out


def _retryable(e: anthropic.APIError) -> bool:
    if isinstance(e, anthropic.APIConnectionError):  # includes timeouts
        return True
    status = getattr(e, "status_code", None)
    return status is not None and (status in (408, 409, 429) or status >= 500)


def _delay(e: anthropic.APIError, attempt: int, cfg: CallConfig) -> float:
    headers = getattr(getattr(e, "response", None), "headers", None)
    retry_after = headers.get("retry-after") if headers else None
    try:
        if retry_after is not None:
            return min(float(retry_after), cfg.max_delay_s)
    except ValueError:
        pass
    delay = min(cfg.max_delay_s, cfg.base_delay_s * 2 ** (attempt - 1))
    return delay * random.uniform(0.5, 1.0) if cfg.jitter else delay


def call_model(
    client: Any, trace: Trace, *, model: str, system: Any, tools: list[dict], messages: list[dict],
    max_tokens: int, config: CallConfig = CallConfig(), deadline: float | None = None,
) -> Any:
    """Returns the SDK response, or raises ModelCallError. `deadline` is a config.clock() time.

    Deadline guarantee: no attempt starts after the deadline, each attempt gets the SDK timeout
    min(timeout_s, time left), and a retry wait that would pass the deadline is skipped. The SDK call is
    synchronous and cannot be interrupted from here, so a call in flight can overrun by the SDK's own
    timeout handling (connect/read timeouts apply per phase). Do not promise a hard wall-clock cap.
    """
    # Our retry policy is the only one: turn the SDK's own retries off so they don't multiply ours.
    api = client.with_options(max_retries=0) if hasattr(client, "with_options") else client
    limit = config.clock() + config.overall_timeout_s
    if deadline is not None:
        limit = min(limit, deadline)
    request = {
        "model": model, "system": system, "max_tokens": max_tokens, "tools": [t["name"] for t in tools],
        "messages": [{"role": m["role"], "content": _summarize(m["content"])} for m in messages],
    }
    attempt = 0
    while True:
        remaining = limit - config.clock()
        if remaining <= 0:
            trace.add(Span("model", model, request, None, 0.0, is_error=True, attempt=attempt + 1, error="DeadlineExceeded"))
            raise ModelCallError("DeadlineExceeded", attempt)
        attempt += 1
        started = time.perf_counter()
        try:
            response = api.messages.create(
                model=model, max_tokens=max_tokens, system=system, tools=tools, messages=messages,
                timeout=min(config.timeout_s, remaining), **config.params,
            )
        except anthropic.APIError as e:
            status = getattr(e, "status_code", None)
            trace.add(Span(
                "model", model, request, None, time.perf_counter() - started, cost_usd=None, is_error=True, attempt=attempt,  # no usage reported: unknown
                error=type(e).__name__ + (f" {status}" if status else ""),
            ))
            if not _retryable(e) or attempt >= config.max_attempts:
                raise ModelCallError(type(e).__name__, attempt, status) from None
            delay = _delay(e, attempt, config)
            if config.clock() + delay >= limit:
                raise ModelCallError(type(e).__name__, attempt, status) from None
            config.sleep(delay)
            continue
        in_tok = getattr(response.usage, "input_tokens", None)
        out_tok = getattr(response.usage, "output_tokens", None)
        cost = trace.cost_usd(model, in_tok, out_tok) if in_tok is not None and out_tok is not None else None  # no usage: unknown, not zero
        trace.add(Span(
            "model", model, request,
            {"stop_reason": response.stop_reason, "text": response_text(response),
             "tool_calls": [b.name for b in response.content if b.type == "tool_use"]},
            time.perf_counter() - started, in_tok or 0, out_tok or 0, cost, attempt=attempt,
        ))
        return response
