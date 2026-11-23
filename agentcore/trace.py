"""Trace of one run: a span for every model call attempt and tool call, with timing, tokens and cost."""

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

# USD per million tokens (input, output). Configuration: pass your own to Trace().
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}

MAX_CHARS = 2000
MAX_DEPTH = 6
_API_KEY = re.compile(r"sk-ant-[A-Za-z0-9_\-]+")


def clip(value: str) -> str:
    if len(value) <= MAX_CHARS:
        return value
    return value[:MAX_CHARS] + f"...[{len(value) - MAX_CHARS} more chars]"


@dataclass
class Span:
    kind: str  # "model" or "tool"
    name: str
    input: Any
    output: Any
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0  # None = model has no known price, so the cost is unknown (not zero)
    is_error: bool = False
    attempt: int = 1  # model calls: which try this was
    error: str | None = None  # exception type and status only, never the exception text


@dataclass
class Trace:
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    spans: list[Span] = field(default_factory=list)
    # Applied to every string stored in a span. The app uses it to mask patient data such as phone numbers.
    redact: Callable[[str], str] | None = None

    def cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float | None:
        # shortcut: cache-read/write tokens are not priced separately.
        if model not in self.prices:
            return None
        price_in, price_out = self.prices[model]
        return (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    def _clean(self, value: Any, depth: int = 0) -> Any:
        """What a span may keep: API keys scrubbed, redact hook applied, strings clipped, objects as type names."""
        if isinstance(value, str):
            value = _API_KEY.sub("[REDACTED]", value)
            return clip(self.redact(value) if self.redact else value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if depth >= MAX_DEPTH:
            return "[too deep]"
        if isinstance(value, dict):
            return {str(k): self._clean(v, depth + 1) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._clean(v, depth + 1) for v in value]
        return f"<{type(value).__name__}>"  # never repr(): it can carry secrets

    def add(self, span: Span) -> Span:
        span.input = self._clean(span.input)
        span.output = self._clean(span.output)
        self.spans.append(span)
        return span

    @property
    def total_tokens(self) -> tuple[int, int]:
        return sum(s.input_tokens for s in self.spans), sum(s.output_tokens for s in self.spans)

    @property
    def total_cost_usd(self) -> float:
        """Sum of the known costs. Check cost_complete before quoting it as the total."""
        return sum(s.cost_usd or 0.0 for s in self.spans)

    @property
    def cost_complete(self) -> bool:
        return all(s.cost_usd is not None for s in self.spans)
