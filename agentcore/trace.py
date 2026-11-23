"""Trace of one run: a span for every model call and tool call, with timing, tokens and cost."""

import uuid
from dataclasses import dataclass, field
from typing import Any

# USD per million tokens (input, output). Configuration: pass your own to Trace().
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
}

MAX_CHARS = 2000


def clip(value: Any) -> Any:
    """Keep spans small: long strings are cut. Other values pass through unchanged."""
    # shortcut: only top-level strings are clipped; nested structures are stored as given.
    if isinstance(value, str) and len(value) > MAX_CHARS:
        return value[:MAX_CHARS] + f"...[{len(value) - MAX_CHARS} more chars]"
    return value


@dataclass
class Span:
    kind: str  # "model" or "tool"
    name: str
    input: Any
    output: Any
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    is_error: bool = False


@dataclass
class Trace:
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(DEFAULT_PRICES))
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    spans: list[Span] = field(default_factory=list)

    def cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float:
        # shortcut: cache-read/write tokens are not priced separately; unknown models cost 0.
        price_in, price_out = self.prices.get(model, (0.0, 0.0))
        return (input_tokens * price_in + output_tokens * price_out) / 1_000_000

    def add(self, span: Span) -> Span:
        self.spans.append(span)
        return span

    @property
    def total_tokens(self) -> tuple[int, int]:
        return sum(s.input_tokens for s in self.spans), sum(s.output_tokens for s in self.spans)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)
