"""Hooks: checkpoints around model and tool calls. Policy lives in the hooks, not in agentcore.

A hook returns None or Allow() to let the call go ahead, Block(reason) to stop it, or (tool calls
only) Confirm(details) to park it until the end user says yes. agentcore never knows what a call means.

  before_tool(call, ctx)          -> decision    runs after input validation, before the tool
  after_tool(call, result, ctx)   -> ToolResult or None (None keeps the result)
  before_model(request, ctx)      -> decision    request = model, system, messages, tools (names), max_tokens
  after_model(response, ctx)      -> decision    Block replaces the reply: its reason is the text to show

For model hooks the run ends with status "blocked" and text == the reason. Hooks run in order; the
first decision that is not Allow wins.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel


@dataclass(frozen=True)
class Allow:
    pass


@dataclass(frozen=True)
class Block:
    reason: str  # shown to the model (tools) or to the user (model hooks)


@dataclass(frozen=True)
class Confirm:
    details: str  # what the end user must confirm; built by the hook from the call's own arguments


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: BaseModel  # validated input

    @property
    def input(self) -> dict:
        return self.args.model_dump(mode="json")

    @property
    def fingerprint(self) -> str:
        """Same tool and same arguments give the same value, whatever the key order."""
        canonical = json.dumps([self.name, self.input], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class PendingCall:
    """A tool call parked by Confirm. Run it later through execute_tool_call once the user agrees."""

    name: str
    input: dict
    fingerprint: str
    details: str


@dataclass
class Hooks:
    before_tool: list[Callable[[ToolCall, Any], Any]] = field(default_factory=list)
    after_tool: list[Callable[[ToolCall, Any, Any], Any]] = field(default_factory=list)
    before_model: list[Callable[[dict, Any], Any]] = field(default_factory=list)
    after_model: list[Callable[[Any, Any], Any]] = field(default_factory=list)


def first_decision(hooks: list[Callable], *args: Any) -> Allow | Block | Confirm:
    for hook in hooks:
        decision = hook(*args)
        if isinstance(decision, (Block, Confirm)):
            return decision
    return Allow()
