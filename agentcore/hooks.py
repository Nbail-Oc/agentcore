"""Hooks: checkpoints around model and tool calls. Policy lives in the hooks, not in agentcore.

A hook returns None or Allow() to let the call go ahead, Block(reason) to stop it, or (tool calls
only) Confirm(details) to park it until the end user says yes. Anything else counts as Block (fail closed). agentcore never knows what a call means.

  before_tool(call, ctx)          -> decision    runs after input validation, before the tool
  after_tool(call, result, ctx)   -> ToolResult or None (None keeps the result)
  before_model(request, ctx)      -> decision    request = model, system, messages, tools (names), max_tokens
  after_model(response, ctx)      -> decision    Block replaces the reply: its reason is the text to show

For model hooks the run ends with status "blocked" and text == the reason. Hooks run in order; the
first decision that is not Allow wins.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

log = logging.getLogger(__name__)


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
    """A tool call parked by Confirm. Run it later through execute_tool_call once the user agrees.

    The fingerprint only says "same tool, same arguments". It is not authentication and not approval.
    The application must: keep the PendingCall with the verified user (not in anything the user controls),
    show the user `details` built from the call's own arguments, accept only that user's clear yes, grant
    approval for this fingerprint only (so changed arguments need a new Confirm), and discard the
    PendingCall once it has run or been refused (so it cannot be replayed).
    """

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
    """A hook that crashes blocks the call (fail closed). Only the exception type is logged."""
    for hook in hooks:
        try:
            decision = hook(*args)
        except Exception as e:
            log.error("hook %s crashed: %s", getattr(hook, "__name__", "hook"), type(e).__name__)
            return Block("A safety check failed, so this was not done.")
        if isinstance(decision, (Block, Confirm)):
            return decision
        if decision is not None and not isinstance(decision, Allow):  # a typo like "deny" must not mean "allow"
            log.error("hook %s returned an unsupported decision: %s", getattr(hook, "__name__", "hook"), type(decision).__name__)
            return Block("A safety check failed, so this was not done.")
    return Allow()
