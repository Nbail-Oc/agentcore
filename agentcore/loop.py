"""The agent loop: ask the model, run the tools it asks for, send results back, repeat."""

from dataclasses import dataclass
from typing import Any

from .tools import ToolRegistry


@dataclass
class AgentResult:
    text: str
    messages: list[dict]  # messages added by this run, in order


def run_agent(
    *, client: Any, model: str, system: str, registry: ToolRegistry,
    messages: list[dict], ctx: Any = None, max_tokens: int = 1024,
) -> AgentResult:
    new: list[dict] = []
    # shortcut: no step limit and only end_turn/tool_use handled; both land in the next increment, before any caller exists.
    while True:
        response = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            tools=registry.definitions(), messages=messages + new,
        )
        new.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                out = registry.execute(block.name, block.input, ctx)
                results.append({
                    "type": "tool_result", "tool_use_id": block.id,
                    "content": out.content, "is_error": out.is_error,
                })
        new.append({"role": "user", "content": results})
    return AgentResult(text="".join(b.text for b in response.content if b.type == "text"), messages=new)
