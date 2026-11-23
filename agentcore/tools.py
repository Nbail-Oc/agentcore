"""Tool registry: turns Python functions with Pydantic inputs into Claude tools.

The registry builds the tool definition, checks the model's input before running
the tool, and turns every failure into a result the model can read.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError
from pydantic_core import PydanticSerializationError, to_json

log = logging.getLogger(__name__)


class ToolError(Exception):
    """An expected failure. The message is shown to the model, so keep it safe to read."""


@dataclass(frozen=True)
class ToolResult:
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class _Tool:
    description: str
    input_model: type[BaseModel]
    fn: Callable[[Any, Any], Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, _Tool] = {}

    def register(self, name: str, description: str, input_model: type[BaseModel], fn: Callable[[Any, Any], Any]) -> None:
        """fn(args, ctx) gets the validated input model and an opaque caller context."""
        self._tools[name] = _Tool(description, input_model, fn)

    def definitions(self) -> list[dict]:
        out = []
        for name, tool in self._tools.items():
            schema = tool.input_model.model_json_schema()
            schema["additionalProperties"] = False
            out.append({"name": name, "description": tool.description, "input_schema": schema})
        return out

    def execute(self, name: str, raw_input: Any, ctx: Any = None) -> ToolResult:
        """Run one tool call. Never raises: every failure becomes an error result."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(f"Unknown tool: {name}", True)
        if not isinstance(raw_input, dict):
            return ToolResult("Tool input must be a JSON object.", True)
        extra = sorted(set(raw_input) - set(tool.input_model.model_fields))
        if extra:
            return ToolResult(f"Unexpected input fields: {', '.join(extra)}", True)
        try:
            args = tool.input_model.model_validate(raw_input)
        except ValidationError as e:
            # Field names and messages only: error["input"] could echo patient data.
            problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
            return ToolResult(f"Invalid input: {problems}", True)
        try:
            output = tool.fn(args, ctx)
        except ToolError as e:
            return ToolResult(str(e), True)
        except Exception as e:
            # Type only: exception text and tracebacks can carry patient data or secrets.
            log.error("tool %s failed unexpectedly: %s", name, type(e).__name__)
            return ToolResult("The tool failed unexpectedly.", True)
        if isinstance(output, str):
            return ToolResult(output)
        try:
            return ToolResult(to_json(output).decode())
        except PydanticSerializationError as e:
            # No fallback=str: it would put the repr of arbitrary objects in front of the model.
            log.error("tool %s returned an unserializable result: %s", name, type(e).__name__)
            return ToolResult("The tool returned a result that could not be converted to JSON.", True)
