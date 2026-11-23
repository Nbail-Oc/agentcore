"""Tests for the generic tool registry in agentcore. No hospital code, no API key."""

from pydantic import BaseModel

from agentcore.tools import ToolError, ToolRegistry


class AddIn(BaseModel):
    a: int
    b: int


def _registry(fn=None):
    calls = []

    def add(args: AddIn, ctx):
        calls.append(args)
        return args.a + args.b

    reg = ToolRegistry()
    reg.register("add", "Add two numbers.", AddIn, fn or add)
    return reg, calls


def test_definition_has_name_description_and_schema():
    reg, _ = _registry()

    [definition] = reg.definitions()

    assert definition["name"] == "add"
    assert definition["description"] == "Add two numbers."
    schema = definition["input_schema"]
    assert set(schema["properties"]) == {"a", "b"}
    assert schema["required"] == ["a", "b"]
    assert schema["additionalProperties"] is False


def test_valid_input_runs_the_tool():
    reg, calls = _registry()

    result = reg.execute("add", {"a": 2, "b": 3}, ctx=None)

    assert (result.content, result.is_error) == ("5", False)
    assert len(calls) == 1


def test_invalid_input_returns_error_and_does_not_run_the_tool():
    reg, calls = _registry()

    result = reg.execute("add", {"a": "x", "b": 1}, ctx=None)

    assert result.is_error
    assert "a" in result.content
    assert calls == []


def test_unknown_field_is_rejected_and_tool_not_run():
    reg, calls = _registry()

    result = reg.execute("add", {"a": 1, "b": 2, "phone": "9876543210"}, ctx=None)

    assert result.is_error
    assert "phone" in result.content
    assert calls == []


def test_unknown_tool_returns_error_result():
    reg, _ = _registry()

    result = reg.execute("nope", {}, ctx=None)

    assert result.is_error
    assert "nope" in result.content


def test_tool_error_message_reaches_the_model():
    def fn(args, ctx):
        raise ToolError("slot is taken")

    reg, _ = _registry(fn)

    result = reg.execute("add", {"a": 1, "b": 2}, ctx=None)

    assert (result.content, result.is_error) == ("slot is taken", True)


def test_unexpected_exception_does_not_leak_internals():
    def fn(args, ctx):
        raise RuntimeError("db password is hunter2")

    reg, _ = _registry(fn)

    result = reg.execute("add", {"a": 1, "b": 2}, ctx=None)

    assert result.is_error
    assert "hunter2" not in result.content
