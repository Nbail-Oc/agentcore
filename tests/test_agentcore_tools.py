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


def test_unserializable_result_becomes_a_generic_error():
    reg, _ = _registry(lambda args, ctx: object())

    result = reg.execute("add", {"a": 1, "b": 2}, ctx=None)

    assert result.is_error
    assert "object at" not in result.content


def test_unexpected_exception_message_is_not_logged(caplog):
    def fn(args, ctx):
        raise RuntimeError("db password is hunter2")

    reg, _ = _registry(fn)

    reg.execute("add", {"a": 1, "b": 2}, ctx=None)

    assert "hunter2" not in caplog.text
    assert "RuntimeError" in caplog.text


def test_duplicate_name_is_rejected_and_keeps_the_first_tool():
    reg, calls = _registry()

    try:
        reg.register("add", "Other.", AddIn, lambda args, ctx: "other")
    except ValueError as e:
        assert "add" in str(e)
    else:
        raise AssertionError("duplicate registration was accepted")

    assert reg.execute("add", {"a": 1, "b": 2}).content == "3"
    assert len(reg.definitions()) == 1


def test_non_object_input_never_reaches_the_handler():
    reg, calls = _registry()

    for raw in (None, "a=1", [1, 2], 5):
        assert reg.execute("add", raw).is_error
    assert calls == []


def test_validation_error_does_not_echo_the_rejected_value():
    reg, calls = _registry()

    result = reg.execute("add", {"a": "9876543210-secret", "b": 1})

    assert result.is_error and "9876543210-secret" not in result.content
    assert calls == []
