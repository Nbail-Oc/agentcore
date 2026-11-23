"""Tests for the generic agent loop, driven by FakeClaude. No hospital code, no API key."""

from pydantic import BaseModel

from agentcore.loop import run_agent
from agentcore.tools import ToolRegistry
from tests.fake_claude import FakeClaude, text, tool_use, tool_uses


class AddIn(BaseModel):
    a: int
    b: int


def _registry():
    reg = ToolRegistry()
    reg.register("add", "Add two numbers.", AddIn, lambda args, ctx: args.a + args.b)
    return reg


def _run(fake, **kw):
    return run_agent(
        client=fake, model="m", system="be brief", registry=_registry(),
        messages=[{"role": "user", "content": "hi"}], **kw,
    )


def test_plain_reply_makes_one_model_call():
    fake = FakeClaude([text("Hello! How can I help?")])

    result = _run(fake)

    assert result.text == "Hello! How can I help?"
    [call] = fake.messages.calls
    assert call["system"] == "be brief"
    assert [t["name"] for t in call["tools"]] == ["add"]


def test_tool_result_goes_back_to_the_model_then_final_reply():
    fake = FakeClaude([tool_use("add", {"a": 2, "b": 3}), text("It is 5.")])

    result = _run(fake)

    assert result.text == "It is 5."
    tool_use_id = result.messages[0]["content"][0].id
    [tool_result] = fake.messages.calls[1]["messages"][-1]["content"]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == tool_use_id
    assert tool_result["content"] == "5"


def test_parallel_tool_calls_all_get_results_in_order():
    fake = FakeClaude([
        tool_uses([("add", {"a": 1, "b": 1}), ("add", {"a": 2, "b": 2})]),
        text("2 and 4."),
    ])

    result = _run(fake)

    ids = [b.id for b in result.messages[0]["content"]]
    results = fake.messages.calls[1]["messages"][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ids
    assert [r["content"] for r in results] == ["2", "4"]


def test_tool_error_is_sent_to_the_model_as_an_error_result():
    fake = FakeClaude([tool_use("add", {"a": "x", "b": 1}), text("Sorry, bad input.")])

    _run(fake)

    [tool_result] = fake.messages.calls[1]["messages"][-1]["content"]
    assert tool_result["is_error"] is True
    assert tool_result["content"].startswith("Invalid input")
