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


def test_step_limit_stops_the_loop_and_skips_the_last_tool_calls():
    calls = []
    reg = ToolRegistry()
    reg.register("add", "Add.", AddIn, lambda args, ctx: calls.append(args) or 1)
    fake = FakeClaude([tool_use("add", {"a": 1, "b": 1}) for _ in range(5)])

    result = run_agent(client=fake, model="m", system="s", registry=reg,
                       messages=[{"role": "user", "content": "hi"}], max_steps=3)

    assert result.status == "step_limit"
    assert len(fake.messages.calls) == 3
    assert len(calls) == 2  # the 3rd response asked for a tool, but the limit was reached


def test_normal_reply_has_status_done():
    assert _run(FakeClaude([text("ok")])).status == "done"


def test_stop_reasons_map_to_statuses():
    cases = {
        "max_tokens": "max_tokens",
        "refusal": "refusal",
        "model_context_window_exceeded": "context_exceeded",
        "stop_sequence": "done",
    }
    for stop_reason, status in cases.items():
        assert _run(FakeClaude([text("x", stop_reason=stop_reason)])).status == status, stop_reason


def test_unknown_stop_reason_is_not_treated_as_done():
    fake = FakeClaude([text("x").model_copy(update={"stop_reason": "brand_new_reason"})])

    assert _run(fake).status == "unexpected_stop"


def test_pause_turn_resumes_with_the_same_conversation():
    fake = FakeClaude([text("working...", stop_reason="pause_turn"), text("done")])

    result = _run(fake)

    assert (result.status, result.text) == ("done", "done")
    assert fake.messages.calls[1]["messages"][-1]["role"] == "assistant"
