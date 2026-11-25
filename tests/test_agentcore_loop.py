"""Tests for the generic agent loop, driven by FakeClaude. No hospital code, no API key."""

from pydantic import BaseModel

from agentcore.loop import run_agent
from agentcore.tools import ToolRegistry
from tests.fake_claude import FakeClaude, _message, text, tool_use, tool_uses


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


def test_crashing_tool_gives_the_model_a_generic_error_and_the_loop_continues():
    def boom(args, ctx):
        raise RuntimeError("patient 9876543210 password hunter2")

    reg = ToolRegistry()
    reg.register("add", "Add two numbers.", AddIn, boom)
    fake = FakeClaude([tool_use("add", {"a": 1, "b": 2}), text("Sorry, that failed.")])

    result = run_agent(client=fake, model="m", system="s", registry=reg, messages=[{"role": "user", "content": "hi"}])

    [tool_result] = fake.messages.calls[1]["messages"][-1]["content"]
    assert tool_result["is_error"] is True
    assert "hunter2" not in tool_result["content"] and "9876543210" not in tool_result["content"]
    assert (result.status, result.text) == ("done", "Sorry, that failed.")


def test_every_sdk_stop_reason_is_handled_and_none_is_unexpected():
    import typing
    from anthropic.types import StopReason

    for reason in typing.get_args(StopReason):
        if reason == "tool_use":
            continue  # continues the loop; covered by the tool tests
        script = [text("x", stop_reason=reason)] + ([text("done")] if reason == "pause_turn" else [])
        assert _run(FakeClaude(script)).status != "unexpected_stop", reason


def test_endless_pause_turn_ends_at_the_step_limit():
    fake = FakeClaude([text("working", stop_reason="pause_turn") for _ in range(3)])

    result = _run(fake, max_steps=3)

    assert result.status == "step_limit" and len(fake.messages.calls) == 3


def test_tool_results_pair_by_id_in_order_for_valid_unknown_and_invalid_calls():
    fake = FakeClaude([
        tool_uses([("add", {"a": 1, "b": 2}), ("nope", {}), ("add", {"a": "x", "b": 1}), ("add", {"a": 3, "b": 4})]),
        text("ok"),
    ])

    result = _run(fake)

    asked = [b.id for b in result.messages[0]["content"] if b.type == "tool_use"]
    sent = fake.messages.calls[1]["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]  # one user message holds all results
    results = sent[-1]["content"]
    assert [r["tool_use_id"] for r in results] == asked
    assert [r["is_error"] for r in results] == [False, True, True, False]
    assert results[1]["content"] == "Unknown tool: nope" and results[2]["content"].startswith("Invalid input")
    assert [results[0]["content"], results[3]["content"]] == ["3", "7"]


class _Clock:
    def __init__(self):
        self.now = 0.0

    def cfg(self):
        from agentcore.model import CallConfig
        return CallConfig(sleep=lambda s: None, clock=lambda: self.now, jitter=False)


class _SlowClient:
    """A model whose answer arrives `seconds` later on the fake clock."""

    def __init__(self, fake, clock, seconds):
        self.messages, self._clock, self._seconds = self, clock, seconds
        self._fake = fake

    def create(self, **kw):
        self._clock.now += self._seconds
        return self._fake.messages.create(**kw)


def test_no_tool_starts_when_the_model_answer_arrives_after_the_deadline():
    ran, clock = [], _Clock()
    reg = ToolRegistry()
    reg.register("add", "Add.", AddIn, lambda a, c: ran.append(a) or 1)
    client = _SlowClient(FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("x")]), clock, 10)

    result = run_agent(client=client, model="m", system="s", registry=reg, messages=[{"role": "user", "content": "hi"}],
                       max_seconds=5, config=clock.cfg())

    assert result.status == "timeout" and ran == []


def test_remaining_tool_calls_are_skipped_once_the_deadline_passes():
    ran, clock = [], _Clock()

    def slow(args, ctx):
        ran.append(args.a)
        clock.now += 10
        return "ok"

    reg = ToolRegistry()
    reg.register("add", "Add.", AddIn, slow)
    fake = FakeClaude([tool_uses([("add", {"a": 1, "b": 0}), ("add", {"a": 2, "b": 0})]), text("x")])

    result = run_agent(client=fake, model="m", system="s", registry=reg, messages=[{"role": "user", "content": "hi"}],
                       max_seconds=5, config=clock.cfg())

    assert result.status == "timeout" and ran == [1]
    results = result.messages[-1]["content"]  # every tool_use still has its result
    assert len(results) == 2 and results[1]["is_error"] is True


def test_tool_use_stop_without_a_tool_call_is_not_sent_back_as_an_empty_result():
    fake = FakeClaude([_message([], "tool_use"), text("never")])

    result = _run(fake)

    assert result.status == "unexpected_stop" and len(fake.messages.calls) == 1


def test_a_timeout_keeps_the_tool_call_that_was_already_parked():
    from agentcore.hooks import Confirm, Hooks

    clock = _Clock()
    advance = lambda call, ctx: setattr(clock, "now", clock.now + 10)  # time passes while the call is parked
    fake = FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("x")])

    result = run_agent(client=fake, model="m", system="s", registry=_registry(), messages=[{"role": "user", "content": "hi"}],
                       max_seconds=5, config=clock.cfg(), hooks=Hooks(before_tool=[advance, lambda c, x: Confirm("pay?")]))

    assert result.status == "timeout" and result.pending is not None


def test_the_result_says_how_many_model_calls_it_used_so_a_resume_can_keep_the_limit():
    fake = FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("done")])

    assert _run(fake).steps == 2


def test_a_model_hook_cannot_edit_the_messages_the_model_will_get():
    from agentcore.hooks import Hooks

    def edit(request, ctx):
        request["messages"][0]["content"] = "EDITED"

    fake, original = FakeClaude([text("x")]), [{"role": "user", "content": "hi"}]

    run_agent(client=fake, model="m", system="s", registry=_registry(), messages=original, hooks=Hooks(before_model=[edit]))

    assert fake.messages.calls[0]["messages"][0]["content"] == "hi" and original[0]["content"] == "hi"


def test_an_after_model_hook_cannot_edit_the_tool_call_that_will_run():
    from agentcore.hooks import Hooks

    seen = []

    def edit(response, ctx):
        response.content[0].input["a"] = 999

    reg = ToolRegistry()
    reg.register("add", "Add.", AddIn, lambda a, c: seen.append(a.a) or 1)
    fake = FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("x")])

    run_agent(client=fake, model="m", system="s", registry=reg, messages=[{"role": "user", "content": "hi"}],
              hooks=Hooks(after_model=[edit]))

    assert seen == [1]


def test_a_finished_text_answer_that_arrives_after_the_deadline_is_still_returned():
    clock = _Clock()
    client = _SlowClient(FakeClaude([text("all done")]), clock, 10)

    result = run_agent(client=client, model="m", system="s", registry=_registry(), messages=[{"role": "user", "content": "hi"}],
                       max_seconds=5, config=clock.cfg())

    assert (result.status, result.text) == ("done", "all done")  # complete work is kept; only new work is refused
