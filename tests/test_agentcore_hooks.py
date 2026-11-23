"""Tests for hooks in agentcore: allow, block with a reason, or park a call until the user confirms."""

from pydantic import BaseModel

from agentcore.hooks import Block, Confirm, Hooks, ToolCall
from agentcore.loop import execute_tool_call, run_agent
from agentcore.tools import ToolRegistry, ToolResult
from agentcore.trace import Trace
from tests.fake_claude import FakeClaude, text, tool_use, tool_uses


class AddIn(BaseModel):
    a: int
    b: int


def _registry(ran):
    reg = ToolRegistry()

    def add(args, ctx):
        ran.append(("add", args.a))
        return args.a + args.b

    reg.register("add", "Add.", AddIn, add)
    reg.register("pay", "Pay.", AddIn, lambda args, ctx: ran.append(("pay", args.a)) or "paid")
    return reg


def _run(fake, ran, hooks, ctx=None, **kw):
    return run_agent(client=fake, model="m", system="s", registry=_registry(ran),
                     messages=[{"role": "user", "content": "hi"}], hooks=hooks, ctx=ctx, **kw)


def _tool_results(fake, call=1):
    return fake.messages.calls[call]["messages"][-1]["content"]


def test_fingerprint_ignores_key_order_and_changes_with_values():
    a = ToolCall("add", AddIn(a=1, b=2))
    b = ToolCall("add", AddIn(b=2, a=1))

    assert a.fingerprint == b.fingerprint
    assert a.fingerprint != ToolCall("add", AddIn(a=1, b=3)).fingerprint
    assert a.fingerprint != ToolCall("pay", AddIn(a=1, b=2)).fingerprint


def test_block_stops_the_tool_and_tells_the_model_why():
    ran = []
    hooks = Hooks(before_tool=[lambda call, ctx: Block("not allowed here")])
    fake = FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("sorry")])

    result = _run(fake, ran, hooks)

    assert ran == []
    [tr] = _tool_results(fake)
    assert (tr["content"], tr["is_error"]) == ("not allowed here", True)
    assert [s.is_error for s in result.trace.spans if s.kind == "tool"] == [True]


def test_hooks_see_validated_input_and_the_first_decision_wins():
    ran, seen = [], []

    def first(call, ctx):
        seen.append(call.args.a)
        return None  # no opinion

    def second(call, ctx):
        return Block("second says no")

    def third(call, ctx):
        seen.append("third ran")

    fake = FakeClaude([tool_use("add", {"a": 5, "b": 1}), text("ok")])

    _run(fake, ran, Hooks(before_tool=[first, second, third]))

    assert seen == [5]
    assert ran == []


def test_hooks_do_not_run_when_input_is_invalid():
    called = []
    hooks = Hooks(before_tool=[lambda call, ctx: called.append(1)])
    fake = FakeClaude([tool_use("add", {"a": "x", "b": 1}), text("ok")])

    _run(fake, [], hooks)

    assert called == []


def test_confirm_parks_the_call_instead_of_running_it():
    ran = []
    hooks = Hooks(before_tool=[lambda call, ctx: Confirm("Pay 7 to account X") if call.name == "pay" else None])
    fake = FakeClaude([tool_use("pay", {"a": 7, "b": 0}), text("Shall I pay 7?")])

    result = _run(fake, ran, hooks)

    assert ran == []
    assert result.status == "done"
    assert result.pending.name == "pay"
    assert result.pending.input == {"a": 7, "b": 0}
    assert result.pending.details == "Pay 7 to account X"
    assert result.pending.fingerprint == ToolCall("pay", AddIn(a=7, b=0)).fingerprint
    [tr] = _tool_results(fake)
    assert "confirm" in tr["content"].lower()


def test_after_a_confirm_the_other_parallel_calls_are_skipped_with_results():
    ran = []
    hooks = Hooks(before_tool=[lambda call, ctx: Confirm("pay") if call.name == "pay" else None])
    fake = FakeClaude([
        tool_uses([("add", {"a": 1, "b": 1}), ("pay", {"a": 2, "b": 0}), ("add", {"a": 3, "b": 3})]),
        text("ok"),
    ])

    result = _run(fake, ran, hooks)

    assert ran == [("add", 1)]
    results = _tool_results(fake)
    ids = [b.id for b in result.messages[0]["content"]]
    assert [r["tool_use_id"] for r in results] == ids
    assert results[2]["is_error"] is True and "not executed" in results[2]["content"].lower()
    assert result.pending.name == "pay"


def test_a_second_confirm_in_the_same_run_is_blocked():
    hooks = Hooks(before_tool=[lambda call, ctx: Confirm(f"pay {call.args.a}")])
    fake = FakeClaude([
        tool_use("pay", {"a": 1, "b": 0}), tool_use("pay", {"a": 2, "b": 0}), text("ok"),
    ])

    result = _run(fake, [], hooks)

    assert result.pending.input["a"] == 1
    assert _tool_results(fake, 2)[0]["is_error"] is True


def test_an_approved_call_runs_through_the_same_pipeline():
    ran, trace = [], Trace()
    call_key = ToolCall("pay", AddIn(a=7, b=0)).fingerprint

    def gate(call, ctx):
        return None if call.fingerprint in ctx["approved"] else Confirm("pay")

    out = execute_tool_call(_registry(ran), "pay", {"a": 7, "b": 0}, {"approved": {call_key}}, trace,
                            Hooks(before_tool=[gate]))
    assert (out.content, ran) == ("paid", [("pay", 7)])
    assert [s.kind for s in trace.spans] == ["tool"]

    denied = execute_tool_call(_registry(ran), "pay", {"a": 8, "b": 0}, {"approved": {call_key}}, trace,
                               Hooks(before_tool=[gate]))
    assert denied.pending is not None and ran == [("pay", 7)]


def test_after_tool_hook_can_replace_the_result():
    hooks = Hooks(after_tool=[lambda call, result, ctx: ToolResult("redacted")])
    fake = FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("ok")])

    _run(fake, [], hooks)

    assert _tool_results(fake)[0]["content"] == "redacted"


def test_before_model_block_stops_the_run_without_calling_the_model():
    fake = FakeClaude([text("never")])

    result = _run(fake, [], Hooks(before_model=[lambda request, ctx: Block("budget used up")]))

    assert (result.status, result.text) == ("blocked", "budget used up")
    assert fake.messages.calls == []


def test_before_model_hook_sees_the_request():
    seen = {}

    def hook(request, ctx):
        seen.update(request)

    _run(FakeClaude([text("ok")]), [], Hooks(before_model=[hook]))

    assert seen["model"] == "m" and seen["system"] == "s" and seen["tools"] == ["add", "pay"]


def test_after_model_block_replaces_the_reply_and_drops_its_tool_calls():
    ran = []
    fake = FakeClaude([tool_use("add", {"a": 1, "b": 1})])

    result = _run(fake, ran, Hooks(after_model=[lambda response, ctx: Block("I can't say that")]))

    assert (result.status, result.text) == ("blocked", "I can't say that")
    assert ran == []


def test_a_before_tool_hook_that_crashes_blocks_the_call():
    ran = []

    def broken(call, ctx):
        raise RuntimeError("secret internal detail")

    fake = FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("ok")])

    result = _run(fake, ran, Hooks(before_tool=[broken]))

    assert ran == []
    [tr] = _tool_results(fake)
    assert tr["is_error"] is True and "secret" not in tr["content"]
    assert "secret" not in repr(result.trace.spans)


def test_a_before_model_hook_that_crashes_stops_the_run():
    def broken(request, ctx):
        raise RuntimeError("secret")

    fake = FakeClaude([text("never")])

    result = _run(fake, [], Hooks(before_model=[broken]))

    assert result.status == "blocked" and fake.messages.calls == []


def test_an_after_tool_hook_that_crashes_keeps_the_tool_result():
    def broken(call, result, ctx):
        raise RuntimeError("secret")

    fake = FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("ok")])

    _run(fake, [], Hooks(after_tool=[broken]))

    assert _tool_results(fake)[0]["content"] == "2"
