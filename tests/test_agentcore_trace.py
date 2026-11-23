"""Tests for tracing in agentcore: every model call and tool call leaves a span."""

import pytest
from pydantic import BaseModel

from agentcore.loop import execute_tool_call, run_agent
from agentcore.tools import ToolRegistry
from agentcore.trace import Span, Trace
from tests.fake_claude import FakeClaude, text, tool_use


class AddIn(BaseModel):
    a: int
    b: int


def _registry(fn=lambda args, ctx: args.a + args.b):
    reg = ToolRegistry()
    reg.register("add", "Add.", AddIn, fn)
    return reg


def _run(fake, reg=None, **kw):
    return run_agent(client=fake, model="claude-haiku-5-5", system="s", registry=reg or _registry(),
                     messages=[{"role": "user", "content": "hi"}], **kw)


def test_each_model_and_tool_call_gets_a_span_with_tokens_and_cost():
    fake = FakeClaude([tool_use("add", {"a": 2, "b": 3}), text("5")])  # 100 in / 20 out tokens each

    spans = _run(fake).trace.spans

    assert [(s.kind, s.name) for s in spans] == [("model", "claude-haiku-5-5"), ("tool", "add"), ("model", "claude-haiku-5-5")]
    model_span = spans[0]
    assert (model_span.input_tokens, model_span.output_tokens) == (100, 20)
    assert model_span.cost_usd == pytest.approx(100 * 0.10 / 1e6 + 20 * 0.50 / 1e6)
    assert spans[1].input == {"a": 2, "b": 3}
    assert spans[1].output == "5"
    assert all(s.duration_s >= 0 for s in spans)


def test_trace_totals_sum_the_spans():
    trace = _run(FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("2")])).trace

    assert trace.total_tokens == (200, 40)
    assert trace.total_cost_usd == pytest.approx(2 * (100 * 0.10 / 1e6 + 20 * 0.50 / 1e6))


def test_unknown_model_cost_is_unknown_not_zero():
    result = run_agent(client=FakeClaude([text("hi")]), model="mystery", system="s", registry=_registry(),
                       messages=[{"role": "user", "content": "hi"}])

    assert result.trace.spans[0].cost_usd is None
    assert result.trace.cost_complete is False


def test_known_models_make_the_cost_complete():
    assert _run(FakeClaude([text("hi")])).trace.cost_complete is True


def test_model_span_input_has_the_system_prompt_and_every_message():
    fake = FakeClaude([tool_use("add", {"a": 2, "b": 3}), text("5")])

    spans = _run(fake).trace.spans

    first, second = spans[0].input, spans[2].input
    assert first["system"] == "s"
    assert first["messages"] == [{"role": "user", "content": "hi"}]
    assert [m["role"] for m in second["messages"]] == ["user", "assistant", "user"]
    assert second["messages"][1]["content"] == [{"tool_use": "add", "input": {"a": 2, "b": 3}}]
    assert second["messages"][2]["content"] == [{"tool_result": "5", "is_error": False}]


def test_nested_values_are_clipped():
    trace = Trace()

    trace.add(Span("tool", "t", {"a": {"b": ["x" * 5000]}}, None, 0.0))

    assert len(trace.spans[0].input["a"]["b"][0]) < 5_000


def test_objects_are_stored_as_a_type_name_not_a_repr():
    class Secretive:
        def __repr__(self):
            return "Secretive(token=hunter2)"

    trace = Trace()

    trace.add(Span("tool", "t", {"obj": Secretive()}, None, 0.0))

    assert trace.spans[0].input == {"obj": "<Secretive>"}


def test_api_keys_are_scrubbed_from_spans():
    trace = Trace()

    trace.add(Span("tool", "t", None, "key sk-ant-api03-abc_DEF-123 leaked", 0.0))

    assert "sk-ant" not in trace.spans[0].output


def test_caller_can_redact_sensitive_values():
    trace = Trace(redact=lambda s: s.replace("9876543210", "******3210"))

    trace.add(Span("tool", "t", {"phone": "9876543210"}, "for 9876543210", 0.0))

    assert trace.spans[0].input == {"phone": "******3210"}
    assert trace.spans[0].output == "for ******3210"


def test_a_shared_trace_collects_spans_from_several_runs():
    trace = Trace()

    _run(FakeClaude([text("a")]), trace=trace)
    _run(FakeClaude([text("b")]), trace=trace)

    assert len(trace.spans) == 2


def test_failed_tool_span_has_no_exception_details():
    def boom(args, ctx):
        raise RuntimeError("db password is hunter2")

    trace = _run(FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("sorry")]), reg=_registry(boom)).trace

    tool_span = trace.spans[1]
    assert tool_span.is_error
    assert "hunter2" not in repr(tool_span)


def test_execute_tool_call_is_the_shared_path_and_records_a_span():
    trace = Trace()

    result = execute_tool_call(_registry(), "add", {"a": 4, "b": 5}, None, trace)

    assert result.content == "9"
    assert [(s.kind, s.name) for s in trace.spans] == [("tool", "add")]


def test_long_values_are_clipped_in_spans():
    reg = _registry(lambda args, ctx: "x" * 50_000)

    trace = _run(FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("ok")]), reg=reg).trace

    assert len(trace.spans[1].output) < 5_000
