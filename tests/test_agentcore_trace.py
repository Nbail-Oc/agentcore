"""Tests for tracing in agentcore: every model call and tool call leaves a span."""

import pytest
from pydantic import BaseModel

from agentcore.loop import execute_tool_call, run_agent
from agentcore.tools import ToolRegistry
from agentcore.trace import Trace
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


def test_unknown_model_costs_zero_instead_of_failing():
    result = run_agent(client=FakeClaude([text("hi")]), model="mystery", system="s", registry=_registry(),
                       messages=[{"role": "user", "content": "hi"}])

    assert result.trace.spans[0].cost_usd == 0.0


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
