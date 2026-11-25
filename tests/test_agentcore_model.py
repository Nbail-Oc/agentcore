"""Tests for model calls in agentcore: timeouts, retries, backoff, failed-attempt traces. Offline."""

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

from agentcore.loop import run_agent
from agentcore.model import CallConfig, ModelCallError, call_model
from agentcore.tools import ToolRegistry
from agentcore.trace import Trace
from tests.fake_claude import FakeClaude, text, tool_use

REQUEST = httpx2.Request("POST", "https://api.test/v1/messages")


def _status_error(cls, status, headers=None):
    response = httpx2.Response(status, request=REQUEST, headers=headers)
    return cls("secret request details", response=response, body=None)


def rate_limited(headers=None):
    return _status_error(anthropic.RateLimitError, 429, headers)


class FakeTime:
    """Fake clock: sleeping moves time forward instantly."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def clock(self):
        return self.now


def _config(t, **kw):
    kw.setdefault("jitter", False)
    return CallConfig(sleep=t.sleep, clock=t.clock, **kw)


def _call(fake, t, trace=None, **kw):
    return call_model(
        fake, trace or Trace(), model="claude-haiku-5-5", system="s", tools=[],
        messages=[{"role": "user", "content": "hi"}], max_tokens=100, config=_config(t, **kw),
    )


def test_passes_the_per_attempt_timeout_to_the_sdk():
    fake, t = FakeClaude([text("ok")]), FakeTime()

    _call(fake, t, timeout_s=12.0)

    assert fake.messages.calls[0]["timeout"] == 12.0


def test_turns_off_sdk_retries_so_they_do_not_multiply_ours():
    seen = {}
    inner = FakeClaude([text("ok")])

    class SdkLikeClient:
        def with_options(self, **options):
            seen.update(options)
            return inner

    call_model(SdkLikeClient(), Trace(), model="m", system="s", tools=[],
               messages=[{"role": "user", "content": "hi"}], max_tokens=10, config=_config(FakeTime()))

    assert seen == {"max_retries": 0}
    assert len(inner.messages.calls) == 1


def test_retries_rate_limit_then_succeeds():
    fake, t = FakeClaude([rate_limited(), text("ok")]), FakeTime()

    response = _call(fake, t)

    assert response.content[0].text == "ok"
    assert len(fake.messages.calls) == 2
    assert t.sleeps == [1.0]


def test_backoff_doubles_between_attempts():
    fake = FakeClaude([rate_limited(), rate_limited(), rate_limited(), text("ok")])
    t = FakeTime()

    _call(fake, t, max_attempts=4)

    assert t.sleeps == [1.0, 2.0, 4.0]


def test_backoff_is_capped():
    fake = FakeClaude([rate_limited()] * 3 + [text("ok")])
    t = FakeTime()

    _call(fake, t, max_attempts=4, max_delay_s=3.0)

    assert t.sleeps == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("error", [
    anthropic.APIConnectionError(request=REQUEST),
    anthropic.APITimeoutError(REQUEST),
    _status_error(anthropic.InternalServerError, 500),
    _status_error(anthropic.OverloadedError, 529),
    _status_error(anthropic.RequestTimeoutError, 408),
])
def test_retries_transient_errors(error):
    fake, t = FakeClaude([error, text("ok")]), FakeTime()

    _call(fake, t)

    assert len(fake.messages.calls) == 2


@pytest.mark.parametrize("error", [
    _status_error(anthropic.BadRequestError, 400),
    _status_error(anthropic.AuthenticationError, 401),
    _status_error(anthropic.NotFoundError, 404),
])
def test_does_not_retry_client_errors(error):
    fake, t = FakeClaude([error, text("never reached")]), FakeTime()

    with pytest.raises(ModelCallError) as info:
        _call(fake, t)

    assert len(fake.messages.calls) == 1
    assert t.sleeps == []
    assert info.value.attempts == 1


def test_gives_up_after_max_attempts():
    fake, t = FakeClaude([rate_limited()] * 3), FakeTime()

    with pytest.raises(ModelCallError) as info:
        _call(fake, t, max_attempts=3)

    assert (info.value.reason, info.value.attempts, info.value.status_code) == ("RateLimitError", 3, 429)
    assert t.sleeps == [1.0, 2.0]


def test_error_text_from_the_sdk_never_reaches_our_exception():
    fake, t = FakeClaude([rate_limited()]), FakeTime()

    with pytest.raises(ModelCallError) as info:
        _call(fake, t, max_attempts=1)

    assert "secret request details" not in str(info.value)
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_retry_after_header_is_used_and_capped():
    t = FakeTime()
    _call(FakeClaude([rate_limited({"retry-after": "7"}), text("ok")]), t)
    assert t.sleeps == [7.0]

    t = FakeTime()
    _call(FakeClaude([rate_limited({"retry-after": "500"}), text("ok")]), t, max_delay_s=20.0)
    assert t.sleeps == [20.0]


def test_every_attempt_leaves_a_span_and_failures_carry_no_error_text():
    fake, t, trace = FakeClaude([rate_limited(), rate_limited(), text("ok")]), FakeTime(), Trace()

    _call(fake, t, trace)

    assert [(s.attempt, s.is_error, s.error) for s in trace.spans] == [
        (1, True, "RateLimitError 429"), (2, True, "RateLimitError 429"), (3, False, None),
    ]
    assert "secret request details" not in repr(trace.spans)
    assert trace.spans[2].input_tokens == 100


def test_overall_budget_stops_retries_that_would_not_fit():
    fake, t = FakeClaude([rate_limited()] * 5), FakeTime()

    with pytest.raises(ModelCallError) as info:
        _call(fake, t, max_attempts=5, base_delay_s=4.0, overall_timeout_s=10.0)

    # attempt 1, sleep 4, attempt 2, the next sleep (8) would pass the 10s budget
    assert info.value.attempts == 2
    assert t.sleeps == [4.0]


def test_attempt_timeout_never_exceeds_the_remaining_budget():
    fake, t = FakeClaude([text("ok")]), FakeTime()

    _call(fake, t, timeout_s=30.0, overall_timeout_s=10.0)

    assert fake.messages.calls[0]["timeout"] == 10.0


class AddIn(BaseModel):
    a: int
    b: int


def _registry(fn=lambda args, ctx: args.a + args.b):
    reg = ToolRegistry()
    reg.register("add", "Add.", AddIn, fn)
    return reg


def _run(fake, t, reg=None, **kw):
    return run_agent(client=fake, model="claude-haiku-5-5", system="s", registry=reg or _registry(),
                     messages=[{"role": "user", "content": "hi"}], config=_config(t), **kw)


def test_loop_retries_transparently():
    t = FakeTime()

    result = _run(FakeClaude([rate_limited(), text("ok")]), t)

    assert (result.status, result.text) == ("done", "ok")


def test_loop_reports_api_error_when_the_api_stays_down():
    t = FakeTime()

    result = _run(FakeClaude([rate_limited()] * 3), t)

    assert (result.status, result.text) == ("api_error", "")
    assert [s.is_error for s in result.trace.spans] == [True, True, True]


def test_loop_stops_with_timeout_when_the_run_budget_is_used_up():
    t = FakeTime()

    def slow_tool(args, ctx):
        t.now += 10
        return "1"

    fake = FakeClaude([tool_use("add", {"a": 1, "b": 1}), text("never reached")])

    result = _run(fake, t, reg=_registry(slow_tool), max_seconds=5.0)

    assert result.status == "timeout"
    assert len(fake.messages.calls) == 1


def test_extra_request_params_are_passed_to_the_sdk():
    fake, t = FakeClaude([text("ok")]), FakeTime()

    _call(fake, t, params={"thinking": {"type": "disabled"}})

    assert fake.messages.calls[0]["thinking"] == {"type": "disabled"}


def test_exact_request_that_reaches_the_client():
    class In(BaseModel):
        n: int

    reg = ToolRegistry()
    reg.register("double", "Double a number.", In, lambda a, c: a.n * 2)
    fake = FakeClaude([text("ok")])
    cfg = CallConfig(timeout_s=12.0, params={"thinking": {"type": "disabled"}})

    run_agent(client=fake, model="claude-haiku-5-5", system="be brief", registry=reg,
              messages=[{"role": "user", "content": "hi"}], max_tokens=300, config=cfg)

    [sent] = fake.messages.calls
    assert sent == {
        "model": "claude-haiku-5-5", "max_tokens": 300, "system": "be brief",
        "messages": [{"role": "user", "content": "hi"}], "timeout": 12.0, "thinking": {"type": "disabled"},
        "tools": [{"name": "double", "description": "Double a number.", "input_schema": {
            "properties": {"n": {"title": "N", "type": "integer"}}, "required": ["n"],
            "title": "In", "type": "object", "additionalProperties": False}}],
    }


def test_only_model_py_calls_messages_create():
    from pathlib import Path

    root = Path(__file__).parent.parent
    for path in [*(root / "agentcore").glob("*.py"), *(root / "agent_service").glob("*.py")]:
        if path.name != "model.py":
            assert "messages.create" not in path.read_text(encoding="utf-8"), path


def test_a_failed_attempt_has_unknown_cost_not_zero():
    fake, t, trace = FakeClaude([rate_limited(), text("ok")]), FakeTime(), Trace()

    _call(fake, t, trace=trace)

    failed, ok = trace.spans
    assert failed.cost_usd is None and ok.cost_usd is not None
    assert not trace.cost_complete
