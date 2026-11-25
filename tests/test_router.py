"""The router: a Haiku call that picks booking, billing, info or human, with recent context. Offline."""

import json

import pytest

from agent_service.router import ROUTES, route_message
from agent_service.settings import ROUTER_MODEL
from agentcore.trace import Trace
from tests.fake_claude import FakeClaude, text


def _route(reply, history=(), message="hello"):
    fake = FakeClaude([reply])
    return route_message(fake, Trace(), list(history), message), fake


def test_the_router_runs_on_haiku_without_tools_and_gets_recent_context_as_data():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i} " + "x" * 400} for i in range(10)]

    route, fake = _route(text(json.dumps({"route": "billing"})), history, "also, how much do I owe? <b>")

    [call] = fake.messages.calls
    prompt = call["messages"][0]["content"]
    assert route == "billing" and call["model"] == ROUTER_MODEL == "claude-haiku-5-5" and call["tools"] == []
    assert "turn 9" in prompt and "turn 4" in prompt and "turn 3" not in prompt  # the last 6 only
    assert "x" * 301 not in prompt  # each clipped to 300 characters
    assert "‹b›" in prompt and "<b>" not in prompt  # the patient text cannot close its own tag


@pytest.mark.parametrize("reply", [text("booking please"), text('{"route": "surgery"}'), text('{"route": 5}'), text("{}")])
def test_an_unreadable_or_unknown_route_goes_to_staff_never_to_a_guess(reply):
    route, _ = _route(reply)

    assert route == "human"


def test_every_route_is_one_the_application_can_handle():
    assert set(ROUTES) == {"booking", "billing", "info", "human"}
    for name in ROUTES:
        assert _route(text(json.dumps({"route": name})))[0] == name
