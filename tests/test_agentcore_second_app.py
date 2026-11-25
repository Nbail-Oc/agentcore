"""A second, non-hospital application on agentcore: a note-taking agent. No agent_service, no hospital state."""

from pydantic import BaseModel

from agentcore.hooks import Block, Hooks
from agentcore.loop import run_agent
from agentcore.tools import ToolRegistry
from tests.fake_claude import FakeClaude, text, tool_use


class NoteIn(BaseModel):
    title: str


def test_a_second_app_uses_loop_registry_and_its_own_hook():
    notes = []
    reg = ToolRegistry()
    reg.register("add_note", "Save a note.", NoteIn, lambda a, ctx: notes.append(a.title) or f"saved {a.title}")
    no_secrets = lambda call, ctx: Block("No passwords in notes.") if "password" in call.args.title else None
    fake = FakeClaude([
        tool_use("add_note", {"title": "buy milk"}), tool_use("add_note", {"title": "my password"}), text("Done."),
    ])

    result = run_agent(client=fake, model="m", system="notes", registry=reg, ctx={"user": "u1"},
                       messages=[{"role": "user", "content": "note things"}], hooks=Hooks(before_tool=[no_secrets]))

    assert (result.status, result.text) == ("done", "Done.")
    assert notes == ["buy milk"]  # the blocked call never reached the handler
    blocked = fake.messages.calls[2]["messages"][-1]["content"][0]
    assert blocked["is_error"] and blocked["content"] == "No passwords in notes."
