"""One /chat turn: emergency check, pending confirmation, router, specialist agent.

Order matters. The emergency check runs first on every message, even when a write is waiting for a yes.
A pending write lives for exactly one patient message: that message either confirms it (and code runs the
stored call through the normal tool pipeline) or clears it.
"""

import re
import time
import uuid
from dataclasses import asdict, dataclass, replace
from typing import Any, Callable

from agentcore.hooks import Block, Hooks
from agentcore.loop import execute_tool_call, run_agent
from agentcore.session import SessionStore
from agentcore.trace import Trace

from . import replies
from .agents import AGENT_TOOLS, system_prompt
from .confirm import is_confirmation
from .context import PatientContext
from .filters import grounded_numbers, no_medical_advice
from .emergency import SAFETY_REPLY, UNVERIFIED_REPLY, check_emergency, has_soft_symptom
from .hooks import write_policy
from .hospital import Hospital, HospitalError
from .router import route_message
from .schemas import ChatRequest, ChatResponse
from .settings import LIGHT, SPECIALIST, SPECIALIST_MAX_SECONDS, SPECIALIST_MODEL
from .tools import build_registry

MAX_HISTORY = 30
MAX_TURN_COST_USD = 0.25  # a runaway turn is stopped here (a normal one costs a fraction of a cent)
_PHONE = re.compile(r"\b[6-9]\d{5}(\d{4})\b")


class SessionConflict(Exception):
    """The session id already belongs to another phone number."""


@dataclass
class Deps:
    client: Any
    hospital: Hospital
    store: SessionStore
    sleep: Callable[[float], None] = time.sleep


@dataclass
class Reply:
    text: str
    agent: str
    handoff: bool = False
    emergency: bool = False


def mask_phones(text: str) -> str:
    return _PHONE.sub(r"******\1", text)


def handle_chat(req: ChatRequest, deps: Deps) -> ChatResponse:
    if not deps.store.bind(req.session_id, req.phone):  # first phone to use a session id keeps it
        raise SessionConflict()
    with deps.store.lock(req.session_id):  # one message at a time per conversation
        session = deps.store.load(req.session_id) or {
            "phone": req.phone, "history": [], "pending": None, "active_agent": None, "failures": 0,
        }
        trace = Trace(redact=mask_phones)
        reply = _turn(req, session, trace, deps)
        history = session["history"] + [
            {"role": "user", "content": req.message}, {"role": "assistant", "content": reply.text},
        ]
        history = history[-MAX_HISTORY:]
        session["history"] = history[1:] if history[0]["role"] != "user" else history
        deps.store.save(req.session_id, session)
        deps.store.add_trace(req.session_id, {
            "trace_id": trace.id, "agent": reply.agent, "handoff": reply.handoff, "emergency": reply.emergency,
            "spans": [asdict(s) for s in trace.spans], "total_tokens": list(trace.total_tokens),
            "total_cost_usd": trace.total_cost_usd, "cost_complete": trace.cost_complete,
        })
    return ChatResponse(reply=reply.text, agent=reply.agent, handoff=reply.handoff, emergency=reply.emergency, trace_id=trace.id)


def _turn(req: ChatRequest, session: dict, trace: Trace, deps: Deps) -> Reply:
    lang = replies.detect_language(req.message)
    light = replace(LIGHT, sleep=deps.sleep)
    ctx = PatientContext(phone=req.phone, hospital=deps.hospital, session_key=req.session_id, sleep=deps.sleep)

    verdict = check_emergency(deps.client, trace, req.message, config=light)
    if verdict == "unverified" and has_soft_symptom(req.message):
        verdict = "emergency"
    if verdict != "clear":  # emergency or unverified: stop everything, nothing waits for a yes
        session["pending"] = None
        if verdict == "emergency":
            return Reply(SAFETY_REPLY, "safety", handoff=True, emergency=True)
        return Reply(UNVERIFIED_REPLY, "safety", handoff=True)

    pending, session["pending"] = session.get("pending"), None  # consumed or cleared by this message
    if pending and is_confirmation(deps.client, trace, pending["details"], req.message, config=light):
        return _run_confirmed(pending, req, session, ctx, trace, deps, lang)

    agent = route_message(deps.client, trace, session["history"], req.message, config=light)
    if agent is None:
        return _failure(session, "info", lang)
    messages = session["history"] + [{"role": "user", "content": req.message}]
    if agent == "human":
        return _human(messages, session, ctx, trace, deps, lang)
    return _run_specialist(agent, messages, session, ctx, trace, deps, lang)


def _hooks(trace: Trace, agent: str) -> Hooks:
    def budget(request, ctx):
        return Block("Sorry, this is taking too long. Please try again.") if trace.total_cost_usd > MAX_TURN_COST_USD else None

    after_model = [no_medical_advice] + ([grounded_numbers] if agent == "info" else [])
    return Hooks(before_tool=[write_policy], before_model=[budget], after_model=after_model)


def _now(deps: Deps):
    try:
        return deps.hospital.now()
    except HospitalError:
        return None


def _run_specialist(agent, messages, session, ctx, trace, deps, lang, lead: str | None = None, failed: bool = False) -> Reply:
    result = run_agent(
        client=deps.client, model=SPECIALIST_MODEL, system=system_prompt(agent, _now(deps)),
        registry=build_registry(AGENT_TOOLS[agent]), messages=messages, ctx=ctx, trace=trace, hooks=_hooks(trace, agent),
        config=replace(SPECIALIST, sleep=deps.sleep), max_seconds=SPECIALIST_MAX_SECONDS, max_tokens=700,
    )
    text = result.text.strip()
    if result.status not in ("done", "blocked") or not text:
        return _failure(session, agent, lang)
    session["failures"], session["active_agent"] = 0, agent
    if failed and replies.claims_success(text):  # the write failed: prose must not say otherwise
        text = replies.AFTER_FAILURE
    if result.pending:
        p = result.pending
        session["pending"] = {"id": uuid.uuid4().hex, "name": p.name, "input": p.input, "fingerprint": p.fingerprint,
                              "details": p.details, "agent": agent}
        text += f"\n\n{replies.DETAILS[lang]}: {p.details}\n{replies.CONFIRM[lang]}"
    if lead:
        text = f"{lead}\n\n{text}"
    return Reply(text, agent, handoff=ctx.flags["handoff"])


def _run_confirmed(pending, req, session, ctx, trace, deps, lang) -> Reply:
    """The patient clearly confirmed: run the stored call through the same pipeline as any model-requested call."""
    ctx.approved.add(pending["fingerprint"])
    ctx.operation_id = pending["id"]  # becomes the booking's idempotency key
    result = execute_tool_call(
        build_registry(AGENT_TOOLS[pending["agent"]]), pending["name"], pending["input"], ctx, trace, _hooks(trace, pending["agent"]),
    )
    outcome = ctx.outcomes[-1] if ctx.outcomes else None
    ctx.approved.clear()  # approval is for this one call only
    ctx.operation_id = None
    if outcome and outcome["outcome"] in ("success", "unknown"):
        session["failures"], session["active_agent"] = 0, pending["agent"]
        return Reply(replies.outcome_text(outcome, lang), pending["agent"], handoff=ctx.flags["handoff"])
    # Definitely not done (the hospital said no, or a check blocked it): the agent continues with the real result.
    tool_id = "toolu_" + uuid.uuid4().hex
    messages = session["history"] + [
        {"role": "user", "content": req.message},
        {"role": "assistant", "content": [{"type": "tool_use", "id": tool_id, "name": pending["name"], "input": pending["input"]}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": result.content, "is_error": result.is_error}]},
    ]
    lead = replies.failure_line(outcome, lang) if outcome else None
    return _run_specialist(pending["agent"], messages, session, ctx, trace, deps, lang, lead=lead, failed=True)


def _human(messages, session, ctx, trace, deps, lang) -> Reply:
    result = run_agent(
        client=deps.client, model=SPECIALIST_MODEL, system=system_prompt("human", None), registry=build_registry([]),
        messages=messages, ctx=ctx, trace=trace, config=replace(SPECIALIST, sleep=deps.sleep), max_tokens=150,
    )
    text = result.text.strip() if result.status == "done" and result.text.strip() else replies.HUMAN_FALLBACK[lang]
    session["failures"] = 0
    return Reply(text, "human", handoff=True)


def _failure(session: dict, agent: str, lang: str) -> Reply:
    """Something failed (model outage, step limit, timeout). The second failure in a row goes to staff."""
    session["failures"] = session.get("failures", 0) + 1
    if session["failures"] >= 2:
        return Reply(replies.HANDOFF[lang], "human", handoff=True)
    return Reply(replies.RETRY[lang], agent)
