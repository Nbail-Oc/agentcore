"""The agent service. Run: uvicorn agent_service.app:app --port 8000"""

from dotenv import load_dotenv

load_dotenv()

import anthropic  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402

from agentcore.session import InMemorySessionStore  # noqa: E402

from .hospital import Hospital  # noqa: E402
from .orchestrator import Deps, SessionConflict, handle_chat  # noqa: E402
from .schemas import ChatRequest, ChatResponse  # noqa: E402

app = FastAPI(title="City Care front desk agent", version="0.2")
_deps: Deps | None = None  # built on first use; tests replace it with fakes


def get_deps() -> Deps:
    global _deps
    if _deps is None:
        _deps = Deps(client=anthropic.Anthropic(), hospital=Hospital(), store=InMemorySessionStore())
    return _deps


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:  # plain def: FastAPI runs it in a thread, so slow calls don't block others
    try:
        return handle_chat(req, get_deps())
    except SessionConflict:
        raise HTTPException(status_code=409, detail="This session id is already in use by another conversation.") from None


@app.get("/trace/{session_id}")
def trace(session_id: str):
    # shortcut: no authentication (the spec gives the endpoint only a session id). Phone numbers are masked in spans.
    traces = get_deps().store.traces(session_id)
    if not traces:
        raise HTTPException(status_code=404, detail="No trace for this session.")
    return {
        "session_id": session_id, "traces": traces,
        "total_cost_usd": sum(t["total_cost_usd"] for t in traces),
        "cost_complete": all(t["cost_complete"] for t in traces),
    }
