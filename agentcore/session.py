"""Session storage behind a small interface, so Redis (or anything else) can replace it later.

Data is plain JSON-able dicts. `owner` is whatever identifies who may use a session (for example a
verified phone number); the first owner to bind a session id keeps it.
"""

import copy
import threading
from contextlib import AbstractContextManager
from typing import Protocol


class SessionStore(Protocol):
    def bind(self, session_id: str, owner: str) -> bool: ...
    def owner(self, session_id: str) -> str | None: ...
    def lock(self, session_id: str) -> AbstractContextManager: ...
    def load(self, session_id: str) -> dict | None: ...
    def save(self, session_id: str, data: dict) -> None: ...
    def add_trace(self, session_id: str, trace: dict) -> None: ...
    def traces(self, session_id: str) -> list[dict]: ...


class InMemorySessionStore:
    """Single-process only. shortcut: grows without limit and is lost on restart; use Redis with a TTL.

    Limits: everything (owners, data, traces, pending actions) is lost on restart. Locks are threading locks,
    so they serialize one process only; several workers would need a shared lock (for example in Redis).
    load/save copy the data, so callers never share references. Data is not encrypted.
    """

    def __init__(self) -> None:
        self._guard = threading.Lock()  # protects the dicts below; never held while a session lock is
        self._owners: dict[str, str] = {}
        self._data: dict[str, dict] = {}
        self._traces: dict[str, list[dict]] = {}
        self._locks: dict[str, threading.Lock] = {}

    def bind(self, session_id: str, owner: str) -> bool:
        with self._guard:
            return self._owners.setdefault(session_id, owner) == owner

    def owner(self, session_id: str) -> str | None:
        with self._guard:
            return self._owners.get(session_id)

    def lock(self, session_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(session_id, threading.Lock())

    def load(self, session_id: str) -> dict | None:
        with self._guard:
            data = self._data.get(session_id)
            return copy.deepcopy(data) if data is not None else None

    def save(self, session_id: str, data: dict) -> None:
        with self._guard:
            self._data[session_id] = copy.deepcopy(data)

    def add_trace(self, session_id: str, trace: dict) -> None:
        with self._guard:
            self._traces.setdefault(session_id, []).append(copy.deepcopy(trace))

    def traces(self, session_id: str) -> list[dict]:
        with self._guard:
            return copy.deepcopy(self._traces.get(session_id, []))
