"""Per-request context handed to every hospital tool and hook. Built only from the verified /chat request."""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .hospital import Hospital, HospitalError


@dataclass
class PatientContext:
    phone: str  # the verified WhatsApp number: the only identity tools and hooks ever use
    hospital: Hospital
    session_key: str
    message: str = ""  # the patient's latest message, only for checks that compare it with what the model asked for
    approved: set[str] = field(default_factory=set)  # fingerprints of calls the patient confirmed (set by code only)
    trace: Any = None  # where failure evidence goes (the turn's Trace)
    write_started: bool = False  # a write handler began: a request may have been sent, whatever happens next
    operation_id: str | None = None  # id of the confirmed action being run; becomes the booking idempotency key
    read_only: bool = False  # writes are refused this turn
    unresolved: list[dict] = field(default_factory=list)  # writes whose outcome is unknown, kept in the session
    outcomes: list[dict] = field(default_factory=list)  # structured results of writes run this turn
    flags: dict[str, Any] = field(default_factory=lambda: {"handoff": False})
    sleep: Callable[[float], None] = time.sleep
    _doctors: list[dict] | None = None

    def doctors(self) -> list[dict]:
        if self._doctors is None:
            self._doctors = self.hospital.doctors()["doctors"]
        return self._doctors

    def doctor(self, doctor_id: str) -> dict:
        try:
            for d in self.doctors():
                if d["id"] == doctor_id:
                    return d
        except HospitalError:
            pass
        return {"id": doctor_id, "name": doctor_id, "specialty": ""}
