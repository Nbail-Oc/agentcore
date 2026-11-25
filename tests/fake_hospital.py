"""A scripted stand-in for the hospital, and a ready PatientContext, for offline tests."""

from datetime import datetime

from agent_service.context import PatientContext
from agent_service.hospital import HospitalError

NEW = "2026-12-05T10:00:00+05:30"
OLD = "2026-12-03T17:30:00+05:30"
UNKNOWN = HospitalError("unknown", "service_unavailable", 503)
NOT_SENT = HospitalError("not_sent")
TIMEOUT = HospitalError("unknown")  # no reply at all: the request may still be running


def definite(code):
    return HospitalError("definite", code, 409, "hospital said no")


class FakeHospital:
    """Scripted hospital: each method pops the next scripted item (a value, or an error to raise)."""

    def __init__(self, **script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def _next(self, name, *args):
        self.calls.append((name, args))
        item = self.script[name].pop(0) if len(self.script[name]) > 1 else self.script[name][0]
        if isinstance(item, Exception):
            raise item
        return item

    def now(self):
        return self._next("now")

    def doctors(self, specialty=None):
        return self._next("doctors", specialty)

    def slots(self, doctor_id, date):
        return self._next("slots", doctor_id, date)

    def appointments(self, phone):
        return self._next("appointments", phone)

    def bills(self, phone):
        return self._next("bills", phone)

    def book(self, doctor_id, start, patient_name, phone, idempotency_key):
        return self._next("book", doctor_id, start, patient_name, phone, idempotency_key)

    def reschedule(self, appointment_id, new_start):
        return self._next("reschedule", appointment_id, new_start)

    def cancel(self, appointment_id):
        return self._next("cancel", appointment_id)

    def payment_link(self, bill_id):
        return self._next("payment_link", bill_id)


def appt(start=NEW, status="booked", id="A1003", doctor_id="d5"):
    return {"id": id, "doctor_id": doctor_id, "start": start, "status": status, "phone": "9988776655",
            "patient_name": "Sameer Kulkarni"}


DOCTORS = {"doctors": [{"id": "d5", "name": "Dr. Imran Khan", "specialty": "Orthopaedics"},
                      {"id": "d3", "name": "Dr. Sunita Iyer", "specialty": "General Medicine"}]}


def _ctx(hospital, op="op-1", phone="9988776655"):
    hospital.script.setdefault("doctors", [DOCTORS])
    ctx = PatientContext(phone=phone, hospital=hospital, session_key="s", operation_id=op)
    ctx.sleeps = []
    ctx.sleep = ctx.sleeps.append
    return ctx
