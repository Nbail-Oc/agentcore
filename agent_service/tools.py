"""Hospital tools. Reads are bound to the verified phone; each write has its own recovery rules.

Write outcomes are structured: success (2xx, or the hospital's own state shows the target), failed
(a definite error code), or unknown (we cannot tell). Only a definite answer or evidence of the target
state counts. "I looked and found nothing" is never treated as proof that a write did not happen.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, Callable

from pydantic import BaseModel, Field, StringConstraints

from agentcore.tools import ToolError, ToolRegistry

from .context import PatientContext
from .hospital import HospitalError

IST = timezone(timedelta(hours=5, minutes=30))
DoctorId = Annotated[str, StringConstraints(pattern=r"^d\d{1,4}$")]
AppointmentId = Annotated[str, StringConstraints(pattern=r"^A\d{1,8}$")]
BillId = Annotated[str, StringConstraints(pattern=r"^B\d{1,8}$")]
PersonName = Annotated[str, StringConstraints(pattern=r"^[A-Za-zऀ-ॿ][A-Za-zऀ-ॿ .'\-]{0,59}$")]
WRITE_TOOLS = {"book_appointment", "reschedule_appointment", "cancel_appointment", "send_payment_link"}


def parse_start(value: str) -> datetime:
    """Hospital time. A time without an offset is taken as IST."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise ToolError("Time must be ISO format with offset, like 2026-12-04T17:00:00+05:30") from None
    return (dt if dt.tzinfo else dt.replace(tzinfo=IST)).astimezone(IST)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def _read(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except HospitalError as e:
        if e.kind == "definite":
            raise ToolError(f"{e.code or 'error'}: {e.message}".strip(": ")) from None
        raise ToolError("The hospital system is not responding. Try again shortly.") from None


def _compact(a: dict, ctx: PatientContext) -> dict:
    return {"id": a["id"], "doctor_id": a["doctor_id"], "doctor": ctx.doctor(a["doctor_id"])["name"],
            "start": a["start"], "status": a["status"]}


# ---------------------------------------------------------------- reads (no phone argument: ctx.phone only)

class FindDoctorsIn(BaseModel):
    specialty: str | None = Field(None, max_length=40)


class FreeSlotsIn(BaseModel):
    doctor_id: DoctorId
    date: date


class NoInput(BaseModel):
    pass


def find_doctors(args: FindDoctorsIn, ctx: PatientContext):
    return _read(lambda: ctx.hospital.doctors(args.specialty))["doctors"]


def free_slots(args: FreeSlotsIn, ctx: PatientContext):
    return _read(lambda: ctx.hospital.slots(args.doctor_id, args.date.isoformat()))


def my_appointments(args: NoInput, ctx: PatientContext):
    return [_compact(a, ctx) for a in _read(lambda: ctx.hospital.appointments(ctx.phone))]


def my_bills(args: NoInput, ctx: PatientContext):
    return _read(lambda: ctx.hospital.bills(ctx.phone))


# ---------------------------------------------------------------- writes

class BookIn(BaseModel):
    doctor_id: DoctorId
    start: str = Field(max_length=40)
    patient_name: PersonName


class RescheduleIn(BaseModel):
    appointment_id: AppointmentId
    new_start: str = Field(max_length=40)


class CancelIn(BaseModel):
    appointment_id: AppointmentId


class SendLinkIn(BaseModel):
    bill_id: BillId


def _lookup(ctx: PatientContext) -> list[dict] | None:
    try:
        return ctx.hospital.appointments(ctx.phone)
    except HospitalError:
        return None


def _record(ctx: PatientContext, tool: str, outcome: str, **extra: Any) -> dict:
    result = {"tool": tool, "outcome": outcome, **extra}
    if outcome == "unknown":
        ctx.flags["handoff"] = True
        result["instruction"] = "Do NOT tell the patient it happened. Say it could not be confirmed and staff will check."
    ctx.outcomes.append(result)
    return result


def _failed(ctx: PatientContext, tool: str, e: HospitalError) -> dict:
    return _record(ctx, tool, "failed", code=e.code or "unreachable", message=e.message)


def _write_with_checks(do: Callable[[], dict], verify: Callable[[], dict | bool | None], ctx: PatientContext):
    """For writes with no idempotency key. Returns ("success", appt, verified) | ("failed", err) | ("unknown",).

    After an unknown result: look at the hospital's state (and once more after a short wait); the single
    retry happens only if the target state is still not visible. A definite error after an unknown try may be
    our own earlier write showing up (slot_taken, not_active), so the state is checked before believing it.
    """
    unknown_seen = False
    for attempt in range(2):
        try:
            return "success", do(), False
        except HospitalError as e:
            if e.kind == "definite":
                found = verify() if unknown_seen else None
                return ("success", found, True) if isinstance(found, dict) else ("failed", e)
            unknown_seen = True
            for wait in ([0] if attempt else [0, 1.0]):
                ctx.sleep(wait) if wait else None
                found = verify()
                if isinstance(found, dict):
                    return "success", found, True
    return ("unknown",)


def _finish(ctx: PatientContext, tool: str, action: str, outcome: tuple, shape: Callable[[dict], dict]) -> dict:
    if outcome[0] == "success":
        return _record(ctx, tool, "success", action=action, verified=outcome[2], **shape(outcome[1]))
    if outcome[0] == "failed":
        return _failed(ctx, tool, outcome[1])
    return _record(ctx, tool, "unknown", action=action)


def _appt_view(ctx: PatientContext):
    return lambda a: {"appointment": _compact(a, ctx)}


def book_appointment(args: BookIn, ctx: PatientContext):
    if not ctx.operation_id:
        raise ToolError("This action has not been confirmed by the patient.")
    start = iso(parse_start(args.start))
    key = ctx.operation_id  # same key for every retry of this one operation; a new operation gets a new key
    for attempt in range(3):
        try:
            appt = ctx.hospital.book(args.doctor_id, start, args.patient_name, ctx.phone, key)
            return _finish(ctx, "book_appointment", "booked", ("success", appt, False), _appt_view(ctx))
        except HospitalError as e:
            if e.kind == "definite":
                return _failed(ctx, "book_appointment", e)
            if attempt < 2:
                ctx.sleep(0.5 * 2**attempt)
    appts = _lookup(ctx)  # same-key retries are the authority; this is only extra evidence
    match = next((a for a in appts or [] if a["doctor_id"] == args.doctor_id and a["status"] == "booked"
                  and parse_start(a["start"]) == parse_start(start)), None)
    return _finish(ctx, "book_appointment", "booked", ("success", match, True) if match else ("unknown",), _appt_view(ctx))


def reschedule_appointment(args: RescheduleIn, ctx: PatientContext):
    new_start = iso(parse_start(args.new_start))

    def verify():
        appts = _lookup(ctx)
        if appts is None:
            return None
        return next((a for a in appts if a["id"] == args.appointment_id and a["status"] == "booked"
                     and parse_start(a["start"]) == parse_start(new_start)), False)

    outcome = _write_with_checks(lambda: ctx.hospital.reschedule(args.appointment_id, new_start), verify, ctx)
    return _finish(ctx, "reschedule_appointment", "rescheduled", outcome, _appt_view(ctx))


def cancel_appointment(args: CancelIn, ctx: PatientContext):
    def verify():
        appts = _lookup(ctx)
        if appts is None:
            return None
        return next((a for a in appts if a["id"] == args.appointment_id and a["status"] == "cancelled"), False)

    outcome = _write_with_checks(lambda: ctx.hospital.cancel(args.appointment_id), verify, ctx)
    return _finish(ctx, "cancel_appointment", "cancelled", outcome, _appt_view(ctx))


def send_payment_link(args: SendLinkIn, ctx: PatientContext):
    # No way to list links, so an unknown result cannot be checked and a retry could create a second link.
    for attempt in range(3):
        try:
            link = ctx.hospital.payment_link(args.bill_id)
            return _record(ctx, "send_payment_link", "success", action="payment_link_sent", verified=False,
                           link={"bill_id": link["bill_id"], "url": link["url"]})
        except HospitalError as e:
            if e.kind == "not_sent":  # provably never arrived: safe to try again
                ctx.sleep(0.5 * 2**attempt)
                continue
            if e.kind == "definite":
                return _failed(ctx, "send_payment_link", e)
            return _record(ctx, "send_payment_link", "unknown", action="payment_link_sent")
    return _failed(ctx, "send_payment_link", HospitalError("not_sent"))


_TOOLS: dict[str, tuple[str, type[BaseModel], Callable]] = {
    "find_doctors": ("List doctors, optionally for one specialty (for example Dermatology).", FindDoctorsIn, find_doctors),
    "free_slots": ("Free 30-minute slots for a doctor on a date (YYYY-MM-DD).", FreeSlotsIn, free_slots),
    "my_appointments": ("The patient's own appointments (no input: the patient is already identified).", NoInput, my_appointments),
    "my_bills": ("The patient's own bills (no input: the patient is already identified).", NoInput, my_bills),
    "book_appointment": ("Book a slot. Needs the patient's name. The patient is asked to confirm first.", BookIn, book_appointment),
    "reschedule_appointment": ("Move one of the patient's appointments to a new start time. Patient confirms first.", RescheduleIn, reschedule_appointment),
    "cancel_appointment": ("Cancel one of the patient's appointments. The patient confirms first.", CancelIn, cancel_appointment),
    "send_payment_link": ("Send a payment link for one of the patient's unpaid bills. Patient confirms first.", SendLinkIn, send_payment_link),
}


def build_registry(names: list[str] | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    for name in (_TOOLS if names is None else names):
        description, model, fn = _TOOLS[name]
        registry.register(name, description, model, fn)
    return registry
