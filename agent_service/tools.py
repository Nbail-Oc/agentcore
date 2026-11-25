"""Hospital tools. Reads are bound to the verified phone; each write has its own recovery rules.

Write outcomes are structured: success (2xx, or the hospital's own state shows the target), failed
(a definite error code), or unknown (we cannot tell). Only a definite answer or evidence of the target
state counts. "I looked and found nothing" is never treated as proof that a write did not happen.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Any, Callable

from pydantic import BaseModel, Field, StringConstraints

from agentcore.tools import ToolError, ToolRegistry
from agentcore.trace import Span

from .context import PatientContext
from .hospital import HospitalError

IST = timezone(timedelta(hours=5, minutes=30))
ClockTime = Annotated[str, StringConstraints(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")]
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


def within_limits(dt: datetime, not_before: str | None, not_after: str | None) -> bool:
    """Inclusive HH:MM limits in hospital time."""
    t = dt.astimezone(IST).strftime("%H:%M")
    return (not not_before or t >= not_before) and (not not_after or t <= not_after)


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def write_target(name: str, inp: dict) -> str:
    """What a write acts on: one appointment, one bill, or one exact booking."""
    if name == "book_appointment":
        return f"book|{inp['doctor_id']}|{iso(parse_start(inp['start']))}|{inp['patient_name']}"
    return inp.get("appointment_id") or inp["bill_id"]


def _read(fn: Callable[[], Any]) -> Any:
    try:
        return fn()
    except HospitalError as e:
        if e.kind == "definite":
            raise ToolError(e.code or "error") from None
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

LIMIT = ("Inclusive HH:MM, 24-hour. Set only when the patient stated a time limit: 'at or after 6:15 PM' is not_before 18:15, "
         "strictly 'after 6:15 PM' is not_before 18:16, 'before 11 AM' is not_after 10:59. If unsure what they mean, ask. "
         "It limits this time and any alternative offered if it is taken.")


class BookIn(BaseModel):
    doctor_id: DoctorId
    start: str = Field(max_length=40)
    patient_name: PersonName
    not_before: ClockTime | None = Field(None, description=LIMIT)
    not_after: ClockTime | None = Field(None, description=LIMIT)


class RescheduleIn(BaseModel):
    appointment_id: AppointmentId
    new_start: str = Field(max_length=40)
    not_before: ClockTime | None = Field(None, description=LIMIT)
    not_after: ClockTime | None = Field(None, description=LIMIT)


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
        result["instruction"] = "Do NOT tell the patient it happened. Say it could not be confirmed and that the chat is flagged for hospital staff (promise no reply time)."
    ctx.outcomes.append(result)
    return result


def _note(ctx: PatientContext, tool: str, e: HospitalError | None = None) -> None:
    """Failure evidence for the (protected) trace: our own classification, never the hospital's text or the patient's.

    A status alone does not show whether a write was saved: a 503 can come before or after the save.
    """
    if ctx.trace is None:
        return
    out = {"category": e.kind, "status": e.status, "code": e.code} if e else {"category": "handler_exception", "status": None, "code": None}
    if out["category"] in ("unknown", "handler_exception"):
        out["saved"] = "unknown"
    ctx.trace.add(Span("hospital", tool, None, out, 0.0, is_error=True, error=out["category"] + (f" {e.status}" if e and e.status else "")))


def unknown_outcome(ctx: PatientContext, tool: str) -> dict:
    """For a handler that began and then died before recording a result: it may have sent its request."""
    _note(ctx, tool)
    return _record(ctx, tool, "unknown", action="unconfirmed")


def _failed(ctx: PatientContext, tool: str, e: HospitalError) -> dict:
    extra = {}
    if e.code == "slot_taken":  # recovery path: a proposal is parked for confirmation, never booked without a yes
        extra["instruction"] = (
            "The slot is gone. Do not ask the patient to choose: look up free_slots for the same doctor and day, then call "
            "the same tool again with the nearest free time. The patient will be asked to confirm it. "
            "Keep any time limit the patient gave (not_before / not_after). Never say anything was booked or moved.")
    code = e.code or ("rejected" if e.kind == "definite" else "unreachable")
    return _record(ctx, tool, "failed", code=code, **extra)


def _appt_ok(reply: Any, appointment_id: str | None = None) -> bool:
    ok = isinstance(reply, dict) and all(isinstance(reply.get(k), str) for k in ("id", "doctor_id", "start", "status"))
    return ok and (appointment_id is None or reply["id"] == appointment_id)


def _same_time(a: str, b: str) -> bool:
    try:
        return parse_start(a) == parse_start(b)
    except ToolError:
        return False


def _expect(call: Callable[[], Any], ok: Callable[[Any], bool]) -> Callable[[], Any]:
    """A 2xx reply only counts if it is about the target we asked for. Anything else is as good as no reply."""
    def run():
        reply = call()
        if not ok(reply):
            raise HospitalError("unknown")  # no status: treated like a timeout, so no retry can collide with it
        return reply
    return run


def _write_with_checks(do: Callable[[], dict], verify: Callable[[], dict | bool | None], ctx: PatientContext, tool: str):
    """For writes with no idempotency key. Returns ("success", appt, verified) | ("failed", err) | ("unknown",).

    What each failure proves (this API has no idempotency key and no operation status):
      not_sent  the request never arrived: retry once, no lookup needed.
      5xx reply the hospital finished handling that request, so it cannot land later: look at its state
                  (twice, after a short wait), and retry once only if the target state is still not visible.
      anything else (timeout, 202 accepted, 408, 429...) the request may still be running, so a second
                  request could collide with it: look at the state, never retry, and stay unknown if the
                  target state is not visible.
    A definite error after an unknown try may be our own write showing up (slot_taken, not_active), so the
    state is checked before believing it. A matching read proves the state, not who caused it.
    """
    unknown_seen, last = False, None
    for attempt in range(2):
        try:
            return "success", do(), False
        except HospitalError as e:
            _note(ctx, tool, e)
            last = e
            if e.kind == "definite":
                if not unknown_seen:
                    return "failed", e
                found = verify()
                return ("success", found, True) if isinstance(found, dict) else ("unknown",)
            if e.kind == "not_sent":
                ctx.sleep(0.5)
                continue
            unknown_seen = True
            for wait in ([0] if attempt else [0, 1.0]):
                ctx.sleep(wait) if wait else None
                found = verify()
                if isinstance(found, dict):
                    return "success", found, True
            if not (e.status and e.status >= 500):  # no reply, 202, 408...: it may still be running
                return ("unknown",)
    return ("unknown",) if unknown_seen else ("failed", last)


def _finish(ctx: PatientContext, tool: str, action: str, outcome: tuple, shape: Callable[[dict], dict]) -> dict:
    if outcome[0] == "success":
        return _record(ctx, tool, "success", action=action, verified=outcome[2], **shape(outcome[1]))
    if outcome[0] == "failed":
        return _failed(ctx, tool, outcome[1])
    return _record(ctx, tool, "unknown", action=action)


def _appt_view(ctx: PatientContext):
    return lambda a: {"appointment": _compact(a, ctx)}


def _confirmed(ctx: PatientContext) -> None:
    """Second lock behind the write policy: only the application sets operation_id, after the patient's yes."""
    if not ctx.operation_id:
        raise ToolError("This action has not been confirmed by the patient.")
    ctx.write_started = True


def book_appointment(args: BookIn, ctx: PatientContext):
    _confirmed(ctx)
    start = iso(parse_start(args.start))
    key = ctx.operation_id  # same key for every retry of this one operation; a new operation gets a new key
    unknown_seen = False
    def ok(r):
        return (_appt_ok(r) and r["doctor_id"] == args.doctor_id and r["status"] == "booked" and _same_time(r["start"], start)
                and r.get("phone") == ctx.phone and r.get("patient_name") == args.patient_name)

    for attempt in range(3):
        try:
            appt = _expect(lambda: ctx.hospital.book(args.doctor_id, start, args.patient_name, ctx.phone, key), ok)()
            return _finish(ctx, "book_appointment", "booked", ("success", appt, False), _appt_view(ctx))
        except HospitalError as e:
            _note(ctx, "book_appointment", e)
            if e.kind == "definite":
                if not unknown_seen:
                    return _failed(ctx, "book_appointment", e)
                break  # an earlier try may still complete, so this refusal proves nothing either
            last = e
            unknown_seen = unknown_seen or e.kind != "not_sent"  # a request that never left cannot have been saved
            if attempt < 2:
                ctx.sleep(0.5 * 2**attempt)
    if not unknown_seen:  # every attempt provably never left the building
        return _failed(ctx, "book_appointment", last)
    # Unknown after every same-key try. A matching record would only show the current state (another actor, or an
    # earlier request, may have made it), so it is not used: only a same-key 2xx reply proves this booking.
    return _finish(ctx, "book_appointment", "booked", ("unknown",), _appt_view(ctx))


def reschedule_appointment(args: RescheduleIn, ctx: PatientContext):
    _confirmed(ctx)
    new_start = iso(parse_start(args.new_start))

    def verify():
        appts = _lookup(ctx)
        if appts is None:
            return None
        return next((a for a in appts if a["id"] == args.appointment_id and a["status"] == "booked"
                     and parse_start(a["start"]) == parse_start(new_start)), False)

    def ok(r):
        return _appt_ok(r, args.appointment_id) and r["status"] == "booked" and _same_time(r["start"], new_start)

    outcome = _write_with_checks(_expect(lambda: ctx.hospital.reschedule(args.appointment_id, new_start), ok), verify, ctx, "reschedule_appointment")
    return _finish(ctx, "reschedule_appointment", "rescheduled", outcome, _appt_view(ctx))


def cancel_appointment(args: CancelIn, ctx: PatientContext):
    _confirmed(ctx)
    def verify():
        appts = _lookup(ctx)
        if appts is None:
            return None
        return next((a for a in appts if a["id"] == args.appointment_id and a["status"] == "cancelled"), False)

    def ok(r):
        return _appt_ok(r, args.appointment_id) and r["status"] == "cancelled"

    outcome = _write_with_checks(_expect(lambda: ctx.hospital.cancel(args.appointment_id), ok), verify, ctx, "cancel_appointment")
    return _finish(ctx, "cancel_appointment", "cancelled", outcome, _appt_view(ctx))


def send_payment_link(args: SendLinkIn, ctx: PatientContext):
    _confirmed(ctx)
    # No way to list links, so an unknown result cannot be checked and a retry could create a second link.
    for attempt in range(3):
        try:
            link = ctx.hospital.payment_link(args.bill_id)
            if not (isinstance(link, dict) and link.get("bill_id") == args.bill_id
                    and isinstance(link.get("url"), str) and link["url"].startswith("https://")):
                return _record(ctx, "send_payment_link", "unknown", action="payment_link_sent")  # not the link we asked for
            return _record(ctx, "send_payment_link", "success", action="payment_link_sent", verified=False,
                           link={"bill_id": link["bill_id"], "url": link["url"]})
        except HospitalError as e:
            _note(ctx, "send_payment_link", e)
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
