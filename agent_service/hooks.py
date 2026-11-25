"""Hospital safety rules, enforced in code before any write. These are the hospital's policies, so they live
here and not in agentcore. Order: own data, current state, 4-hour rule, then the confirmation gate."""

import hashlib
import re
from datetime import datetime, timedelta

from agentcore.hooks import Block, Confirm, ToolCall
from agentcore.tools import ToolError

from .context import PatientContext
from .hospital import HospitalError
from .tools import WRITE_TOOLS, parse_start, within_limits, write_target

FOUR_HOURS = timedelta(hours=4)

_UNIT = r"(?!\s?(?:days?|weeks?|months?|hours?|minutes?|mins?)\b)"
# Words that set a limit, in matching order ("at or after" before "after", "not before" before "before"):
# (words, field, minutes added to the stated time). Strict "after 5 pm" means from 17:01; "at or after" means 17:00.
_LIMIT_WORDS = [
    (r"at or after|on or after|from|starting at|not before|no earlier than", "not_before", 0),
    (r"after|later than", "not_before", 1),
    (r"at or before|on or before|not after|no later than|until", "not_after", 0),
    (r"before|earlier than", "not_after", -1),
]
_LIMIT = re.compile(
    r"\b(?:" + "|".join(f"(?P<w{i}>{words})" for i, (words, _, _) in enumerate(_LIMIT_WORDS)) + r")\s+"
    r"(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s?(?P<ap>am|pm)?" + _UNIT, re.IGNORECASE)
_HINGLISH_LIMIT = re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm|baje)\s?(?:ke\s)?(?:baad|pehle)\b", re.IGNORECASE)


def stated_limits(message: str) -> dict | str | None:
    """What the patient's latest message says about a time limit.

    None: no limit phrase. A dict: the inclusive not_before / not_after those words mean. "ask": a limit is
    there but its time is ambiguous (no AM or PM) or in a form this cannot read: clarify, do not guess.
    shortcut: English words and Hinglish "baje ke baad" only; Hindi script and other wording are not read.
    """
    limits = {}
    for m in _LIMIT.finditer(message):
        i = next(i for i in range(len(_LIMIT_WORDS)) if m.group(f"w{i}"))
        _, field, shift = _LIMIT_WORDS[i]
        hour, ap = int(m["h"]), (m["ap"] or "").lower()
        if ap and 1 <= hour <= 12:
            hour = hour % 12 + (12 if ap == "pm" else 0)
        elif not (m["m"] and hour >= 13) or hour > 23:  # no AM/PM and not a 24-hour time
            return "ask"
        total = hour * 60 + int(m["m"] or 0) + shift
        if not 0 <= total < 1440:
            return "ask"
        limits[field] = f"{total // 60:02d}:{total % 60:02d}"
    if _HINGLISH_LIMIT.search(message):
        return "ask"
    return limits or None


def _limit_problem(args, at: datetime, ctx: PatientContext):
    """The limit the model set must match what the patient just said and must hold for the time proposed."""
    how = "The limits are not_before / not_after (inclusive HH:MM, hospital time)."
    stated = stated_limits(ctx.message)
    if stated == "ask":
        return Block(f"The patient gave a time limit without AM or PM, or in a form that is not clear. Ask them what they mean "
                     f"before setting not_before / not_after. {how}")
    for field, value in (stated or {}).items():
        if getattr(args, field) != value:
            return Block(f"The patient's words mean {field}={value}. Set it, or ask the patient what they mean. {how}")
    if (args.not_before or args.not_after) and not within_limits(at, args.not_before, args.not_after):
        return Block(f"That time is outside the time limit the patient gave. Pick a time inside it, or ask the patient. {how}")


def approval_key(fingerprint: str, details: str) -> str:
    """What a patient's yes covers: this exact call AND the details that were read back to them."""
    return f"{fingerprint}:{hashlib.sha256(details.encode()).hexdigest()}"


def when(dt: datetime) -> str:
    return f"{dt:%A} {dt.day} {dt:%b %Y}, {dt:%H:%M}"


def _doctor(ctx: PatientContext, doctor_id: str) -> dict | None:
    """The doctor named in a read-back. A failed lookup raises (the policy then blocks); it never falls back to the id."""
    return next((d for d in ctx.doctors() if d["id"] == doctor_id), None)


def _conflicts(unresolved: str, target: str, name: str) -> bool:
    """Does a new write clash with an earlier one whose outcome is unknown?

    A booking is not a clash with itself (it is retried with the same key) nor with unrelated bookings; it is
    one only for the same patient with the same doctor on the same day at another time (a likely replacement,
    which would give two appointments if the first one lands). Any other write clashes on the same target.
    """
    if name != "book_appointment":
        return unresolved == target
    old, new = unresolved.split("|"), target.split("|")
    return old[0] == "book" and old[1] == new[1] and old[2][:10] == new[2][:10] and old[3] == new[3] and old[2] != new[2]


def write_policy(call: ToolCall, ctx: PatientContext):
    """before_tool hook. Reads pass through; every write is checked fresh, every time it is attempted."""
    if call.name not in WRITE_TOOLS:
        return None
    try:
        return _policy(call, ctx)
    except (HospitalError, ToolError):
        return Block("The hospital system could not be checked right now, so nothing was changed. Try again shortly.")


def _policy(call: ToolCall, ctx: PatientContext):
    args, now = call.args, ctx.hospital.now()  # the hospital's clock, never this machine's

    # An earlier write on this target may still land: no other change to it. (A booking is instead retried
    # with the same idempotency key, see the orchestrator, so the hospital can never make two of it.)
    target = write_target(call.name, call.input)
    if any(_conflicts(u["target"], target, call.name) for u in ctx.unresolved if not (u.get("in_flight") and u["operation_id"] == ctx.operation_id)):
        ctx.flags["handoff"] = True
        return Block("An earlier request for this is still unconfirmed, so nothing more is changed. Tell the patient it needs hospital "
                     "staff, that you have flagged the chat for them (promise no reply time), and that they can call the front desk on 020-4000-1234.")

    if call.name == "book_appointment":
        start = parse_start(args.start)
        if start <= now:
            return Block("That time has already passed. Offer a future slot.")
        if (problem := _limit_problem(args, start, ctx)):
            return problem
        doctor = _doctor(ctx, args.doctor_id)
        if doctor is None:
            return Block(f"Doctor {args.doctor_id} was not found. Use find_doctors.")
        details = f"Book {doctor['name']} ({doctor['specialty']}) on {when(start)} for {args.patient_name}"
        for u in ctx.unresolved:  # not assumed to be a replacement: the patient decides, knowing both may exist
            if u["target"].startswith("book|") and u["target"] != target:
                _, earlier_doctor, earlier_start, _ = u["target"].split("|")
                details += (f". Note: your earlier request for {ctx.doctor(earlier_doctor)['name']} on {when(parse_start(earlier_start))} "
                            "could not be confirmed; if it went through, you will have both appointments")

    elif call.name == "send_payment_link":
        bill = next((b for b in ctx.hospital.bills(ctx.phone) if b["id"] == args.bill_id), None)
        if bill is None:  # same answer whether it does not exist or belongs to someone else
            return Block(f"Bill {args.bill_id} was not found.")
        if bill["status"] != "unpaid":
            return Block("This bill is already paid.")
        details = f"Send a payment link for bill {bill['id']}: Rs {bill['amount_inr']}"  # no hospital free text in a read-back

    else:  # reschedule_appointment, cancel_appointment
        appt = next((a for a in ctx.hospital.appointments(ctx.phone) if a["id"] == args.appointment_id), None)
        if appt is None:
            return Block(f"Appointment {args.appointment_id} was not found.")
        if appt["status"] != "booked":
            return Block("That appointment is not active (it is already cancelled).")
        current = parse_start(appt["start"])
        if current - now < FOUR_HOURS:
            ctx.flags["handoff"] = True
            return Block("This appointment is less than 4 hours away (or has started), so it cannot be changed in chat. "
                         "Tell the patient it needs hospital staff, that you have flagged the chat for them (promise no reply time), and that they can call the front desk on 020-4000-1234.")
        doctor = _doctor(ctx, appt["doctor_id"])
        if doctor is None:
            return Block("The doctor for that appointment could not be identified, so nothing was changed. Tell the patient it needs hospital staff.")
        doctor = doctor["name"]
        if call.name == "cancel_appointment":
            details = f"Cancel appointment {appt['id']} with {doctor} on {when(current)}"
        else:
            new = parse_start(args.new_start)
            if new <= now:
                return Block("That new time has already passed. Offer a future slot.")
            if (problem := _limit_problem(args, new, ctx)):
                return problem
            details = f"Move appointment {appt['id']} with {doctor} from {when(current)} to {when(new)}"

    if ctx.read_only:
        return Block("Changes are paused for this message. Tell the patient to send their request again shortly.")
    if approval_key(call.fingerprint, details) in ctx.approved:  # same call, same details as read back, checks just passed again
        return None
    return Confirm(details)
