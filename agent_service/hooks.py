"""Hospital safety rules, enforced in code before any write. These are the hospital's policies, so they live
here and not in agentcore. Order: own data, current state, 4-hour rule, then the confirmation gate."""

from datetime import datetime, timedelta

from agentcore.hooks import Block, Confirm, ToolCall
from agentcore.tools import ToolError

from .context import PatientContext
from .hospital import HospitalError
from .tools import WRITE_TOOLS, parse_start

FOUR_HOURS = timedelta(hours=4)


def when(dt: datetime) -> str:
    return f"{dt:%A} {dt.day} {dt:%b %Y}, {dt:%H:%M}"


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

    if call.name == "book_appointment":
        start = parse_start(args.start)
        if start <= now:
            return Block("That time has already passed. Offer a future slot.")
        doctor = ctx.doctor(args.doctor_id)
        details = f"Book {doctor['name']} ({doctor['specialty']}) on {when(start)} for {args.patient_name}"

    elif call.name == "send_payment_link":
        bill = next((b for b in ctx.hospital.bills(ctx.phone) if b["id"] == args.bill_id), None)
        if bill is None:  # same answer whether it does not exist or belongs to someone else
            return Block(f"Bill {args.bill_id} was not found.")
        if bill["status"] != "unpaid":
            return Block("This bill is already paid.")
        details = f"Send a payment link for bill {bill['id']}: Rs {bill['amount_inr']} ({bill['description']})"

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
        doctor = ctx.doctor(appt["doctor_id"])["name"]
        if call.name == "cancel_appointment":
            details = f"Cancel appointment {appt['id']} with {doctor} on {when(current)}"
        else:
            new = parse_start(args.new_start)
            if new <= now:
                return Block("That new time has already passed. Offer a future slot.")
            details = f"Move appointment {appt['id']} with {doctor} from {when(current)} to {when(new)}"

    if ctx.read_only:
        return Block("Changes are paused for this message. Tell the patient to send their request again shortly.")
    if call.fingerprint in ctx.approved:  # the patient confirmed exactly this call; the checks above just passed again
        return None
    return Confirm(details)
