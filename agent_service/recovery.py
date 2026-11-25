"""Recovery after a lost race (slot_taken): propose the same action at the nearest free time.

The proposal goes through the normal hooks and is parked like any write, so it needs its own clear yes.
Nothing here can book anything.
"""

from datetime import datetime

from agentcore.tools import ToolError

from .context import PatientContext
from .hospital import HospitalError
from .tools import iso, parse_start, within_limits


def nearest_free_slot(free: list[str], wanted: datetime) -> str | None:
    """The free time closest to the wanted one; a later time wins a tie."""
    slots = [parse_start(s) for s in free]
    if not slots:
        return None
    return iso(min(slots, key=lambda s: (abs((s - wanted).total_seconds()), s <= wanted)))


def alternative_input(pending: dict, ctx: PatientContext) -> dict | None:
    """The stored call's input with the nearest free time (same doctor, same day), or None if there is none."""
    try:
        if pending["name"] == "book_appointment":
            doctor_id, field = pending["input"]["doctor_id"], "start"
        elif pending["name"] == "reschedule_appointment":
            appointment = next(a for a in ctx.hospital.appointments(ctx.phone) if a["id"] == pending["input"]["appointment_id"])
            doctor_id, field = appointment["doctor_id"], "new_start"
        else:
            return None
        wanted = parse_start(pending["input"][field])
        lo, hi = pending["input"].get("not_before"), pending["input"].get("not_after")  # limits the patient gave
        free = [s for s in ctx.hospital.slots(doctor_id, wanted.date().isoformat())["free"] if within_limits(parse_start(s), lo, hi)]
        best = nearest_free_slot(free, wanted)
    except (HospitalError, ToolError, StopIteration, KeyError):
        return None
    return {**pending["input"], field: best} if best else None
