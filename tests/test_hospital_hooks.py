"""Tests for the hospital safety hooks: own data only, state, 4-hour rule, confirmation gate. Offline."""

from datetime import datetime, timedelta

import pytest

from agent_service.hooks import write_policy
from agent_service.hospital import HospitalError
from agent_service.tools import IST, build_registry
from agentcore.hooks import Hooks
from agentcore.loop import execute_tool_call
from agentcore.trace import Trace
from tests.fake_hospital import NEW, UNKNOWN, FakeHospital, _ctx, appt

NOW = datetime(2026, 12, 1, 9, 0, tzinfo=IST)
FAR = "2026-12-03T17:30:00+05:30"
BILL = {"id": "B1", "phone": "9988776655", "amount_inr": 500, "description": "Consultation", "status": "unpaid"}


def _hospital(appointments=None, bills=None, **script):
    return FakeHospital(now=[NOW], appointments=[appointments if appointments is not None else [appt(FAR)]],
                        bills=[bills if bills is not None else [BILL]], **script)


def _exec(name, args, ctx):
    return execute_tool_call(build_registry(), name, args, ctx, Trace(), Hooks(before_tool=[write_policy]))


def _writes(hospital):
    return [c for c in hospital.calls if c[0] in ("book", "reschedule", "cancel", "payment_link")]


def _approve_and_run(name, args, ctx):
    first = _exec(name, args, ctx)
    ctx.approved.add(first.pending.fingerprint)
    return first, _exec(name, args, ctx)


# ---------------------------------------------------------------- own data only

@pytest.mark.parametrize("name,args", [
    ("cancel_appointment", {"appointment_id": "A1001"}),
    ("reschedule_appointment", {"appointment_id": "A1001", "new_start": NEW}),
    ("send_payment_link", {"bill_id": "B9"}),
])
def test_ids_the_patient_does_not_own_are_refused_exactly_like_missing_ones(name, args):
    hospital = _hospital()  # this patient owns A1003 and B1 only

    result = _exec(name, args, _ctx(hospital))

    assert result.is_error and "not found" in result.content
    assert result.pending is None
    assert _writes(hospital) == []


def test_the_refusal_does_not_reveal_whether_the_id_exists_for_someone_else():
    hospital = _hospital()
    ctx = _ctx(hospital)

    other = _exec("cancel_appointment", {"appointment_id": "A1001"}, ctx).content
    missing = _exec("cancel_appointment", {"appointment_id": "A7777"}, ctx).content

    assert other.replace("A1001", "X") == missing.replace("A7777", "X")
    assert "10:30" not in other and "Priya" not in other


def test_the_lookup_is_scoped_to_the_verified_phone():
    hospital = _hospital()

    _exec("cancel_appointment", {"appointment_id": "A1003"}, _ctx(hospital, phone="9988776655"))

    assert [c for c in hospital.calls if c[0] == "appointments"][0][1] == ("9988776655",)


def test_if_the_hospital_cannot_be_checked_the_write_is_refused():
    hospital = FakeHospital(now=[NOW], appointments=[UNKNOWN])

    result = _exec("cancel_appointment", {"appointment_id": "A1003"}, _ctx(hospital))

    assert result.is_error and result.pending is None
    assert _writes(hospital) == []


# ---------------------------------------------------------------- state

def test_cancelled_appointments_and_paid_bills_cannot_be_changed():
    cancelled = _hospital(appointments=[appt(FAR, status="cancelled")])
    paid = _hospital(bills=[{**BILL, "status": "paid"}])

    assert "not active" in _exec("cancel_appointment", {"appointment_id": "A1003"}, _ctx(cancelled)).content
    assert "already paid" in _exec("send_payment_link", {"bill_id": "B1"}, _ctx(paid)).content


def test_booking_or_moving_to_a_time_in_the_past_is_refused():
    hospital = _hospital()
    ctx = _ctx(hospital)

    booked = _exec("book_appointment", {"doctor_id": "d5", "start": "2026-12-01T08:00:00+05:30",
                                        "patient_name": "Sameer"}, ctx)
    moved = _exec("reschedule_appointment", {"appointment_id": "A1003", "new_start": "2026-11-30T10:00:00+05:30"}, ctx)

    assert booked.is_error and moved.is_error
    assert _writes(hospital) == []


# ---------------------------------------------------------------- 4-hour rule

@pytest.mark.parametrize("delta,blocked", [
    (timedelta(hours=3, minutes=59), True),
    (timedelta(hours=4), False),  # "up to 4 hours before" is still allowed
    (timedelta(hours=4, minutes=1), False),
    (timedelta(hours=-1), True),  # already started
])
@pytest.mark.parametrize("name,extra", [("cancel_appointment", {}), ("reschedule_appointment", {"new_start": NEW})])
def test_four_hour_rule_at_the_boundary(delta, blocked, name, extra):
    start = (NOW + delta).isoformat()
    hospital = _hospital(appointments=[appt(start)])
    ctx = _ctx(hospital)

    result = _exec(name, {"appointment_id": "A1003", **extra}, ctx)

    assert result.is_error is blocked
    assert (result.pending is None) is blocked
    assert ctx.flags["handoff"] is blocked
    assert _writes(hospital) == []


def test_the_four_hour_block_tells_the_model_that_staff_take_over():
    hospital = _hospital(appointments=[appt((NOW + timedelta(hours=1)).isoformat())])

    result = _exec("cancel_appointment", {"appointment_id": "A1003"}, _ctx(hospital))

    assert "staff" in result.content.lower() and "4 hours" in result.content


# ---------------------------------------------------------------- confirmation gate

def test_a_write_is_parked_with_details_built_from_the_exact_arguments():
    hospital = _hospital(cancel=[appt(FAR, status="cancelled")])

    result = _exec("cancel_appointment", {"appointment_id": "A1003"}, _ctx(hospital))

    assert result.pending is not None and _writes(hospital) == []
    details = result.pending.details
    assert "A1003" in details and "Dr. Imran Khan" in details and "Thursday 3 Dec 2026, 17:30" in details


def test_book_reschedule_and_payment_details_name_every_important_field():
    hospital = _hospital()
    ctx = _ctx(hospital)

    book = _exec("book_appointment", {"doctor_id": "d5", "start": NEW, "patient_name": "Sameer Kulkarni"}, ctx)
    move = _exec("reschedule_appointment", {"appointment_id": "A1003", "new_start": NEW}, ctx)
    pay = _exec("send_payment_link", {"bill_id": "B1"}, ctx)

    assert all(s in book.pending.details for s in ("Dr. Imran Khan", "Saturday 5 Dec 2026, 10:00", "Sameer Kulkarni"))
    assert all(s in move.pending.details for s in ("Thursday 3 Dec 2026, 17:30", "Saturday 5 Dec 2026, 10:00"))
    assert "Rs 500" in pay.pending.details and "B1" in pay.pending.details


def test_an_approved_call_runs_after_the_checks_pass_again():
    hospital = _hospital(cancel=[appt(FAR, status="cancelled")])
    ctx = _ctx(hospital)

    first, second = _approve_and_run("cancel_appointment", {"appointment_id": "A1003"}, ctx)

    assert first.pending is not None and second.pending is None and not second.is_error
    assert len(_writes(hospital)) == 1


def test_approval_covers_only_the_exact_call_that_was_read_back():
    hospital = _hospital(appointments=[appt(FAR), appt(FAR, id="A1004")])
    ctx = _ctx(hospital)
    first = _exec("cancel_appointment", {"appointment_id": "A1003"}, ctx)
    ctx.approved.add(first.pending.fingerprint)

    other = _exec("cancel_appointment", {"appointment_id": "A1004"}, ctx)

    assert other.pending is not None and _writes(hospital) == []


def test_things_that_changed_since_the_read_back_block_even_an_approved_call():
    # read back while it was 2 days away, confirmed after the clock moved to 1 hour before
    hospital = _hospital()
    ctx = _ctx(hospital)
    first = _exec("cancel_appointment", {"appointment_id": "A1003"}, ctx)
    ctx.approved.add(first.pending.fingerprint)
    hospital.script["now"] = [datetime.fromisoformat(FAR) - timedelta(hours=1)]

    late = _exec("cancel_appointment", {"appointment_id": "A1003"}, ctx)

    assert late.is_error and ctx.flags["handoff"] is True and _writes(hospital) == []

    # appointment was cancelled by someone else in between
    hospital.script["now"] = [NOW]
    hospital.script["appointments"] = [[appt(FAR, status="cancelled")]]
    gone = _exec("cancel_appointment", {"appointment_id": "A1003"}, ctx)

    assert gone.is_error and _writes(hospital) == []


def test_read_only_turns_refuse_every_write_even_approved_ones():
    hospital = _hospital()
    ctx = _ctx(hospital)
    first = _exec("cancel_appointment", {"appointment_id": "A1003"}, ctx)
    ctx.approved.add(first.pending.fingerprint)
    ctx.read_only = True

    result = _exec("cancel_appointment", {"appointment_id": "A1003"}, ctx)

    assert result.is_error and result.pending is None and _writes(hospital) == []


def test_reads_are_not_touched_by_the_write_policy():
    hospital = FakeHospital(appointments=[[appt(FAR)]])  # no clock needed

    result = _exec("my_appointments", {}, _ctx(hospital))

    assert not result.is_error and result.pending is None
