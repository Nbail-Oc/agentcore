"""Tests for hospital tools: patient-bound reads and per-write recovery from unknown outcomes. Offline."""

import json
from datetime import datetime

import pytest

from agent_service.hospital import HospitalError
from agent_service.tools import build_registry
from tests.fake_hospital import DOCTORS, NEW, NOT_SENT, OLD, UNKNOWN, FakeHospital, _ctx, appt, definite

def _run(name, args, ctx, tools=None):
    result = build_registry(tools).execute(name, args, ctx)
    return result, (json.loads(result.content) if not result.is_error else None)


BOOK = {"doctor_id": "d5", "start": NEW, "patient_name": "Sameer Kulkarni"}


# ---------------------------------------------------------------- reads are bound to the verified phone

def test_reads_use_the_verified_phone_and_take_no_phone_from_the_model():
    hospital = FakeHospital(appointments=[[appt()]], bills=[[]])
    ctx = _ctx(hospital)

    _run("my_appointments", {}, ctx)
    _run("my_bills", {}, ctx)
    rejected, _ = _run("my_appointments", {"phone": "9876543210"}, ctx)

    assert [c for c in hospital.calls if c[0] in ("appointments", "bills")] == [
        ("appointments", ("9988776655",)), ("bills", ("9988776655",))]
    assert rejected.is_error


@pytest.mark.parametrize("name,args", [
    ("cancel_appointment", {"appointment_id": "../_admin/reset"}),
    ("reschedule_appointment", {"appointment_id": "A1/../x", "new_start": NEW}),
    ("send_payment_link", {"bill_id": "B1;drop"}),
    ("free_slots", {"doctor_id": "d1/../..", "date": "2026-12-04"}),
    ("book_appointment", {**BOOK, "doctor_id": "x"}),
    ("book_appointment", {**BOOK, "patient_name": "Robert'); DROP TABLE"}),
    ("book_appointment", {**BOOK, "phone": "9876543210"}),
])
def test_unsafe_tool_input_is_rejected_before_any_hospital_call(name, args):
    hospital = FakeHospital()

    result, _ = _run(name, args, _ctx(hospital))

    assert result.is_error
    assert hospital.calls == []


def test_a_read_error_reaches_the_model_without_internals():
    hospital = FakeHospital(bills=[UNKNOWN])

    result, _ = _run("my_bills", {}, _ctx(hospital))

    assert result.is_error and "503" not in result.content


def test_naive_start_times_are_treated_as_hospital_time_and_bad_ones_rejected():
    hospital = FakeHospital(book=[appt()])

    _run("book_appointment", {**BOOK, "start": "2026-12-05T10:00:00"}, _ctx(hospital))
    bad, _ = _run("book_appointment", {**BOOK, "start": "next friday"}, _ctx(hospital))

    assert [c for c in hospital.calls if c[0] == "book"][-1][1][1] == NEW
    assert bad.is_error


# ---------------------------------------------------------------- booking

def test_booking_success_uses_the_operation_id_as_idempotency_key():
    hospital = FakeHospital(book=[appt()])
    ctx = _ctx(hospital, op="op-7")

    _, out = _run("book_appointment", BOOK, ctx)

    assert out["outcome"] == "success" and out["verified"] is False
    assert [c for c in hospital.calls if c[0] == "book"][-1][1][-1] == "op-7"
    assert ctx.outcomes[-1]["outcome"] == "success"


def test_booking_retries_the_same_key_after_an_unknown_result_and_never_makes_a_new_one():
    hospital = FakeHospital(book=[UNKNOWN, UNKNOWN, appt()])

    _, out = _run("book_appointment", BOOK, _ctx(hospital, op="op-9"))

    assert out["outcome"] == "success"
    keys = [c[1][-1] for c in hospital.calls if c[0] == "book"]
    assert keys == ["op-9", "op-9", "op-9"]


def test_two_separate_operations_get_different_keys():
    hospital = FakeHospital(book=[appt()])

    _run("book_appointment", BOOK, _ctx(hospital, op="op-A"))
    _run("book_appointment", BOOK, _ctx(hospital, op="op-B"))

    assert [c[1][-1] for c in hospital.calls if c[0] == "book"] == ["op-A", "op-B"]


def test_booking_stays_unknown_when_lookup_finds_nothing_because_the_request_may_still_complete():
    hospital = FakeHospital(book=[UNKNOWN], appointments=[[]])
    ctx = _ctx(hospital)

    _, out = _run("book_appointment", BOOK, ctx)

    assert out["outcome"] == "unknown"
    assert ctx.flags["handoff"] is True


def test_booking_is_success_when_lookup_shows_it_saved():
    hospital = FakeHospital(book=[UNKNOWN], appointments=[[appt()]])

    _, out = _run("book_appointment", BOOK, _ctx(hospital))

    assert (out["outcome"], out["verified"]) == ("success", True)


def test_booking_stays_unknown_when_the_lookup_itself_fails():
    hospital = FakeHospital(book=[UNKNOWN], appointments=[UNKNOWN])

    _, out = _run("book_appointment", BOOK, _ctx(hospital))

    assert out["outcome"] == "unknown"


def test_booking_a_taken_slot_is_a_definite_failure_without_retry():
    hospital = FakeHospital(book=[definite("slot_taken")])

    _, out = _run("book_appointment", BOOK, _ctx(hospital))

    assert (out["outcome"], out["code"]) == ("failed", "slot_taken")
    assert len([c for c in hospital.calls if c[0] == "book"]) == 1


# ---------------------------------------------------------------- reschedule and cancel (no idempotency key)

def test_reschedule_unknown_but_lookup_shows_the_new_time_is_success_without_a_second_write():
    hospital = FakeHospital(reschedule=[UNKNOWN], appointments=[[appt(NEW)]])

    _, out = _run("reschedule_appointment", {"appointment_id": "A1003", "new_start": NEW}, _ctx(hospital))

    assert (out["outcome"], out["verified"]) == ("success", True)
    assert len([c for c in hospital.calls if c[0] == "reschedule"]) == 1


def test_reschedule_unchanged_twice_then_one_retry_succeeds():
    hospital = FakeHospital(reschedule=[UNKNOWN, appt(NEW)], appointments=[[appt(OLD)]])
    ctx = _ctx(hospital)

    _, out = _run("reschedule_appointment", {"appointment_id": "A1003", "new_start": NEW}, ctx)

    assert out["outcome"] == "success"
    assert len([c for c in hospital.calls if c[0] == "reschedule"]) == 2
    assert len([c for c in hospital.calls if c[0] == "appointments"]) == 2  # check, wait, check again, then retry
    assert ctx.sleeps  # waited before the recheck


def test_reschedule_never_retries_more_than_once_and_stays_unknown():
    hospital = FakeHospital(reschedule=[UNKNOWN], appointments=[[appt(OLD)]])
    ctx = _ctx(hospital)

    _, out = _run("reschedule_appointment", {"appointment_id": "A1003", "new_start": NEW}, ctx)

    assert out["outcome"] == "unknown"
    assert len([c for c in hospital.calls if c[0] == "reschedule"]) == 2
    assert ctx.flags["handoff"] is True


def test_a_misleading_slot_taken_after_our_own_saved_write_is_resolved_by_lookup():
    # first try saved but returned 503; the retry then sees our own new slot as taken
    saved = FakeHospital(reschedule=[UNKNOWN, definite("slot_taken")], appointments=[[appt(OLD)], [appt(OLD)], [appt(NEW)]])
    _, out = _run("reschedule_appointment", {"appointment_id": "A1003", "new_start": NEW}, _ctx(saved))
    assert (out["outcome"], out["verified"]) == ("success", True)

    taken = FakeHospital(reschedule=[UNKNOWN, definite("slot_taken")], appointments=[[appt(OLD)]])
    _, out = _run("reschedule_appointment", {"appointment_id": "A1003", "new_start": NEW}, _ctx(taken))
    assert (out["outcome"], out["code"]) == ("failed", "slot_taken")


def test_reschedule_definite_failure_on_first_try_needs_no_lookup():
    hospital = FakeHospital(reschedule=[definite("slot_taken")])

    _, out = _run("reschedule_appointment", {"appointment_id": "A1003", "new_start": NEW}, _ctx(hospital))

    assert out["outcome"] == "failed"
    assert not [c for c in hospital.calls if c[0] == "appointments"]


def test_cancel_unknown_then_lookup_shows_cancelled_is_success():
    hospital = FakeHospital(cancel=[UNKNOWN], appointments=[[appt(status="cancelled")]])

    _, out = _run("cancel_appointment", {"appointment_id": "A1003"}, _ctx(hospital))

    assert (out["outcome"], out["verified"]) == ("success", True)
    assert len([c for c in hospital.calls if c[0] == "cancel"]) == 1


def test_cancel_not_active_after_an_unknown_try_is_checked_not_trusted():
    hospital = FakeHospital(cancel=[UNKNOWN, definite("not_active")],
                            appointments=[[appt()], [appt()], [appt(status="cancelled")]])

    _, out = _run("cancel_appointment", {"appointment_id": "A1003"}, _ctx(hospital))

    assert (out["outcome"], out["verified"]) == ("success", True)


# ---------------------------------------------------------------- payment links

def test_payment_link_is_never_retried_after_an_unknown_result():
    hospital = FakeHospital(payment_link=[UNKNOWN, {"bill_id": "B1", "url": "https://pay.example/1"}])
    ctx = _ctx(hospital)

    _, out = _run("send_payment_link", {"bill_id": "B1"}, ctx)

    assert out["outcome"] == "unknown"
    assert len([c for c in hospital.calls if c[0] == "payment_link"]) == 1
    assert ctx.flags["handoff"] is True


def test_payment_link_retries_only_when_the_request_provably_never_left():
    link = {"bill_id": "B1", "url": "https://pay.example/1"}
    hospital = FakeHospital(payment_link=[NOT_SENT, NOT_SENT, link])

    _, out = _run("send_payment_link", {"bill_id": "B1"}, _ctx(hospital))

    assert (out["outcome"], out["link"]["url"]) == ("success", link["url"])
    assert len([c for c in hospital.calls if c[0] == "payment_link"]) == 3


def test_payment_link_that_never_reaches_the_hospital_is_a_definite_failure():
    hospital = FakeHospital(payment_link=[NOT_SENT])

    _, out = _run("send_payment_link", {"bill_id": "B1"}, _ctx(hospital))

    assert out["outcome"] == "failed"


def test_payment_link_for_a_paid_bill_is_a_definite_failure():
    hospital = FakeHospital(payment_link=[definite("already_paid")])

    _, out = _run("send_payment_link", {"bill_id": "B2"}, _ctx(hospital))

    assert (out["outcome"], out["code"]) == ("failed", "already_paid")


def test_the_start_time_helper_formats_in_hospital_time():
    from agent_service.tools import parse_start

    assert parse_start("2026-12-05T04:30:00+00:00") == datetime.fromisoformat(NEW)
