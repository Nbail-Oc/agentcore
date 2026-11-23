"""An earlier ambiguous request may still complete later: never report a definite failure unless something proves it."""

from agent_service.tools import build_registry
from tests.fake_hospital import NEW, OLD, UNKNOWN, FakeHospital, _ctx, appt, definite
import json

BOOK = {"doctor_id": "d5", "start": NEW, "patient_name": "Sameer Kulkarni"}


def _run(name, args, ctx):
    result = build_registry().execute(name, args, ctx)
    return result, (json.loads(result.content) if not result.is_error else None)


def test_a_definite_error_after_an_unknown_try_is_not_reported_as_failed_when_nothing_proves_it():
    for name, args, script in (
        ("reschedule_appointment", {"appointment_id": "A1003", "new_start": NEW},
         {"reschedule": [UNKNOWN, definite("slot_taken")], "appointments": [[appt(OLD)]]}),
        ("cancel_appointment", {"appointment_id": "A1003"},
         {"cancel": [UNKNOWN, definite("not_active")], "appointments": [[appt(OLD)]]}),
        ("book_appointment", BOOK, {"book": [UNKNOWN, definite("slot_taken")], "appointments": [[]]}),
    ):
        ctx = _ctx(FakeHospital(**script))

        _, out = _run(name, args, ctx)

        assert out["outcome"] == "unknown", name  # the first request may still be on its way
        assert ctx.flags["handoff"] is True


def test_a_booking_that_shows_up_after_a_later_definite_error_is_success():
    hospital = FakeHospital(book=[UNKNOWN, definite("slot_taken")], appointments=[[appt(NEW)]])

    _, out = _run("book_appointment", BOOK, _ctx(hospital))

    assert (out["outcome"], out["verified"]) == ("success", True)


class DelayedReschedule(FakeHospital):
    """The first reschedule times out but is applied a little later: it shows up from the Nth look onward."""

    def __init__(self, lands_on_look, **script):
        super().__init__(**script)
        self.looks, self.lands_on_look = 0, lands_on_look

    def appointments(self, phone):
        self.looks += 1
        return [appt(NEW if self.looks >= self.lands_on_look else OLD)]


def test_a_reschedule_that_lands_before_the_recheck_is_found_without_a_second_write():
    hospital = DelayedReschedule(2, reschedule=[UNKNOWN])  # look 1 sees nothing, look 2 (after the wait) sees it

    _, out = _run("reschedule_appointment", {"appointment_id": "A1003", "new_start": NEW}, _ctx(hospital))

    assert (out["outcome"], out["verified"]) == ("success", True)
    assert len([c for c in hospital.calls if c[0] == "reschedule"]) == 1


def test_a_reschedule_that_lands_only_after_everything_was_checked_stays_unknown_not_failed():
    hospital = DelayedReschedule(99, reschedule=[UNKNOWN])  # never visible while we look; the retry also times out

    _, out = _run("reschedule_appointment", {"appointment_id": "A1003", "new_start": NEW}, _ctx(hospital))

    assert out["outcome"] == "unknown"
