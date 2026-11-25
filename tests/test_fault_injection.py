"""Fault injection through /chat against the real mock hospital (only the model is scripted). Offline.

The mock's chaos switches and admin state are used the way the eval runner uses them, to check what really
happened to the hospital's records. Fake-model evidence only: it shows the code paths, not real Claude behaviour.
"""

from agent_service.orchestrator import handle_chat
from agent_service.schemas import ChatRequest
from tests.fake_claude import text, tool_use
from tests.test_against_mock import CLEAR, YES, appts, chat, deps, fresh_mock, mock, route  # noqa: F401  (fresh_mock resets the mock)

PHONE = "9988776655"  # A1003 with Dr. Khan, Thu 3 Dec 17:30 (more than 4 hours away)
CANCEL = tool_use("cancel_appointment", {"appointment_id": "A1003"})
ASK = [CLEAR, route("booking"), CANCEL, text("Cancel it?")]


def status():
    return appts(PHONE)[0]["status"], appts(PHONE)[0]["start"]


def test_a_cancel_saved_before_the_503_is_reported_as_cancelled_once():
    d = deps(ASK + [CLEAR, YES])
    chat(d, "please cancel my appointment", PHONE)
    mock.post("/_admin/chaos", json={"commit_then_fail_next": 1})

    res = chat(d, "Yes, cancel it.", PHONE)

    assert status()[0] == "cancelled" and "cancelled" in res.reply.lower() and res.handoff is False


def test_when_hospital_reads_fail_nothing_is_proposed_or_changed():
    mock.post("/_admin/chaos", json={"error_rate": 1.0})
    d = deps([CLEAR, route("booking"), CANCEL, text("I can't check that right now.")])

    res = chat(d, "please cancel my appointment", PHONE)
    mock.post("/_admin/chaos", json={})

    assert status()[0] == "booked" and "Details:" not in res.reply and d.store.load("s")["pending"] is None


def test_details_that_changed_after_the_read_back_do_not_inherit_the_yes():
    d = deps(ASK + [CLEAR, YES, text("That appointment has changed. Do you still want to cancel it?")])
    chat(d, "please cancel my appointment", PHONE)
    free = mock.get("/doctors/d5/slots", params={"date": "2026-12-05"}).json()["free"]
    mock.post("/appointments/A1003/reschedule", json={"new_start": free[0]})  # staff moved it before the yes

    res = chat(d, "Yes, cancel it.", PHONE)

    assert status()[0] == "booked" and status()[1][:10] == "2026-12-05" and "is cancelled" not in res.reply


def test_an_emergency_while_a_cancel_is_waiting_cancels_nothing_and_a_later_yes_does_nothing():
    d = deps(ASK + [CLEAR, route("booking"), text("What would you like to do?")])
    chat(d, "please cancel my appointment", PHONE)

    emergency = chat(d, "I have severe chest pain and cannot breathe", PHONE)
    later = chat(d, "Yes.", PHONE)

    assert (emergency.agent, emergency.emergency, emergency.handoff) == ("safety", True, True)
    assert status()[0] == "booked" and "cancelled" not in later.reply.lower()


def test_two_yes_messages_in_a_row_cancel_once_and_the_second_changes_nothing():
    d = deps(ASK + [CLEAR, YES, CLEAR, route("booking"), text("It is already cancelled.")])
    chat(d, "please cancel my appointment", PHONE)
    chat(d, "Yes, cancel it.", PHONE)
    before = mock.get("/_admin/state").json()

    chat(d, "Yes.", PHONE)

    assert mock.get("/_admin/state").json() == before and status()[0] == "cancelled"
