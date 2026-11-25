"""Tests for the two small reply filters: numbers must come from hospital_info.md, and no medical advice.

They are safety nets, not guarantees. The tests show what they catch and, on purpose, what they do not.
"""

import json
from datetime import datetime, timezone

import pytest

from agent_service.filters import (
    SAFE_ADVICE_REPLY, SAFE_UNKNOWN_REPLY, grounded_numbers, grounded_topics, no_medical_advice, unsupported_topics,
)
from agent_service.orchestrator import Deps, handle_chat
from agent_service.schemas import ChatRequest
from agentcore.hooks import Block
from agentcore.session import InMemorySessionStore
from tests.fake_claude import FakeClaude, text
from tests.fake_hospital import FakeHospital


def reply(message):
    return text(message)


@pytest.mark.parametrize("message", [
    "The consultation fee is Rs 500 for every doctor.", "It is ₹500.", "That is 500 rupees.",
    "We are open 10am to 1pm and 5pm to 8pm, Monday to Saturday.", "The pharmacy is open 8am to 10pm every day.",
    "Call the front desk on 020-4000-1234 (9am to 8pm).", "Please arrive 15 minutes early.",
    "Sample collection is 8am to 12pm.", "Dr. Anjali Mehta and Dr. Vikram Rao see skin patients.",
    "Payment links are valid for 24 hours.", "I'm not sure, I don't have that information.", "",
])
def test_info_replies_with_numbers_from_the_hospital_file_pass(message):
    assert grounded_numbers(reply(message), None) is None


@pytest.mark.parametrize("message", [
    "The consultation fee is Rs 700.", "Consultation costs ₹1,000.", "That is 800 rupees.",
    "We are open 9am to 9pm.", "Pharmacy closes at 11 pm.", "Call us on 020-9999-0000.", "Call 9876543210 for details.",
])
def test_info_replies_with_numbers_not_in_the_file_are_replaced(message):
    decision = grounded_numbers(reply(message), None)

    assert isinstance(decision, Block) and decision.reason == SAFE_UNKNOWN_REPLY


@pytest.mark.parametrize("message", [
    "Yes, we accept Star Health insurance with cashless.", "Your mediclaim will cover it.", "Cashless treatment is available.",
    "Coverage depends on your TPA.", "Insurance covers up to Rs 5000.",
])
def test_claims_about_topics_the_file_does_not_mention_are_replaced_with_or_without_numbers(message):
    decision = grounded_topics(reply(message), None)

    assert isinstance(decision, Block) and decision.reason == SAFE_UNKNOWN_REPLY


def test_a_topic_the_file_does_mention_is_allowed():
    assert unsupported_topics("We accept insurance, cashless too.", info="Insurance and cashless are accepted.") == set()
    assert unsupported_topics("We accept insurance, cashless too.", info="Fees: Rs 500.") == {"insurance", "cashless"}


def test_known_limit_unsupported_claims_in_other_words_are_not_caught():
    assert grounded_topics(reply("Yes, we work with your provider, no need to pay upfront."), None) is None


@pytest.mark.parametrize("message", [
    "Take paracetamol 500 mg twice a day.", "You should take an antibiotic.", "It's probably nothing.",
    "It is likely just a cold.", "Nothing to worry about.", "You might have an infection.", "Try this home remedy.",
    "Take some ibuprofen for the pain.", "You probably have the flu.", "Drink warm water and rest, you should be fine.",
])
def test_medical_advice_is_replaced(message):
    decision = no_medical_advice(reply(message), None)

    assert isinstance(decision, Block) and decision.reason == SAFE_ADVICE_REPLY


@pytest.mark.parametrize("message", [
    "You have an appointment with Dr. Iyer on Friday at 5pm.", "Your bill is Rs 500.", "I can book you with a doctor.",
    "I can't give medical advice, but I can book a doctor for you.", "Shall I book a general medicine doctor?",
    "Please bring your old reports.", "",
])
def test_ordinary_front_desk_replies_are_left_alone(message):
    assert no_medical_advice(reply(message), None) is None


def test_known_limit_advice_in_other_words_is_not_caught():
    assert no_medical_advice(reply("Cold compress and lots of sleep usually helps."), None) is None


# ---------------------------------------------------------------- wired into the agents

def _chat(model_turns):
    deps = Deps(client=FakeClaude([text('{"emergency": false}'), *model_turns]),
                hospital=FakeHospital(now=[datetime(2026, 12, 1, 9, 0, tzinfo=timezone.utc)]),
                store=InMemorySessionStore(), sleep=lambda s: None)
    return handle_chat(ChatRequest(session_id="s", phone="9988776655", message="hello"), deps)


def test_the_info_agent_cannot_state_a_fee_that_is_not_in_the_file():
    res = _chat([text(json.dumps({"route": "info"})), text("The consultation fee is Rs 700.")])

    assert res.agent == "info" and res.reply == SAFE_UNKNOWN_REPLY


def test_the_booking_agent_cannot_give_medical_advice():
    res = _chat([text(json.dumps({"route": "booking"})), text("It's probably nothing, take paracetamol.")])

    assert res.agent == "booking" and res.reply == SAFE_ADVICE_REPLY


def test_the_info_agent_cannot_claim_insurance_that_is_not_in_the_file():
    res = _chat([text(json.dumps({"route": "info"})), text("Yes, we accept Star Health insurance with cashless.")])

    assert res.agent == "info" and res.reply == SAFE_UNKNOWN_REPLY
