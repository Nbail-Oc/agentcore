"""Patient-facing text must not claim staff were contacted: there is only a handoff flag and no staff system."""

import re
from pathlib import Path

from agent_service import agents, emergency, filters, replies

PROMISES = re.compile(
    r"staff (will|shall)|will (contact|reply|respond|call|follow|check|help|take over)|alerting|connecting (you|them)|get back to you"
    r"|स्टाफ आपसे संपर्क|जवाब देंगे|बताएगा|सूचित कर रहे|jaldi reply|batayega|reply karenge|jod raha",
    re.IGNORECASE,
)
FRONT_DESK = "020-4000-1234"


def _patient_facing():
    for table in (replies.UNKNOWN, replies.HANDOFF, replies.HUMAN_FALLBACK):
        yield from table.values()
    yield from (emergency.SAFETY_REPLY, emergency.UNVERIFIED_REPLY, filters.SAFE_UNKNOWN_REPLY)


def test_no_fixed_reply_promises_that_staff_will_do_anything():
    for text in _patient_facing():
        assert not PROMISES.search(text), text


def test_prompts_and_tool_instructions_do_not_tell_the_model_to_promise_it_either():
    src = Path(agents.__file__).parent
    for text in (agents.COMMON, *agents.PROMPTS.values(), (src / "tools.py").read_text(encoding="utf-8"),
                 (src / "hooks.py").read_text(encoding="utf-8")):
        found = PROMISES.search(text)
        assert not found, found.group(0)


def test_handoff_replies_say_the_chat_is_flagged_and_give_the_front_desk_number_from_hospital_info():
    assert FRONT_DESK in agents.HOSPITAL_INFO
    for table in (replies.UNKNOWN, replies.HANDOFF, replies.HUMAN_FALLBACK):
        assert all(FRONT_DESK in table[lang] for lang in ("en", "hinglish", "hi"))
    assert "flagged" in replies.UNKNOWN["en"] and "flagged" in replies.HUMAN_FALLBACK["en"]
    assert FRONT_DESK in filters.SAFE_UNKNOWN_REPLY
