"""Fixed patient-facing text written by code (never by the model): write outcomes, failures, prompts.

Outcomes come from the verified structured result of a write, so the patient can never be told something
happened that did not. Languages: en, hinglish (Roman Hindi), hi (Devanagari). Names and dates stay in English.
"""

import re

from .hooks import when
from .tools import parse_start

_HINGLISH = {
    "haan", "han", "nahi", "nahin", "kar", "karo", "kardo", "mujhe", "mera", "meri", "mere", "hai", "hain", "hoon", "chahiye",
    "kal", "aaj", "aap", "ko", "mein", "me", "ka", "ki", "ke", "kya", "kaise", "bhai", "ji", "theek", "thik", "accha",
    "abhi", "dard", "saans", "bataiye", "batao", "karna", "lena", "dena", "raha", "rahi", "ho",
}


def detect_language(message: str) -> str:
    if re.search(r"[ऀ-ॿ]", message):
        return "hi"
    words = re.findall(r"[a-z']+", message.lower())
    return "hinglish" if sum(w in _HINGLISH for w in words) >= 2 else "en"


LEAD = {"en": "Done!", "hinglish": "Ho gaya!", "hi": "हो गया!"}
NOT_DONE = {"en": "This was NOT done", "hinglish": "Yeh NAHI hua", "hi": "यह नहीं हो पाया"}
UNKNOWN = {
    "en": "I couldn't confirm whether this went through, so please don't assume it did. Our staff will check and contact you.",
    "hinglish": "Mujhe confirm nahi ho paya ki yeh hua ya nahi, isliye ise pura na maanein. Hamara staff check karke aapko batayega.",
    "hi": "मैं पुष्टि नहीं कर पाया कि यह हुआ या नहीं, कृपया इसे पूरा न मानें। हमारा स्टाफ जाँच कर आपसे संपर्क करेगा।",
}
RETRY = {
    "en": "Sorry, I'm having trouble right now. Please try again in a moment.",
    "hinglish": "Maaf kijiye, abhi dikkat aa rahi hai. Kripya thodi der mein dobara try karein.",
    "hi": "क्षमा करें, अभी समस्या आ रही है। कृपया थोड़ी देर में फिर कोशिश करें।",
}
HANDOFF = {
    "en": "Sorry, I'm still having trouble. I'm handing this chat to our staff, who will reply soon.",
    "hinglish": "Maaf kijiye, abhi bhi dikkat hai. Main yeh chat hamare staff ko de raha hoon, woh jaldi reply karenge.",
    "hi": "क्षमा करें, अभी भी समस्या है। मैं यह चैट हमारे स्टाफ को दे रहा हूँ, वे जल्द जवाब देंगे।",
}
HUMAN_FALLBACK = {
    "en": "I'm connecting you with our staff, who will reply soon.",
    "hinglish": "Main aapko hamare staff se jod raha hoon, woh jaldi reply karenge.",
    "hi": "मैं आपको हमारे स्टाफ से जोड़ रहा हूँ, वे जल्द जवाब देंगे।",
}
CONFIRM = {"en": "Reply YES to confirm.", "hinglish": "Confirm karne ke liye YES likhein.", "hi": "पुष्टि के लिए हाँ लिखें।"}
DETAILS = {"en": "Details", "hinglish": "Details", "hi": "विवरण"}
AFTER_FAILURE = "Please tell me how you'd like to go ahead."

_REASONS = {
    "slot_taken": "that slot was just taken",
    "invalid_slot": "that time is not available",
    "not_active": "the appointment is no longer active",
    "already_paid": "the bill is already paid",
    "appointment_not_found": "the appointment was not found",
    "bill_not_found": "the bill was not found",
    "doctor_not_found": "the doctor was not found",
    "unreachable": "the hospital system could not be reached",
}
_SUCCESS_WORDS = re.compile(r"\b(booked|confirmed|scheduled|rescheduled|cancelled|canceled|done|sent)\b", re.IGNORECASE)


def claims_success(text: str) -> bool:
    return bool(_SUCCESS_WORDS.search(text))


def failure_line(outcome: dict, lang: str) -> str:
    return f"{NOT_DONE[lang]}: {_REASONS.get(outcome.get('code'), 'the hospital system refused it')}."


def outcome_text(outcome: dict, lang: str) -> str:
    """The reply for a write whose result is known. Only for success and unknown outcomes."""
    if outcome["outcome"] == "unknown":
        return UNKNOWN[lang]
    if outcome["action"] == "payment_link_sent":
        link = outcome["link"]
        body = f"Payment link for bill {link['bill_id']}: {link['url']}"
    else:
        a = outcome["appointment"]
        at = when(parse_start(a["start"]))
        body = {
            "booked": f"Appointment booked: {a['doctor']}, {at} (ref {a['id']}).",
            "rescheduled": f"Appointment {a['id']} moved: {a['doctor']}, {at}.",
            "cancelled": f"Appointment {a['id']} with {a['doctor']} on {at} is cancelled.",
        }[outcome["action"]]
    return f"{LEAD[lang]} {body}"
