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
    "en": "I couldn't confirm whether this went through, so please don't assume it did. I've flagged this chat for hospital staff. "
          "To check sooner, call the front desk on 020-4000-1234 (Mon to Sat, 9am to 8pm).",
    "hinglish": "Mujhe confirm nahi ho paya ki yeh hua ya nahi, isliye ise pura na maanein. Maine yeh chat hospital staff ke liye flag kar di hai. "
                "Jaldi check karne ke liye front desk ko 020-4000-1234 par call karein (Mon-Sat, 9am-8pm).",
    "hi": "मैं पुष्टि नहीं कर पाया कि यह हुआ या नहीं, कृपया इसे पूरा न मानें। मैंने यह चैट अस्पताल स्टाफ के लिए चिह्नित कर दी है। "
          "जल्दी जाँच के लिए फ्रंट डेस्क को 020-4000-1234 पर कॉल करें (सोम से शनि, सुबह 9 से रात 8)।",
}
RETRY = {
    "en": "Sorry, I'm having trouble right now. Please try again in a moment.",
    "hinglish": "Maaf kijiye, abhi dikkat aa rahi hai. Kripya thodi der mein dobara try karein.",
    "hi": "क्षमा करें, अभी समस्या आ रही है। कृपया थोड़ी देर में फिर कोशिश करें।",
}
HANDOFF = {
    "en": "Sorry, I'm still having trouble. I've flagged this chat for hospital staff. You can also call the front desk on 020-4000-1234 (Mon to Sat, 9am to 8pm).",
    "hinglish": "Maaf kijiye, abhi bhi dikkat hai. Maine yeh chat hospital staff ke liye flag kar di hai. Aap front desk ko 020-4000-1234 par bhi call kar sakte hain (Mon-Sat, 9am-8pm).",
    "hi": "क्षमा करें, अभी भी समस्या है। मैंने यह चैट अस्पताल स्टाफ के लिए चिह्नित कर दी है। आप फ्रंट डेस्क को 020-4000-1234 पर भी कॉल कर सकते हैं (सोम से शनि, सुबह 9 से रात 8)।",
}
HUMAN_FALLBACK = {
    "en": "I've flagged this chat for hospital staff. If it's urgent, please call the front desk on 020-4000-1234 (Mon to Sat, 9am to 8pm).",
    "hinglish": "Maine yeh chat hospital staff ke liye flag kar di hai. Zaroori ho to front desk ko 020-4000-1234 par call karein (Mon-Sat, 9am-8pm).",
    "hi": "मैंने यह चैट अस्पताल स्टाफ के लिए चिह्नित कर दी है। ज़रूरी हो तो फ्रंट डेस्क को 020-4000-1234 पर कॉल करें (सोम से शनि, सुबह 9 से रात 8)।",
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
