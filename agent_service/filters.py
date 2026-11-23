"""Two small safety nets on model replies. They catch obvious mistakes; they are not guarantees.

grounded_numbers (info agent): every amount, clock time and phone number in the reply must appear in
hospital_info.md. Cannot catch a wrong claim that has no number ("we accept Star Health insurance").
no_medical_advice (all agents): replaces replies containing dosages, drug names or "it's probably nothing"
style reassurance. Cannot catch advice in other words or other languages. Prompts and evals cover the rest.
"""

import re

from agentcore.hooks import Block
from agentcore.model import response_text

from .agents import HOSPITAL_INFO

SAFE_UNKNOWN_REPLY = "I don't have that information. Please call the front desk on 020-4000-1234 (Mon to Sat, 9am to 8pm) and they can help."
SAFE_ADVICE_REPLY = "I can't give medical advice, but I can book you with a doctor. Would you like that?"

_AMOUNT = re.compile(r"(?:rs\.?|inr|₹)\s?(\d[\d,]*)|(\d[\d,]*)\s?(?:rupees|rs\b)", re.IGNORECASE)
_TIME = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s?(am|pm)\b", re.IGNORECASE)
_PHONE = re.compile(r"\b\d{2,4}[-\s]\d{3,4}[-\s]\d{3,4}\b|\b\d{10}\b")


def _numbers(text: str) -> tuple[set[str], set[str], set[str]]:
    amounts = {(a or b).replace(",", "") for a, b in _AMOUNT.findall(text)}
    times = {f"{int(h)}:{m or '00'}{ap.lower()}" for h, m, ap in _TIME.findall(text)}
    phones = {re.sub(r"\D", "", p) for p in _PHONE.findall(text)}
    return amounts, times, phones


_KNOWN = _numbers(HOSPITAL_INFO)


def grounded_numbers(response, ctx):
    """after_model hook for the info agent."""
    found = _numbers(response_text(response))
    if any(not got <= known for got, known in zip(found, _KNOWN)):
        return Block(SAFE_UNKNOWN_REPLY)
    return None


_ADVICE = re.compile(
    r"\b\d+\s?(mg|ml|mcg)\b|paracetamol|ibuprofen|aspirin|crocin|dolo\b|antibiotic|azithromycin|amoxicillin|cetirizine|painkiller"
    r"|you (probably|likely|might|may) have|it(?:'s| is) (probably|likely|just)\b|nothing to worry|should be fine"
    r"|home remed|diagnos",
    re.IGNORECASE,
)


def no_medical_advice(response, ctx):
    """after_model hook for every agent."""
    if _ADVICE.search(response_text(response).replace("’", "'")):
        return Block(SAFE_ADVICE_REPLY)
    return None
