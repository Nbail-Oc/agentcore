"""Client for the hospital system. Classifies every failure by what we can know about the write.

HospitalError.kind:
  definite  the hospital answered 400, 404, 409 or 422 (its documented rejections): nothing was changed
  not_sent  the request provably never reached the hospital (connection refused or timed out connecting)
  unknown   anything else: timeout after sending, 5xx (a 503 can come after the write was saved), 408, 429,
            other 4xx, a redirect, 202/204, or an unreadable reply. None of these proves that nothing happened,
            and a 2xx other than 200/201 does not prove that the write was done.

GET requests are retried with backoff. Writes are never retried here: the tools decide, per write.
Ids are validated before they go into a URL path (httpx collapses "../", which would reach other routes).
"""

import os
import re
import time
from datetime import datetime
from typing import Any, Callable

import httpx

_ID = {"appointment": r"A\d{1,8}", "bill": r"B\d{1,8}", "doctor": r"d\d{1,4}", "date": r"\d{4}-\d{2}-\d{2}"}


class HospitalError(Exception):
    def __init__(self, kind: str, code: str | None = None, status: int | None = None, message: str = ""):
        super().__init__(f"{kind} {code or ''} {status or ''}".strip())
        self.kind, self.code, self.status, self.message = kind, code, status, message


def _safe(kind: str, value: str) -> str:
    if not re.fullmatch(_ID[kind], value or ""):
        raise ValueError(f"invalid {kind}")
    return value


def _text(value: Any) -> str:
    """Free text from the hospital is untrusted: one short plain line, no markup, no control characters."""
    return " ".join(re.sub("[" + chr(0) + "-" + chr(31) + chr(127) + "<>]+", " ", str(value)).split())[:80]


def _own(items: list, phone: str, kind: str) -> list[dict]:
    """The hospital is untrusted. Other patients' records are dropped. A record that may be ours but cannot be
    read (unknown shape, id or owner) makes the whole read fail: an incomplete list is never shown as complete."""
    mine = []
    for i in items:
        if isinstance(i, dict) and isinstance(i.get("phone"), str) and i["phone"] != phone:
            continue
        if not (isinstance(i, dict) and i.get("phone") == phone and isinstance(i.get("id"), str) and re.fullmatch(_ID[kind], i["id"])):
            raise HospitalError("unknown", None, 200)
        mine.append(i)
    return [{**i, "description": _text(i["description"])} if "description" in i else i for i in mine]


class Hospital:
    def __init__(self, base_url: str | None = None, client: httpx.Client | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self._client = client or httpx.Client(
            base_url=base_url or os.getenv("HOSPITAL_URL", "http://localhost:8001"),
            timeout=httpx.Timeout(8.0, connect=3.0),
        )
        self._sleep = sleep

    def _once(self, method: str, path: str, params: dict | None, body: dict | None) -> Any:
        try:
            response = self._client.request(method, path, params=params, json=body)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
            raise HospitalError("not_sent") from None
        except httpx.HTTPError:
            raise HospitalError("unknown") from None
        try:
            data = response.json()
        except ValueError:
            data = None
        error = data.get("error") if isinstance(data, dict) else None
        if not (isinstance(error, str) and re.fullmatch(r"[a-z_]{1,40}", error)):
            error = None  # only a code shaped like the documented ones is kept
        message = ""  # the hospital's free text is untrusted and is never passed on (to the model or the patient)
        status = response.status_code
        if status in (400, 404, 409, 422):
            raise HospitalError("definite", error, status, message)
        if status not in (200, 201) or not isinstance(data, (dict, list)):
            raise HospitalError("unknown", error, status, message)
        return data

    def _request(self, method: str, path: str, params: dict | None = None, body: dict | None = None) -> Any:
        attempts = 3 if method == "GET" else 1
        for i in range(attempts):
            try:
                return self._once(method, path, params, body)
            except HospitalError as e:
                if e.kind == "definite" or i == attempts - 1:
                    raise
                self._sleep(0.3 * 2**i)

    def _field(self, path: str, key: str, kind: type, params: dict | None = None) -> Any:
        """A reply that lacks the field we need is an unusable reply (unknown), never a crash."""
        data = self._request("GET", path, params)
        value = data.get(key) if isinstance(data, dict) else None
        if not isinstance(value, kind):
            raise HospitalError("unknown", None, 200)
        return value

    def now(self) -> datetime:
        try:
            return datetime.fromisoformat(self._field("/now", "now", str))
        except ValueError:
            raise HospitalError("unknown", None, 200) from None

    def doctors(self, specialty: str | None = None) -> dict:
        doctors = self._field("/doctors", "doctors", list, {"specialty": specialty} if specialty else None)
        return {"doctors": [{**d, "name": _text(d.get("name", "")), "specialty": _text(d.get("specialty", ""))}
                            for d in doctors if isinstance(d, dict) and isinstance(d.get("id"), str)
                            and re.fullmatch(_ID["doctor"], d["id"])]}

    def slots(self, doctor_id: str, date: str) -> dict:
        return self._request("GET", f"/doctors/{_safe('doctor', doctor_id)}/slots", {"date": _safe("date", date)})

    def appointments(self, phone: str) -> list[dict]:
        return _own(self._field("/appointments", "appointments", list, {"phone": phone}), phone, "appointment")

    def bills(self, phone: str) -> list[dict]:
        return _own(self._field("/bills", "bills", list, {"phone": phone}), phone, "bill")

    def book(self, doctor_id: str, start: str, patient_name: str, phone: str, idempotency_key: str) -> dict:
        return self._request("POST", "/appointments", body={
            "doctor_id": _safe("doctor", doctor_id), "start": start, "patient_name": patient_name,
            "phone": phone, "idempotency_key": idempotency_key,
        })

    def reschedule(self, appointment_id: str, new_start: str) -> dict:
        return self._request("POST", f"/appointments/{_safe('appointment', appointment_id)}/reschedule",
                             body={"new_start": new_start})

    def cancel(self, appointment_id: str) -> dict:
        return self._request("POST", f"/appointments/{_safe('appointment', appointment_id)}/cancel", body={})

    def payment_link(self, bill_id: str) -> dict:
        return self._request("POST", f"/bills/{_safe('bill', bill_id)}/payment-link", body={})
