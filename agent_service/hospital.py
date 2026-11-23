"""Client for the hospital system. Classifies every failure by what we can know about the write.

HospitalError.kind:
  definite  the hospital answered with a 4xx error code: nothing was changed
  not_sent  the request provably never reached the hospital (connection refused or timed out connecting)
  unknown   timeout after sending, 5xx (a 503 can come after the write was saved), or an unreadable reply

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
        message = str(data.get("message", ""))[:200] if isinstance(data, dict) else ""
        if response.status_code >= 500:
            raise HospitalError("unknown", error, response.status_code, message)
        if response.status_code >= 400:
            raise HospitalError("definite", error, response.status_code, message)
        if not isinstance(data, (dict, list)):
            raise HospitalError("unknown", None, response.status_code)
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

    def now(self) -> datetime:
        return datetime.fromisoformat(self._request("GET", "/now")["now"])

    def doctors(self, specialty: str | None = None) -> dict:
        return self._request("GET", "/doctors", {"specialty": specialty} if specialty else None)

    def slots(self, doctor_id: str, date: str) -> dict:
        return self._request("GET", f"/doctors/{_safe('doctor', doctor_id)}/slots", {"date": _safe("date", date)})

    def appointments(self, phone: str) -> list[dict]:
        return self._request("GET", "/appointments", {"phone": phone})["appointments"]

    def bills(self, phone: str) -> list[dict]:
        return self._request("GET", "/bills", {"phone": phone})["bills"]

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
