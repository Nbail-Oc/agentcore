"""Tests for the hospital HTTP client: error classification, safe URLs, read retries. Offline (MockTransport)."""

from pathlib import Path

import httpx
import pytest

from agent_service.hospital import Hospital, HospitalError


def _hospital(handler, sleeps=None):
    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://hospital.test")
    return Hospital(client=client, sleep=(sleeps.append if sleeps is not None else lambda s: None))


def _json(status, body):
    return httpx.Response(status, json=body)


def test_book_sends_the_idempotency_key_and_returns_the_booking():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path, request.content))
        return _json(201, {"id": "A1", "doctor_id": "d1", "start": "2026-12-04T17:00:00+05:30"})

    appt = _hospital(handler).book("d1", "2026-12-04T17:00:00+05:30", "Rohan", "9000000001", "key-1")

    assert appt["id"] == "A1"
    method, path, body = seen[0]
    assert (method, path) == ("POST", "/appointments")
    assert b'"idempotency_key":"key-1"' in body.replace(b" ", b"")


def test_a_4xx_with_an_error_code_is_a_definite_failure():
    hospital = _hospital(lambda r: _json(409, {"error": "slot_taken", "message": "That slot was just taken."}))

    with pytest.raises(HospitalError) as info:
        hospital.book("d1", "2026-12-04T17:00:00+05:30", "Rohan", "9000000001", "k")

    assert (info.value.kind, info.value.code, info.value.status) == ("definite", "slot_taken", 409)


def test_503_on_a_write_is_unknown_because_the_write_may_have_been_saved():
    hospital = _hospital(lambda r: _json(503, {"error": "service_unavailable", "message": "later"}))

    with pytest.raises(HospitalError) as info:
        hospital.cancel("A1")

    assert info.value.kind == "unknown"


def test_transport_failures_are_classified_by_whether_the_request_could_have_arrived():
    def refuse(request):
        raise httpx.ConnectError("refused")

    def read_timeout(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(HospitalError) as not_sent:
        _hospital(refuse).cancel("A1")
    with pytest.raises(HospitalError) as unknown:
        _hospital(read_timeout).cancel("A1")

    assert not_sent.value.kind == "not_sent"
    assert unknown.value.kind == "unknown"


def test_a_200_with_a_body_that_is_not_json_is_unknown():
    hospital = _hospital(lambda r: httpx.Response(200, text="<html>proxy error</html>"))

    with pytest.raises(HospitalError) as info:
        hospital.cancel("A1")

    assert info.value.kind == "unknown"


def test_reads_retry_with_backoff_and_then_succeed():
    calls, sleeps = [], []

    def handler(request):
        calls.append(1)
        return _json(503, {"error": "service_unavailable"}) if len(calls) < 3 else _json(200, {"appointments": []})

    assert _hospital(handler, sleeps).appointments("9000000001") == []
    assert len(calls) == 3
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0]


def test_reads_give_up_after_three_attempts():
    calls = []

    def handler(request):
        calls.append(1)
        return _json(503, {"error": "service_unavailable"})

    with pytest.raises(HospitalError):
        _hospital(handler).appointments("9000000001")

    assert len(calls) == 3


def test_writes_are_never_retried_by_the_client():
    calls = []

    def handler(request):
        calls.append(1)
        return _json(503, {"error": "service_unavailable"})

    with pytest.raises(HospitalError):
        _hospital(handler).payment_link("B1")

    assert len(calls) == 1


@pytest.mark.parametrize("bad_id", ["../_admin/reset", "A1/../../_admin/reset", "a1", "A", "A1;x", "A1 ", "%2e%2e", ""])
def test_ids_that_could_change_the_url_path_are_rejected_before_any_request(bad_id):
    sent = []
    hospital = _hospital(lambda r: sent.append(r) or _json(200, {}))

    for call in (hospital.cancel, lambda i: hospital.reschedule(i, "2026-12-04T17:00:00+05:30"), hospital.payment_link):
        with pytest.raises(ValueError):
            call(bad_id)
    with pytest.raises(ValueError):
        hospital.slots(bad_id, "2026-12-04")

    assert sent == []


def test_httpx_really_does_collapse_dot_segments_so_validation_is_needed():
    # Reproduced: the starter client would have sent this id straight into the path.
    path = httpx.Request("POST", "http://h/appointments/../_admin/reset/cancel").url.path

    assert path == "/_admin/reset/cancel"


def test_now_returns_an_aware_datetime():
    hospital = _hospital(lambda r: _json(200, {"now": "2026-12-01T09:00:00+05:30", "weekday": "Tuesday"}))

    now = hospital.now()

    assert now.utcoffset() is not None and (now.year, now.hour) == (2026, 9)


def test_agent_code_never_mentions_the_admin_routes():
    for path in Path(__file__).parent.parent.glob("agent_service/*.py"):
        assert "_admin" not in path.read_text(encoding="utf-8"), path.name
