"""Tests for the session store interface and its in-memory implementation."""

import threading

from agentcore.session import InMemorySessionStore


def test_first_owner_wins_and_others_are_refused():
    store = InMemorySessionStore()

    assert store.bind("s1", "alice") is True
    assert store.bind("s1", "alice") is True
    assert store.bind("s1", "bob") is False
    assert store.owner("s1") == "alice"


def test_concurrent_first_binds_have_exactly_one_winner():
    for trial in range(30):
        store, results = InMemorySessionStore(), []
        barrier = threading.Barrier(8)

        def claim(owner):
            barrier.wait()
            results.append((owner, store.bind("s", owner)))

        threads = [threading.Thread(target=claim, args=(f"p{i}",)) for i in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        winners = {owner for owner, ok in results if ok}
        assert len(winners) == 1 and store.owner("s") in winners


def test_load_and_save_round_trip_copies_not_references():
    store = InMemorySessionStore()
    data = {"history": ["hi"]}

    store.save("s", data)
    data["history"].append("mutated after save")
    loaded = store.load("s")
    loaded["history"].append("mutated after load")

    assert store.load("s") == {"history": ["hi"]}
    assert store.load("missing") is None


def test_lock_serializes_work_on_one_session_but_not_across_sessions():
    store, order = InMemorySessionStore(), []
    entered_second = threading.Event()

    def first():
        with store.lock("s"):
            order.append("first in")
            entered_second.wait(timeout=0.3)  # second thread must NOT get in meanwhile
            order.append("first out")

    def second():
        with store.lock("s"):
            order.append("second in")
            entered_second.set()

    t1, t2 = threading.Thread(target=first), threading.Thread(target=second)
    t1.start()
    while "first in" not in order:
        pass
    t2.start()
    t1.join()
    t2.join()

    assert order == ["first in", "first out", "second in"]

    def other_session():
        with store.lock("b"):
            order.append("b done")

    with store.lock("a"):
        t = threading.Thread(target=other_session)
        t.start()
        t.join(timeout=1)
        assert not t.is_alive()  # a different session is never blocked by "a"
    assert order[-1] == "b done"


def test_traces_are_kept_per_session_in_order():
    store = InMemorySessionStore()

    store.add_trace("s", {"id": "t1"})
    store.add_trace("s", {"id": "t2"})

    assert [t["id"] for t in store.traces("s")] == ["t1", "t2"]
    assert store.traces("other") == []


def test_sessions_do_not_see_each_others_data_owner_or_traces():
    store = InMemorySessionStore()
    store.bind("a", "1111")
    store.bind("b", "2222")
    store.save("a", {"secret": "a-only"})
    store.add_trace("a", {"id": "t"})

    assert store.load("b") is None and store.traces("b") == []
    assert (store.owner("a"), store.owner("b")) == ("1111", "2222")
    assert store.lock("a") is not store.lock("b")
