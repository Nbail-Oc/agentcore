"""The eval scenarios are valid and use only check types the runner understands (no API key needed)."""

from pathlib import Path

from evals.run_evals import load_scenarios, run_check

FOLDER = Path(__file__).parent.parent / "evals" / "scenarios"
EMPTY_STATE = {"appointments": [], "bills": [], "payment_links": []}
SCENARIOS = load_scenarios(None, FOLDER)


def test_the_ten_given_scenarios_are_there_and_at_least_three_new_ones_were_added():
    numbers = sorted(int(p.name[:2]) for p in FOLDER.glob("*.yaml"))

    assert numbers[:10] == list(range(1, 11))
    assert len([n for n in numbers if n >= 11]) >= 3


def test_scenario_ids_are_unique_and_every_scenario_has_what_the_runner_needs():
    ids = [s["id"] for s in SCENARIOS]

    assert len(ids) == len(set(ids))
    for s in SCENARIOS:
        assert s["phone"] and s["turns"] and s["checks"] and s["description"], s["id"]


def test_every_check_type_is_known_to_the_runner():
    responses = [{"reply": "", "agent": "x", "handoff": False, "emergency": False}] * 5
    for s in SCENARIOS:
        for check in s["checks"]:
            assert not str(run_check(check, EMPTY_STATE, responses)).startswith("unknown check type"), (s["id"], check)
