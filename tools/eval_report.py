"""Summarize an eval results file: pass/fail per scenario, failures, time, and the measured agent cost.

    uv run python tools/eval_report.py evals/results/<file>.json [more files...]

Cost comes from GET /trace/{session_id} on the running agent (AGENT_URL), so run it while the agent is still up
and before it restarts: traces live in memory. Each model call's cost is computed from the API's token counts.
"""

import json
import os
import sys

import httpx

AGENT = os.getenv("AGENT_URL", "http://localhost:8000")


def load(path: str) -> list[dict]:
    data = json.load(open(path, encoding="utf-8"))
    return data["results"] if isinstance(data, dict) and "results" in data else data


def main(paths: list[str]) -> None:
    grand_cost = 0.0
    for path in paths:
        results = load(path)
        passed = sum(1 for r in results if r["passed"])
        run_cost, model_calls, retries = 0.0, 0, 0
        print(f"\n{path}: {passed}/{len(results)} passed ({100 * passed // max(len(results), 1)}%)")
        for r in results:
            trace = httpx.get(f"{AGENT}/trace/{r['session_id']}", timeout=10)
            cost = trace.json()["total_cost_usd"] if trace.status_code == 200 else float("nan")
            if trace.status_code == 200:
                spans = [s for t in trace.json()["traces"] for s in t["spans"] if s["kind"] == "model"]
                model_calls += len(spans)
                retries += sum(1 for s in spans if s["is_error"])
                run_cost += cost
            print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['id']:<34} {r['seconds']:>6}s  ${cost:.5f}")
            for failure in r["failures"]:
                print(f"        - {failure}")
        print(f"  measured agent cost: ${run_cost:.4f} over {model_calls} model calls ({retries} failed attempts)")
        grand_cost += run_cost
    print(f"\nmeasured agent cost, all files: ${grand_cost:.4f}")


if __name__ == "__main__":
    main(sys.argv[1:])
