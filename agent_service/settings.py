"""Model choices and call settings for the hospital agents."""

import os

from agentcore.model import CallConfig

ROUTER_MODEL = "claude-haiku-5-5"  # required by the assignment; also used for the small yes/no checks
SPECIALIST_MODEL = os.getenv("SPECIALIST_MODEL", "claude-haiku-5-5")  # swap to claude-sonnet-5-5 only if evals need it

# The 5.5 models think by default. Thinking costs tokens and these jobs are short, so it is turned off.
NO_THINKING = {"thinking": {"type": "disabled"}}

# Short, cheap calls (router, emergency and confirmation checks): fail fast, one retry.
LIGHT = CallConfig(timeout_s=10.0, overall_timeout_s=25.0, max_attempts=2, base_delay_s=0.5, params=NO_THINKING)
# Specialist agents.
SPECIALIST = CallConfig(timeout_s=30.0, overall_timeout_s=60.0, max_attempts=3, params=NO_THINKING)
SPECIALIST_MAX_SECONDS = 75.0
