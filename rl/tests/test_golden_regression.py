"""
Exact-equivalence guard for behaviour-preserving refactors.

Compares the current rl-env binary and evaluate.py harness against fixtures
recorded on a known-good commit (see rl/tests/golden/golden_lib.py). Any
difference -- one bit in one float of one step -- fails.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "golden"))

import golden_lib  # noqa: E402
from environment import RL_ENV_BIN  # noqa: E402

# Fixtures need gitignored local inputs (TradeData/, rl/models/, the release
# binary); skip cleanly where they are absent, e.g. CI.
_REQUIRED = [
    RL_ENV_BIN,
    golden_lib._book(golden_lib.CALM_DATE),
    golden_lib._book(golden_lib.CRASH_DATE),
    f"{golden_lib.EVAL_MODEL}.zip",
]
pytestmark = pytest.mark.skipif(
    not all(os.path.exists(p) for p in _REQUIRED),
    reason="requires local TradeData/, rl/models/ and target/release/rl-env",
)


@pytest.fixture(scope="module")
def env_actual():
    return golden_lib.collect_env_golden(RL_ENV_BIN)


@pytest.fixture(scope="module")
def env_expected():
    return json.loads(golden_lib.ENV_FIXTURE.read_text())


@pytest.mark.parametrize("config", sorted(golden_lib.STDIO_CONFIGS))
def test_stdio_trajectories_unchanged(config, env_actual, env_expected):
    assert env_actual["stdio"][config] == env_expected["stdio"][config]


@pytest.mark.parametrize("name", ["synthetic", "counterfactual_crash"])
def test_batched_trajectories_unchanged(name, env_actual, env_expected):
    assert env_actual["batched"][name] == env_expected["batched"][name]


def test_cli_queries_unchanged(env_actual, env_expected):
    for key in ("print_action_fractions", "print_calibration"):
        assert env_actual[key] == env_expected[key], key


def test_eval_harness_unchanged():
    expected = json.loads(golden_lib.EVAL_FIXTURE.read_text())
    assert golden_lib.collect_eval_golden() == expected
