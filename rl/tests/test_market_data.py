"""Dataset selection for the per-coin experiment scripts (doc 34)."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import market_data as md


@pytest.fixture(autouse=True)
def _restore_env(monkeypatch):
    # setenv first so teardown restores the original state even when the variable was unset
    # (delenv of an unset name records nothing, and the test's dataset leaked into later tests)
    monkeypatch.setenv("STRATAEXEC_DATASET", md.DEFAULT_DATASET)
    monkeypatch.delenv("STRATAEXEC_DATASET")


def test_default_dataset_keeps_result_names():
    assert md.results_tag() == ""


def test_dataset_flag_sets_env_and_tag():
    rest = md.use_dataset_from_argv(["v4", "--dataset", "bybit_solusdt_perp", "signflip"])
    assert rest == ["v4", "signflip"]
    assert os.environ["STRATAEXEC_DATASET"] == "bybit_solusdt_perp"
    assert md.results_tag() == "_solusdt"
    assert md.book_path("2024-05-01").startswith("TradeData/Bybit/SOLUSDT/")


def test_no_flag_leaves_args_and_dataset_alone():
    assert md.use_dataset_from_argv(["v4"]) == ["v4"]
    assert "STRATAEXEC_DATASET" not in os.environ


@pytest.mark.parametrize("argv", [["--dataset"], ["--dataset", "nope"]])
def test_bad_dataset_flag_is_rejected(argv):
    with pytest.raises(ValueError):
        md.use_dataset_from_argv(argv)
