"""Doc 35 report: must work on the saved CSV, not only on the in-memory frame."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd

_EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXP))
sys.path.insert(0, str(_EXP.parent))
_SPEC = importlib.util.spec_from_file_location("m0_rescore", _EXP / "m0_rescore.py")
mr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mr)


def _frame():
    rows = []
    for s, date, regime in (("v3", "2024-04-15", "crash"), ("oos", "2024-10-20", "calm"),
                            ("oos", "2024-10-21", "normal")):
        for e in range(2):
            # the model's evaluator reports forced as a bool, the pace runner as a float
            rows.append(dict(date=date, strategy="M", episode=e, IS=-0.5, forced=False, steps=600,
                             set=s, regime=regime))
            for p in mr.PACES:
                rows.append(dict(date=date, strategy=f"pace {p}", episode=e, IS=-0.4, forced=0.0,
                                 steps=1918, set=s, regime=regime))
    return pd.DataFrame(rows)


def test_report_runs_on_a_round_tripped_csv(tmp_path, capsys):
    """Regression: mixed bool/float 'forced' reloads as strings and the report crashed."""
    path = tmp_path / "r.csv"
    _frame().to_csv(path, index=False)
    mr.report(pd.read_csv(path), "M")
    out = capsys.readouterr().out
    assert "M forced-liquidation share: max 0.0000" in out
    assert "model - TWAP -0.100" in out
