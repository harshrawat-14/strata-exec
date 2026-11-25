"""Doc 33 calibration helpers agree with doc 29/31's curve code and the simulator's formula."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_EXP = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(_EXP))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _EXP / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cal = _load("impact_decay_calibrate")
fit29 = _load("impact_decay_fit")


def test_real_curve_matches_doc29_curve_at_its_check_lags():
    beta = np.array([1.0, -0.2, -0.05, -0.013, -0.003, 0.0005, 0.0005, 0.0002, 0.0003, -0.00003, -0.00007])
    expected = fit29.curve(beta)
    got = cal.real_curve(beta, np.array(list(fit29.TAUS.values())))
    np.testing.assert_allclose(got, [expected[k] for k in fit29.TAUS], rtol=1e-12)


def test_sim_curve_at_old_params_reproduces_documented_values():
    got = cal.sim_curve(0.3, 5.0, np.array(list(fit29.TAUS.values())))
    np.testing.assert_allclose(got, [fit29.SIM[k] for k in fit29.TAUS], atol=1e-3)


def test_fit_recovers_known_parameters():
    curve = cal.sim_curve(0.45, 800.0, cal.GRID)
    pi, rho = cal.fit(curve)
    assert abs(pi - 0.45) < 1e-4 and abs(rho / 800.0 - 1) < 1e-3
