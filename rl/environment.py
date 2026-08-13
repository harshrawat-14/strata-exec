import gymnasium as gym
import subprocess
import json
import numpy as np
from typing import Optional


def _query_action_fractions() -> list:
    """
    Single source of truth for the action->fraction mapping: query the real
    Rust environment (`ACTION_FRACTIONS` in rl_env.rs) rather than maintaining
    a hand-copied mirror in Python.

    Phase 2R finding: this project has hand-maintained mirrors of this exact
    table diverge silently before (rl/evaluate.py's V5_FRACTIONS/V6_FRACTIONS,
    one of which matched the real table at only 3/20 indices). This module
    used to carry its own third hardcoded copy -- currently byte-identical to
    the real table by luck, not by construction, and unprotected by any test.
    Root-fixed the same way as evaluate.py's copy: read from the one
    canonical source at runtime.
    """
    result = subprocess.run(
        ["./target/release/rl-env", "--print-action-fractions"],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout.strip())


ACTION_FRACTIONS = _query_action_fractions()


# Pre-Step-0 fallback. The Step 0 commit (0e3356b) recalibrated sigma_ref from
# this value to CALIBRATED_SIGMA_REF; binaries built before it have neither the
# new constant nor the --print-env-constants flag. 0.20 is the CORRECT value to
# use against such a binary -- it is what that binary's environment actually
# runs -- so this fallback is a version-matched default, not a guess.
_PRE_STEP0_SIGMA_REF = 0.20


def _query_env_constants() -> dict:
    """
    Single source of truth for scalar environment constants, same mechanism as
    `_query_action_fractions` above (see that docstring for why this project
    reads canonical values at runtime instead of mirroring them).

    Phase 3, bug-class instance #5: Step 0 recalibrated `sigma_ref` in Rust
    while `rl/evaluate.py`'s AdaptiveACPolicy and `rl/batched_environment.py`
    each kept a hardcoded `sigma_ref: float = 0.20` default that no call site
    overrode -- so the Python baseline silently reconstructed volatility
    against a reference the environment no longer used.

    Falls back for binaries predating the flag rather than raising: evaluating
    the pre-Step-0 RL16/RL12 checkpoints REQUIRES a pre-Step-0 binary (see
    `rl/results/PROVENANCE.md`), so that path is a supported configuration, not
    an error. `sigma_ref_source` records which case was taken so callers and
    tests can assert on it instead of inferring.
    """
    try:
        # stdin=DEVNULL and a timeout are load-bearing, not defensive padding:
        # `parse_cli` IGNORES unrecognized flags, so a pre-Step-0 binary handed
        # --print-env-constants does not error -- it falls through and starts
        # the stdin JSON loop. Inheriting the parent's stdin there would block
        # this import indefinitely. DEVNULL makes that path hit EOF and exit;
        # the timeout is the backstop if it ever does neither.
        result = subprocess.run(
            ["./target/release/rl-env", "--print-env-constants"],
            capture_output=True, text=True, check=True,
            stdin=subprocess.DEVNULL, timeout=30,
        )
        constants = json.loads(result.stdout.strip())
        if "calibrated_sigma_ref" not in constants:
            raise KeyError("calibrated_sigma_ref")
        constants["sigma_ref_source"] = "binary"
        return constants
    except (
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
        FileNotFoundError,
        KeyError,
    ):
        # Pre-Step-0 binary: unrecognized flag ignored, env loop ran and exited
        # on EOF, so stdout is a startup banner rather than the constants JSON.
        return {
            "calibrated_sigma_ref": _PRE_STEP0_SIGMA_REF,
            "default_sigma": _PRE_STEP0_SIGMA_REF,
            "trading_steps_per_year": None,
            "sigma_ref_source": "pre_step0_fallback",
        }


ENV_CONSTANTS = _query_env_constants()

#: sigma_ref in force for historical/counterfactual mode -- the modes every
#: Python-side policy is evaluated in. Do NOT hardcode this value anywhere.
CALIBRATED_SIGMA_REF = ENV_CONSTANTS["calibrated_sigma_ref"]

#: True when the binary in use emits the Step 0 continuous regime dims
#: (s10 = tanh(ln(current_sigma/sigma_ref))); False against a pre-Step-0
#: binary, where s10 is the old dead 3-way bucket {0.0, 0.5, 1.0}.
#:
#: This gates which state dim a Python policy should invert to recover
#: volatility, and the difference is material, not stylistic. s2 =
#: tanh(ratio) saturates: with own-impact-driven volatility the ratio reaches
#: ~13 on 2024-08-05, where tanh is 1.0 to within 6e-12 and no inversion can
#: recover it -- s2 caps out around ratio 8.4 (sigma ~4.2), understating a
#: true sigma of 6.57 by ~36%. s10 applies the log FIRST, so the same ratio
#: encodes as tanh(2.58) = 0.988 and inverts exactly. Prefer s10 wherever it
#: is available; fall back to s2 only where it is not.
REGIME_DIMS_ARE_CONTINUOUS = ENV_CONSTANTS["sigma_ref_source"] == "binary"

class StrataExecEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        mode: str = "synthetic",
        adversarial: bool = False,
        lob_date: Optional[str] = None,
        agg_date: Optional[str] = None,
        btc_target: float = 500.0,
        seed: int = 42,
        n_state_dims: int = 8,
        fixed_steps: bool = False,
        fixed_size: bool = False,
        jump_diffusion: bool = False,
        no_time_features: bool = False,
        total_steps: Optional[int] = None,
    ):
        import os
        if lob_date:

            lob_path = f"TradeData/BookDepth/BTCUSDT-bookDepth-{lob_date}.csv"
            if not os.path.exists(lob_path):
                raise FileNotFoundError(f"Historical LOB data file not found for date: {lob_date} (path: {lob_path})")
        if agg_date:
            agg_path = f"TradeData/AggTrades/BTCUSDT-aggTrades-{agg_date}.csv"
            if not os.path.exists(agg_path):
                raise FileNotFoundError(f"Historical AggTrades data file not found for date: {agg_date} (path: {agg_path})")

        super().__init__()


        self.cmd = [
            "./target/release/rl-env",
            "--rl-mode", mode,
        ]
        if adversarial:
            self.cmd.append("--adversarial")
        if lob_date and mode != "counterfactual":
            self.cmd.extend([
                "--lob-data",
                f"TradeData/BookDepth/BTCUSDT-bookDepth-{lob_date}.csv",
                "--btc-target", str(btc_target),
            ])
        if mode == "counterfactual":
            assert lob_date is not None, "counterfactual mode requires lob_date"
            self.cmd.extend([
                "--lob-data",
                f"TradeData/BookDepth/BTCUSDT-bookDepth-{lob_date}.csv",
                "--btc-target", str(btc_target),
            ])
            if agg_date:
                self.cmd.extend([
                    "--agg-trades",
                    f"TradeData/AggTrades/BTCUSDT-aggTrades-{agg_date}.csv",
                ])
        elif agg_date:
            self.cmd.extend([
                "--agg-trades",
                f"TradeData/AggTrades/BTCUSDT-aggTrades-{agg_date}.csv"
            ])

        if total_steps is not None:
            self.cmd.extend(["--total-steps", str(total_steps)])
        if fixed_steps:
            self.cmd.append("--fixed-steps")
        if fixed_size:
            self.cmd.append("--fixed-size")
        if jump_diffusion:
            self.cmd.append("--jump-diffusion")
        if no_time_features:
            self.cmd.append("--no-time-features")

        # Diagnostic: print the full command so --agg-trades presence is easy to verify.
        if mode == "counterfactual":
            import sys
            print(f"[StrataExecEnv] cmd: {' '.join(self.cmd)}", file=sys.stderr)

        # Counterfactual mode produces 12-dim state; others produce 8.
        if n_state_dims == 8 and mode == "counterfactual":
            n_state_dims = 12

        self.n_state_dims = n_state_dims
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_state_dims,),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(20)


        self.process = None
        self.current_seed = seed
        self._last_info = {}

    def _send(self, obj: dict) -> dict:
        if self.process is None or self.process.poll() is not None:
            self._start_process()
        line = json.dumps(obj) + "\n"
        self.process.stdin.write(line.encode())
        self.process.stdin.flush()
        response = self.process.stdout.readline()
        return json.loads(response.decode())

    def _start_process(self):
        self.process = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        use_seed = seed if seed is not None else self.current_seed
        result = self._send({"reset": True, "seed": use_seed})
        self._last_info = result.get("info", {})
        obs = np.array(result["state"][:self.n_state_dims], dtype=np.float32)
        return obs, self._last_info


    def step(self, action: int):
        result = self._send({"action": int(action)})
        obs    = np.array(result["state"][:self.n_state_dims], dtype=np.float32)

        reward = float(result["reward"])
        done   = bool(result["done"])
        trunc  = False
        info   = result.get("info", {})
        self._last_info = info
        return obs, reward, done, trunc, info

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait()
