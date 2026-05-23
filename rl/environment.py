import gymnasium as gym
import subprocess
import json
import numpy as np
from typing import Optional

ACTION_FRACTIONS = [
    0.000, 0.001, 0.003, 0.005, 0.008,
    0.010, 0.015, 0.020, 0.030, 0.050,
    0.070, 0.100, 0.150, 0.200, 0.300,
    0.400, 0.500, 0.700, 0.900, 1.000
]

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
    ):
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

        # Diagnostic: print the full command so --agg-trades presence is easy to verify.
        if mode == "counterfactual":
            import sys
            print(f"[StrataExecEnv] cmd: {' '.join(self.cmd)}", file=sys.stderr)

        # Counterfactual mode produces 10-dim state; others produce 8.
        if n_state_dims == 8 and mode == "counterfactual":
            n_state_dims = 10

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
        obs = np.array(result["state"], dtype=np.float32)
        return obs, self._last_info

    def step(self, action: int):
        result = self._send({"action": int(action)})
        obs    = np.array(result["state"], dtype=np.float32)
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
