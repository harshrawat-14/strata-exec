"""
Phase S baseline throughput benchmark for the JSON-over-stdio rl-env
transport, BEFORE any transport changes. Run from repo root:

    rl/.venv/bin/python docs/system_design/phase_s_bench_baseline.py

Measures steps/sec and episodes/sec driving StrataExecEnv exactly the way
rl/evaluate.py's _run_episode does (TWAPPolicy, deterministic, real
2024-01-15 historical data, fixed_steps/fixed_size so episode length is
deterministic).
"""
import sys
import time

sys.path.insert(0, "rl")

from environment import StrataExecEnv  # noqa: E402
from evaluate import TWAPPolicy  # noqa: E402

DATE = "2024-01-15"
N_EPISODES = 20


def main():
    env = StrataExecEnv(
        mode="historical",
        lob_date=DATE,
        agg_date=DATE,
        n_state_dims=12,
        fixed_steps=True,
        fixed_size=True,
        no_time_features=True,
    )
    policy = TWAPPolicy(12)

    total_steps = 0
    t0 = time.perf_counter()
    ep_times = []

    for ep in range(N_EPISODES):
        ep_t0 = time.perf_counter()
        obs, info = env.reset(seed=ep)
        done = False
        n_steps = 0
        while not done:
            action, _ = policy.predict(obs)
            obs, reward, done, trunc, step_info = env.step(int(action))
            n_steps += 1
        ep_times.append(time.perf_counter() - ep_t0)
        total_steps += n_steps

    elapsed = time.perf_counter() - t0
    env.close()

    print(f"date={DATE} episodes={N_EPISODES} total_steps={total_steps}")
    print(f"wall_time_s={elapsed:.4f}")
    print(f"steps_per_sec={total_steps / elapsed:.1f}")
    print(f"episodes_per_sec={N_EPISODES / elapsed:.4f}")
    print(f"mean_episode_time_s={sum(ep_times) / len(ep_times):.4f}")
    print(f"mean_steps_per_episode={total_steps / N_EPISODES:.1f}")


if __name__ == "__main__":
    main()
