"""
Protocol v3 model selection: pick a training run's checkpoint offline.

Rule (fixed 2026-10-18, before M0's selection was run): the checkpoint with the
best mean IS over VALIDATION_DATES_V3 x 20 evenly spaced starts, with every
checkpoint decoded by MeanFractionDecoder. In-training EvalCallback selection
is not used under v3 -- it decodes by argmax, which made M0's "best" model the
near-random 50k checkpoint (docs/experiments/13-m0-protocol-v3-result.md).

    python rl/experiments/select_checkpoint_v3.py --run m0_v3
"""
import argparse
import glob
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import numpy as np
import pandas as pd


def _evaluate(args):
    name, path, date = args
    from environment import StrataExecEnv
    from evaluation.decoding import MeanFractionDecoder
    from evaluation.episode import _run_episode, load_model
    from evaluation.protocol import EPISODE_STEPS, N_EVAL_EPISODES, eval_start_fracs

    model = MeanFractionDecoder(load_model(path))
    env = StrataExecEnv(mode="counterfactual", lob_date=date, agg_date=date, btc_target=50_000.0,
                        n_state_dims=16, fixed_steps=True, fixed_size=True,
                        episode_steps=EPISODE_STEPS)
    try:
        out = [_run_episode(model, env, seed=ep, start_frac=f)
               for ep, f in enumerate(eval_start_fracs(N_EVAL_EPISODES))]
    finally:
        env.close()
    return [dict(checkpoint=name, date=date, IS=o[0], forced=o[2]) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training run name (train.py --name)")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from evaluation.dates import VALIDATION_DATES_V3

    ckpts = {}
    for p in glob.glob(f"rl/models/checkpoints/{args.run}/ckpt_*_steps.zip"):
        steps = int(re.search(r"ckpt_(\d+)_steps", p).group(1))
        ckpts[f"{steps:08d}"] = p[:-4]
    final = f"rl/models/{args.run}_final"
    if os.path.exists(final + ".zip"):
        ckpts["final"] = final
    if not ckpts:
        sys.exit(f"no checkpoints found for run {args.run!r}")

    jobs = [(n, p, d) for n, p in ckpts.items() for d in VALIDATION_DATES_V3]
    with Pool(args.workers) as pool:
        rows = [r for rs in pool.map(_evaluate, jobs, chunksize=1) for r in rs]
    df = pd.DataFrame(rows)
    out = f"rl/results/{args.run}_checkpoint_selection.csv"
    df.to_csv(out, index=False)

    table = df.groupby("checkpoint").agg(mean_IS=("IS", "mean"), forced_rate=("forced", "mean"))
    print(table.round(4).to_string())
    best = table.mean_IS.idxmax()
    print(f"\nSELECTED: {best} -> {ckpts[best]}  (mean validation IS {table.mean_IS[best]:.4f})")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
