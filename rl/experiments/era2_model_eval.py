"""
Score one era-2 RL checkpoint against the era-2 baselines (doc 22).

The checkpoint is decoded the protocol-v3 way, run at 5% of ADV on the 20 evenly
spaced starts of every training and validation date, and compared, date by date,
with TWAP and the constant paces in rl/results/era2_baselines.csv (same starts,
same sizes). Held-out dates are never touched.

    python rl/experiments/era2_model_eval.py --model rl/models/checkpoints/m0_e2/ckpt_N_steps --name m0_e2
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/

import numpy as np
import pandas as pd

SHARE = 0.05


def _job(args):
    from select_checkpoint_v3 import _evaluate
    return _evaluate(args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint path without .zip")
    ap.add_argument("--name", required=True)
    ap.add_argument("--action-space", choices=["raw", "residual_ac", "rate"], default="raw")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from evaluation.dates import SEVERITY_RANK_V2, TRAINING_DATES_V3, VALIDATION_DATES_V3

    dates = list(TRAINING_DATES_V3) + list(VALIDATION_DATES_V3)
    jobs = [(args.name, args.model, d, args.action_space, SHARE) for d in dates]
    with Pool(args.workers) as pool:
        rl = pd.DataFrame([r for rs in pool.map(_job, jobs, chunksize=1) for r in rs])
    rl["regime"] = rl.date.map(lambda d: SEVERITY_RANK_V2[d][1])
    rl["split"] = np.where(rl.date.isin(TRAINING_DATES_V3), "train", "validation")
    out = f"rl/results/{args.name}_era2_eval.csv"
    rl.to_csv(out, index=False)

    base = pd.read_csv("rl/results/era2_baselines.csv")
    base = base[(base.share == SHARE) & (base.kind == "pace")]
    paces = base.pivot_table(index="date", columns="param", values="IS").add_prefix("pace ")
    model = rl.groupby("date").agg(IS=("IS", "mean"), forced=("forced", "mean"))
    t = model.join(paces)
    t["regime"] = t.index.map(lambda d: SEVERITY_RANK_V2[d][1])
    t["split"] = np.where(t.index.isin(TRAINING_DATES_V3), "train", "validation")
    t["vs TWAP"] = t.IS - t["pace 1.0"]
    t["vs best pace"] = t.IS - t[[c for c in t.columns if c.startswith("pace ")]].max(axis=1)
    print(t.sort_values(["split", "regime"]).round(3).to_string())
    print("\nmeans by split:")
    print(t.groupby("split")[["IS", "pace 1.0", "pace 1.5", "pace 2.0", "vs TWAP", "forced"]].mean().round(4))
    print("\nmeans by regime (all dates):")
    print(t.groupby("regime")[["IS", "pace 1.0", "vs TWAP"]].mean().round(4))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
