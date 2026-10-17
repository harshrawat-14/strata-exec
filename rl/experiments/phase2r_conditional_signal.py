"""
Phase 2R conditional-signal analysis (the interesting question).

Under the drift-neutral VWAP benchmark, the WAIT-vs-EXECUTE preference is
per-date conditional (rl/experiments/phase2r_vwap_gate.py): execution wins on intra-window
crashes (2024-03-05), waiting wins on already-crashed/flat-thin books
(2024-08-05). The question that decides whether a rail is needed:

  Is that preference distinguishable from the STATE VECTOR the agent sees?

If the EXEC-wins dates have a separable early-window signature in the observable
dims (realized vol s2, spread s3, depth s4, OFI s5, short drift s6), the correct
policy is conditional and learnable -- no inventory rail needed. If EXEC-wins and
WAIT-wins dates are indistinguishable from the state, that's a state-representation
finding.

Method: capture the undisturbed state path (wait rollout) over the EARLY window
(first `EARLY_STEPS` steps -- before heavy execution must commit), average each
observable dim per date, group by the VWAP verdict from the gate.

Run: rl/.venv/bin/python rl/experiments/phase2r_conditional_signal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # rl/, for bare imports like `import evaluate`

import numpy as np

from environment import StrataExecEnv

# VWAP verdicts from rl/experiments/phase2r_vwap_gate.py (EXEC wins = execution beats waiting)
VWAP_VERDICT = {
    "2024-01-15": "EXEC", "2024-02-20": "WAIT", "2024-04-15": "EXEC",
    "2024-06-10": "WAIT", "2024-09-15": "EXEC", "2024-10-15": "WAIT",
    "2024-11-06": "WAIT", "2024-08-05": "WAIT", "2024-03-05": "EXEC",
}
# Observable state dims that could plausibly distinguish "crashing now" from "flat thin book"
DIMS = {2: "vol(s2)", 3: "spread(s3)", 4: "depth(s4)", 5: "OFI(s5)", 6: "drift(s6)"}
EARLY_STEPS = 60
N_SEEDS = 10


def early_signature(date):
    env = StrataExecEnv(mode="historical", adversarial=False, lob_date=date,
                        agg_date=date, n_state_dims=16, fixed_steps=True,
                        fixed_size=True, no_time_features=False)
    per_dim = {d: [] for d in DIMS}
    for seed in range(N_SEEDS):
        obs, _ = env.reset(seed=seed)
        rows = [obs]
        done = False
        step = 0
        while not done and step < EARLY_STEPS:
            obs, _, done, _, _ = env.step(0)  # wait -> undisturbed market
            rows.append(obs)
            step += 1
        arr = np.array(rows)
        for d in DIMS:
            per_dim[d].append(float(arr[:, d].mean()))
    env.close()
    return {d: float(np.mean(v)) for d, v in per_dim.items()}


def main():
    print("=" * 88)
    print(f"PHASE 2R CONDITIONAL-SIGNAL ANALYSIS -- early-window ({EARLY_STEPS} steps) state signature")
    print("Grouped by VWAP verdict. Question: are EXEC-wins dates separable from WAIT-wins dates?")
    print("=" * 88)
    header = f"{'date':12s} {'verdict':8s} " + " ".join(f"{n:>11s}" for n in DIMS.values())
    print(header)

    sigs = {}
    for date in VWAP_VERDICT:
        sigs[date] = early_signature(date)

    # print EXEC group then WAIT group
    for grp in ("EXEC", "WAIT"):
        for date, verd in VWAP_VERDICT.items():
            if verd != grp:
                continue
            s = sigs[date]
            print(f"{date:12s} {verd:8s} " + " ".join(f"{s[d]:+11.4f}" for d in DIMS))
        print("-" * 88)

    print("\nGROUP MEANS (EXEC-wins vs WAIT-wins):")
    for grp in ("EXEC", "WAIT"):
        dates = [d for d, v in VWAP_VERDICT.items() if v == grp]
        means = {d: np.mean([sigs[dt][d] for dt in dates]) for d in DIMS}
        print(f"  {grp}: " + "  ".join(f"{DIMS[d]}={means[d]:+.4f}" for d in DIMS))

    print("\nSEPARABILITY (|EXEC_mean - WAIT_mean| / pooled_std per dim):")
    for d in DIMS:
        ex = [sigs[dt][d] for dt, v in VWAP_VERDICT.items() if v == "EXEC"]
        wa = [sigs[dt][d] for dt, v in VWAP_VERDICT.items() if v == "WAIT"]
        pooled = np.sqrt((np.var(ex) + np.var(wa)) / 2) + 1e-9
        sep = abs(np.mean(ex) - np.mean(wa)) / pooled
        flag = "  <-- separable" if sep > 1.0 else ""
        print(f"  {DIMS[d]:12s}: EXEC={np.mean(ex):+.4f}  WAIT={np.mean(wa):+.4f}  sep={sep:.2f}{flag}")

    print("\nThe decisive contrast -- both crash dates, opposite verdicts:")
    for date in ("2024-03-05", "2024-08-05"):
        s = sigs[date]
        print(f"  {date} ({VWAP_VERDICT[date]}): " + "  ".join(f"{DIMS[d]}={s[d]:+.4f}" for d in DIMS))


if __name__ == "__main__":
    main()
