Pre-check for the regime belief head (doc 26). Run from the repo root:

    python rl/experiments/belief_precheck/state_dims.py                 # per-dim mean/sd, calm/crash/normal
    python rl/experiments/belief_precheck/collect_states.py states.parquet   # 16 dates x 5 starts, action 0
    python rl/experiments/belief_precheck/predictability.py states.parquet   # drift R^2, regime accuracy
