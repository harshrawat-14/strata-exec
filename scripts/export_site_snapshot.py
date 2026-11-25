"""
Export the results shown on the public site into one frozen JSON file.

The site is pinned to the PPO-LSTM 16-dim model (RL-16, `ppo_ablation_16dim_cf_best`). Its results
are read from git at the pinned commit, never from the working tree, so research on `main` cannot
change what the site shows.

    python scripts/export_site_snapshot.py            # writes frontend/src/data/pinned/results.json
    python scripts/export_site_snapshot.py --check    # fails if the committed file is out of date
"""
import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PIN = "site-pin-ppo-lstm"  # tag on ae90ac1, Phase 3 Step 1 (doc 05): 25 dates, 50 episodes each
EPISODES = "rl/results/phase3_tail_validation_episodes.csv"
SUMMARY = "rl/results/phase3_tail_validation_summary.csv"
ABLATION = "rl/results/phase2r_ablation_honest_lambda.csv"
OUT = Path(__file__).resolve().parents[1] / "frontend/src/data/pinned/results.json"

STRATEGIES = {  # csv name -> site key, label
    "twap": ("twap", "TWAP"),
    "heuristic": ("heuristic", "Heuristic"),
    "ac_optimal": ("ac_optimal", "AC Optimal"),
    "adaptive_ac": ("adaptive_ac", "Adaptive AC"),
    "rl_agent": ("rl", "RL Agent"),
}
EVENTS = {
    "A": "Iran–Israel escalation",
    "B": "German BTC sales / Mt. Gox fear",
    "C": "US jobs report → yen-carry unwind",
    "isolated": "Isolated day",
}


def _csv(path: str) -> pd.DataFrame:
    text = subprocess.run(["git", "show", f"{PIN}:{path}"], check=True, capture_output=True, text=True).stdout
    return pd.read_csv(io.StringIO(text))


def _cvar95(x) -> float:
    """Mean of the worst 5% of episodes (IS: higher is better, so the lowest values)."""
    s = sorted(x)
    k = max(1, round(len(s) * 0.05))
    return float(sum(s[:k]) / k)


def _event(window: str) -> tuple:
    """'C (jobs-report -> Aug 5 culmination)' -> ('C', 'US jobs report → yen-carry unwind · Aug 5 culmination')."""
    key = window.split(" ")[0]
    name = EVENTS.get(key, window)
    note = window[len(key):].strip(" ()").replace("->", "→") if "(" in window else ""
    if key == "isolated":
        return key, note[:1].upper() + note[1:] if note else name
    if "culmination" in note:
        return key, f"{name} · {note.split('→')[-1].strip()}"
    return key, name


def build() -> dict:
    ep = _csv(EPISODES)
    summary = _csv(SUMMARY).set_index("date")
    abl = _csv(ABLATION)
    commit = subprocess.run(["git", "rev-list", "-n", "1", PIN], check=True, capture_output=True,
                            text=True).stdout.strip()

    dates = []
    for date, g in ep.groupby("date"):
        row = summary.loc[date]
        event, detail = _event(row.window_v2)
        strategies = {}
        for name, (key, _) in STRATEGIES.items():
            x = g[g.strategy == name].sort_values("episode_seed").is_pct.tolist()
            s = pd.Series(x)
            strategies[key] = dict(mean=round(s.mean(), 4), std=round(s.std(), 4), cvar95=round(_cvar95(x), 4),
                                   episodes=[round(v, 4) for v in x])
        dates.append(dict(date=date, regime=row.regime_v2,
                          split="held_out" if row.split_v2 == "held_out_v2" else "train",
                          event=event, event_detail=detail, strategies=strategies))

    ablation = [dict(date=r.date, regime=r.regime, split="held_out" if "held" in r.split else "train",
                     rl12=round(r.rl12_mean_IS, 4), rl16=round(r.rl16_mean_IS, 4), ac=round(r.ac_mean_IS, 4),
                     delta16=round(r.delta_16_vs_ac_mean_IS, 4), delta16_lo=round(r.delta_16_vs_ac_ci_lower, 4),
                     delta16_hi=round(r.delta_16_vs_ac_ci_upper, 4), delta12=round(r.delta_12_vs_ac_mean_IS, 4))
                for r in abl.itertuples()]

    return dict(
        model=dict(name="PPO-LSTM", variant="RL-16 (16-dim state)", algorithm="RecurrentPPO, LSTM policy",
                   checkpoint="ppo_ablation_16dim_cf_best", pinned_tag=PIN, pinned_commit=commit),
        protocol=dict(episodes_per_strategy=int(ep.groupby(["date", "strategy"]).size().iloc[0]),
                      metric="Implementation shortfall, % of arrival value; higher is better",
                      venue="Binance BTC-USDT perpetual futures, 2024"),
        strategies=[dict(key=k, label=lab) for k, lab in STRATEGIES.values()],
        events=EVENTS,
        dates=sorted(dates, key=lambda d: d["date"]),
        ablation=sorted(ablation, key=lambda d: d["date"]),
    )


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit(f"{OUT} is out of date with tag {PIN}; rerun scripts/export_site_snapshot.py")
        print("snapshot matches", PIN)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
