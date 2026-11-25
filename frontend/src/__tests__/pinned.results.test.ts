/**
 * The public site must show exactly the pinned PPO-LSTM results (tag site-pin-ppo-lstm).
 * These numbers are the published ones (doc 05, Phase 3 Step 1); a change here means the
 * snapshot was regenerated from something other than the pinned tag.
 */
import { describe, it, expect } from "vitest";
import {
  PINNED, edgeVsAc, episodeHistogram, filterDates, strategyMeansByRegime, summariseByRegime,
} from "../data/pinned/results";

const byDate = (d: string) => PINNED.dates.find(x => x.date === d)!;

describe("pinned snapshot", () => {
  it("is the RL-16 PPO-LSTM checkpoint at the pinned commit", () => {
    expect(PINNED.model.checkpoint).toBe("ppo_ablation_16dim_cf_best");
    expect(PINNED.model.pinned_tag).toBe("site-pin-ppo-lstm");
    expect(PINNED.model.pinned_commit.startsWith("ae90ac1")).toBe(true);
  });

  // Regression: RL-16 trained on the 7 V1 dates (rl/evaluate.py TRAINING_DATES at the tag).
  // The later V2 split labelled 16 days "training", 9 of which this model never saw.
  it("covers 25 days: the 7 RL-16 training days and 18 it never saw", () => {
    expect(PINNED.dates).toHaveLength(25);
    expect(filterDates(PINNED.dates, "train", "all").map(d => d.date)).toEqual([
      "2024-01-15", "2024-02-20", "2024-04-15", "2024-06-10", "2024-09-15", "2024-10-15", "2024-11-06",
    ]);
    expect(filterDates(PINNED.dates, "held_out", "all")).toHaveLength(18);
  });

  it("has 50 episodes per strategy per day, and each mean is the mean of its episodes", () => {
    for (const d of PINNED.dates) {
      for (const s of PINNED.strategies) {
        const r = d.strategies[s.key];
        expect(r.episodes).toHaveLength(50);
        const m = r.episodes.reduce((a, b) => a + b, 0) / r.episodes.length;
        expect(Math.abs(m - r.mean)).toBeLessThan(1e-3);
      }
    }
  });

  it("reproduces the published RL - AC Optimal edges", () => {
    expect(edgeVsAc(byDate("2024-08-05"))).toBeCloseTo(1.407, 2);
    expect(edgeVsAc(byDate("2024-03-05"))).toBeCloseTo(1.274, 2);
    expect(edgeVsAc(byDate("2024-07-08"))).toBeCloseTo(-1.328, 2);
    expect(byDate("2024-08-05").strategies.rl.mean).toBeCloseTo(-6.481, 2);
  });
});

describe("summaries", () => {
  it("wins every crash day, including the 7 it never trained on", () => {
    const all = summariseByRegime(PINNED.dates).find(s => s.regime === "crash")!;
    expect([all.wins, all.days]).toEqual([8, 8]);
    const unseen = summariseByRegime(filterDates(PINNED.dates, "held_out", "all")).find(s => s.regime === "crash")!;
    expect([unseen.wins, unseen.days]).toEqual([7, 7]);
  });

  it("wins 1 of 17 calm and normal days", () => {
    const quiet = summariseByRegime(PINNED.dates).filter(s => s.regime !== "crash");
    expect(quiet.reduce((a, s) => a + s.days, 0)).toBe(17);
    expect(quiet.reduce((a, s) => a + s.wins, 0)).toBe(1);
  });

  it("averages strategies within each regime present", () => {
    const rows = strategyMeansByRegime(filterDates(PINNED.dates, "all", "crash"));
    expect(rows).toHaveLength(1);
    const crash = filterDates(PINNED.dates, "all", "crash");
    const rl = crash.reduce((a, d) => a + d.strategies.rl.mean, 0) / crash.length;
    expect(rows[0]["RL Agent"]).toBeCloseTo(rl, 10);
  });

  it("puts every episode in exactly one histogram bin", () => {
    const hist = episodeHistogram(byDate("2024-08-05"), ["ac_optimal", "rl"]);
    for (const k of ["ac_optimal", "rl"]) {
      expect(hist.reduce((a, b) => a + (b[k] as number), 0)).toBe(50);
    }
  });
});
