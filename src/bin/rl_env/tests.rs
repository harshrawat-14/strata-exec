//! rl-env unit tests.

#[cfg(feature = "alloc-timing")]
use crate::alloc_timing;
use crate::batch_arrow;
use crate::batched::*;
use crate::config::*;
use crate::env::*;
use crate::transport::*;

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[test]
fn action_fractions_starts_zero_ends_one_strictly_increasing() {
    // Phase 2R fraction-table bug guard: this const IS the single source
    // of truth (Python queries it via --print-action-fractions rather
    // than hand-maintaining a mirror). Sanity-check its own shape here;
    // the cross-language identity check lives in
    // rl/tests/test_evaluate.py::TestActionFractionsSingleSourceOfTruth.
    assert_eq!(ACTION_FRACTIONS[0], 0.0);
    assert_eq!(ACTION_FRACTIONS[ACTION_FRACTIONS.len() - 1], 1.0);
    for w in ACTION_FRACTIONS.windows(2) {
        assert!(
            w[0] < w[1],
            "action fractions must be strictly increasing: {w:?}"
        );
    }
}

/// Bit-for-bit hash of one full episode's trajectory: every state vector
/// (hashed via f64::to_bits(), not float comparison, so this is a true
/// bit-identity check with no tolerance) plus the terminal reward/inventory.
/// Two runs with the same seed must produce the same hash, deterministically,
/// or something in the state machine (RNG usage, HashMap iteration order,
/// uninitialised memory, the new per-date cache, etc.) is non-deterministic.
fn run_episode_and_hash(env: &mut RlEnv, seed: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    let reset_result = env.reset(Some(seed));
    for bit in reset_result.state.iter().map(|f| f.to_bits()) {
        bit.hash(&mut hasher);
    }
    let mut done = reset_result.done;
    let mut action_cycle = 0usize;
    while !done {
        // Deterministic pseudo-varied action sequence (not RNG-driven) so the
        // trajectory actually exercises different code paths across steps,
        // while remaining identical run-to-run for a fixed seed.
        let action = action_cycle % ACTION_FRACTIONS.len();
        action_cycle += 1;
        let r = env.step(action);
        for bit in r.state.iter().map(|f| f.to_bits()) {
            bit.hash(&mut hasher);
        }
        r.reward.to_bits().hash(&mut hasher);
        r.done.hash(&mut hasher);
        done = r.done;
    }
    hasher.finish()
}

#[test]
fn synthetic_episode_hash_identical_across_100_runs_same_seed() {
    let seed = 12345;
    let mut hashes = Vec::with_capacity(100);
    for _ in 0..100 {
        let mut env = make_env();
        hashes.push(run_episode_and_hash(&mut env, seed));
    }
    let first = hashes[0];
    for (i, h) in hashes.iter().enumerate() {
        assert_eq!(
            *h, first,
            "run {i} produced a different trajectory hash than run 0 for the \
             same seed -- the environment is not deterministic"
        );
    }
}

#[test]
#[cfg(feature = "alloc-timing")]
#[ignore = "manual benchmark; run with --features alloc-timing -- --ignored --nocapture"]
fn step_allocation_time_share_of_wall_time() {
    // Follow-up to the dhat-heap allocation-COUNT measurement in
    // docs/performance/baseline-profiling.md: 15.01 allocs/step is a count,
    // not a cost. This measures what fraction of a step iteration's actual
    // wall time is spent inside the allocator, now that reset()'s ~239ms
    // no longer dominates -- run with:
    //   cargo test --release --features alloc-timing \
    //     step_allocation_time_share_of_wall_time -- --nocapture --ignored
    // (--ignored because this needs real 2024-01-15 data and is a
    // benchmark, not a correctness assertion; it prints a number rather
    // than asserting a threshold, since "how much allocation costs" is a
    // measurement to report, not a pass/fail gate.)
    let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
    if !std::path::Path::new(lob_path).exists() {
        eprintln!("skipping: {lob_path} not present in this checkout");
        return;
    }

    let cfg = EnvConfig {
        mode: RlMode::Historical,
        total_inventory: 1_000_000.0,
        total_steps: HISTORICAL_DEFAULT_STEPS,
        seed: 42,
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: Some(lob_path.to_string()),
        btc_target: 50_000.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: true,
        fixed_size: true,
        no_time_features: true,
        episode_steps: None,
    };
    let mut env = RlEnv::new(cfg);

    // Warm the per-date cache (excluded from timing -- this measurement is
    // about the steady-state step() cost, not the already-measured reset()
    // cost).
    env.reset(Some(1));
    env.reset(Some(2));

    const N: usize = 2000;
    let incoming_line = "{\"action\": 0}";
    alloc_timing::reset();
    let wall_t0 = std::time::Instant::now();
    for i in 0..N {
        // Full main()-loop scope, calling the REAL production functions
        // (not reimplementations): parse the incoming action message
        // (serde_json::from_str::<InMsg>, same as main()'s hot loop -- an
        // earlier version of this benchmark skipped this and undercounted
        // allocations relative to the dhat-heap measurement in
        // docs/performance/baseline-profiling.md, which does include the input side) ->
        // step() (includes build_state) -> step_result_to_msg -> write_msg.
        // io::sink() discards bytes without buffering/allocating, isolating
        // step()+serialization cost from actual socket/pipe I/O syscall
        // latency, which is not allocation.
        let in_msg: InMsg = serde_json::from_str(incoming_line).unwrap();
        let r = env.step(in_msg.action.unwrap_or(0));
        let out_msg = step_result_to_msg(r);
        write_msg(&mut std::io::sink(), &out_msg).unwrap();
        if i % 500 == 0 {
            // Historical mode terminates after HISTORICAL_DEFAULT_STEPS;
            // reset periodically (not timed as part of the loop average
            // in the sense that matters -- reset() cost is already
            // measured separately and is now ~1.2ms cached, negligible
            // next to N=2000 steps, but exclude it from the allocator
            // share calculation for a clean "steady-state step()" number).
            alloc_timing::reset();
            env.reset(Some((i as u64) % 5 + 1));
            alloc_timing::reset();
        }
    }
    let wall_elapsed = wall_t0.elapsed();
    let (alloc_nanos, alloc_calls) = alloc_timing::snapshot();

    let wall_ns = wall_elapsed.as_nanos() as u64;
    let pct = 100.0 * alloc_nanos as f64 / wall_ns as f64;

    println!("N={N} steps");
    println!(
        "total wall time:      {:.3} ms ({:.1} ns/step)",
        wall_elapsed.as_secs_f64() * 1000.0,
        wall_ns as f64 / N as f64
    );
    println!(
        "time inside allocator: {:.3} ms ({:.1} ns/step, {} calls, {:.2} calls/step)",
        alloc_nanos as f64 / 1e6,
        alloc_nanos as f64 / N as f64,
        alloc_calls,
        alloc_calls as f64 / N as f64
    );
    println!("allocator share of step wall time: {pct:.1}%");
    println!(
        "NOTE: Instant::now() has ~20-40ns overhead on macOS, called twice per \
               allocation call here -- this is an order-of-magnitude measurement, not \
               a precise one."
    );
}

#[test]
fn historical_lob_episode_hash_identical_across_20_runs_same_seed() {
    // The "Phase S LOB hash" determinism guard called out explicitly by the
    // Phase S plan: real LobReplay data (not synthetic), same seed, run
    // repeatedly, must produce a bit-identical trajectory every time. This is
    // also the guard that would catch a bug in the new per-date MarketDataCache
    // (e.g. Arc-shared snapshot data accidentally mutated or racily reused).
    let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
    if !std::path::Path::new(lob_path).exists() {
        eprintln!("skipping: {lob_path} not present in this checkout");
        return;
    }
    let seed = 7;
    let make_hist_env = || {
        let cfg = EnvConfig {
            mode: RlMode::Historical,
            total_inventory: 1_000_000.0,
            total_steps: HISTORICAL_DEFAULT_STEPS,
            seed: 42,
            sigma_ref: 0.20,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: Some(lob_path.to_string()),
            btc_target: 50_000.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: true,
            fixed_size: true,
            no_time_features: true,
            episode_steps: None,
        };
        RlEnv::new(cfg)
    };

    let mut hashes = Vec::with_capacity(20);
    for _ in 0..20 {
        let mut env = make_hist_env();
        hashes.push(run_episode_and_hash(&mut env, seed));
    }
    let first = hashes[0];
    for (i, h) in hashes.iter().enumerate() {
        assert_eq!(
            *h, first,
            "run {i} produced a different historical-LOB trajectory hash than \
             run 0 for the same seed"
        );
    }
}

#[test]
fn synthetic_episode_hash_differs_across_seeds() {
    // Sanity check on the hash itself: it must not be a constant that would
    // make the "identical across runs" test trivially/vacuously pass.
    let mut env_a = make_env();
    let mut env_b = make_env();
    let hash_a = run_episode_and_hash(&mut env_a, 1);
    let hash_b = run_episode_and_hash(&mut env_b, 2);
    assert_ne!(
        hash_a, hash_b,
        "different seeds must produce different trajectories"
    );
}

fn make_env() -> RlEnv {
    let cfg = EnvConfig {
        mode: RlMode::Synthetic,
        total_inventory: 1_000_000.0,
        total_steps: 100,
        seed: 42,
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: None,
        btc_target: 500.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: false,
        fixed_size: false,
        no_time_features: true,
        episode_steps: None,
    };
    RlEnv::new(cfg)
}

#[test]
fn rl_env_reset_returns_twelve_dimensional_state() {
    let mut env = make_env();
    let r = env.reset(Some(42));
    assert_eq!(
        r.state.len(),
        12,
        "state vector must have exactly 12 dimensions (8 base + 2 impact + 2 regime)"
    );
    assert!(!r.done);
    assert_eq!(r.reward, 0.0);
}

#[test]
fn reset_on_missing_lob_file_terminates_episode_gracefully_instead_of_panicking() {
    // Phase S graceful-degradation test: a missing/corrupt historical data file
    // must force-terminate the episode with a defined penalty, not crash the
    // process (which would kill every other in-flight episode / the whole
    // training run).
    let cfg = EnvConfig {
        mode: RlMode::Historical,
        total_inventory: 1_000_000.0,
        total_steps: 100,
        seed: 42,
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: Some("TradeData/BookDepth/does-not-exist-fixture.csv".to_string()),
        btc_target: 500.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: true,
        fixed_size: true,
        no_time_features: true,
        episode_steps: None,
    };
    let mut env = RlEnv::new(cfg);
    let r = env.reset(Some(1));

    assert!(
        r.done,
        "episode must be marked done, not left half-initialised"
    );
    assert_eq!(
        r.reward, DATA_INGESTION_FAILURE_PENALTY,
        "must apply the defined ingestion-failure penalty, not a normal step reward"
    );
    assert_eq!(
        r.state.len(),
        12,
        "state vector must still be the correct width so the Python side doesn't \
         choke on a malformed observation"
    );
    let err = r.info.get("error").and_then(|v| v.as_str());
    assert!(
        err.is_some(),
        "info must explain what went wrong: {:?}",
        r.info
    );
    assert_eq!(
        r.info.get("terminated_reason").and_then(|v| v.as_str()),
        Some("data_ingestion_failure")
    );
}

#[test]
fn reset_on_lob_file_with_zero_valid_snapshots_terminates_gracefully() {
    // Distinct failure mode from "file missing": the file opens and has a valid
    // header, but every row fails to parse (e.g. a truncated/corrupted download),
    // leaving zero snapshots -- LobReplay::from_csv returns Err in this case too,
    // and it must be handled the same way, not panic.
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "rl_env_zero_snapshot_fixture_{}.csv",
        std::process::id()
    ));
    std::fs::write(
        &path,
        "timestamp,percentage,depth,notional\nnot-a-timestamp,-1,1.0,1.0\n",
    )
    .expect("failed to write test fixture");

    let cfg = EnvConfig {
        mode: RlMode::Historical,
        total_inventory: 1_000_000.0,
        total_steps: 100,
        seed: 7,
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: Some(path.to_string_lossy().to_string()),
        btc_target: 500.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: true,
        fixed_size: true,
        no_time_features: true,
        episode_steps: None,
    };
    let mut env = RlEnv::new(cfg);
    let r = env.reset(Some(1));

    let _ = std::fs::remove_file(&path);

    assert!(r.done);
    assert_eq!(r.reward, DATA_INGESTION_FAILURE_PENALTY);
    assert_eq!(
        r.info.get("terminated_reason").and_then(|v| v.as_str()),
        Some("data_ingestion_failure")
    );
}

#[test]
fn rl_env_action_zero_does_not_change_inventory() {
    let mut env = make_env();
    env.reset(Some(42));
    let before = env.remaining;
    let r = env.step(0);
    assert!(
        (env.remaining - before).abs() < 1e-9,
        "action=0 must not change inventory: before={before} after={}",
        env.remaining
    );
    assert!(
        r.reward < 0.0,
        "wait action still incurs inventory pressure"
    );
}

#[test]
fn rl_env_action_nineteen_liquidates_all_remaining() {
    let mut env = make_env();
    env.reset(Some(42));
    let r = env.step(19);
    // The synthetic book has finite depth, so a 1.0×inventory order may have residual.
    // The intent is that action 19 ATTEMPTS to liquidate all remaining; whatever fills,
    // the remaining inventory must shrink dramatically. Check < 1% of initial.
    assert!(
        env.remaining < 0.01 * 1_000_000.0,
        "action=19 must fill ~all inventory; remaining={}",
        env.remaining
    );
    assert!(r.info.get("fill_qty").is_some());
}

#[test]
fn rl_env_done_when_inventory_exhausted() {
    let mut env = make_env();
    env.reset(Some(42));
    let r = env.step(19);
    assert!(
        r.done,
        "after fully liquidating, done must be true (remaining = {})",
        env.remaining
    );
}

#[test]
fn rl_env_done_when_steps_exhausted() {
    let cfg = EnvConfig {
        mode: RlMode::Synthetic,
        total_inventory: 1_000_000.0,
        total_steps: 5, // very short horizon
        seed: 7,
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: None,
        btc_target: 500.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: false,
        fixed_size: false,
        no_time_features: true,
        episode_steps: None,
    };
    let mut env = RlEnv::new(cfg);
    env.reset(None);

    // Wait 4 times — stays under termination.
    for _ in 0..4 {
        let r = env.step(0);
        assert!(!r.done, "should not be done before exhausting steps");
    }
    // Fifth step should trigger time termination and forced liquidation.
    let r = env.step(0);
    assert!(r.done, "done must be true at the step horizon");
    assert!(
        r.info.get("forced_liquidation_qty").is_some(),
        "step-exhausted termination must report forced_liquidation_qty",
    );
}

#[test]
fn rl_env_reward_negative_for_adverse_fill() {
    let mut env = make_env();
    env.reset(Some(42));
    // Action 9 → 5% of remaining (50k units). Synthetic book has half-spread > 0
    // so the fill is below mid → fill_cost > 0 → reward strictly negative.
    let r = env.step(9);
    assert!(
        r.reward < 0.0,
        "adverse fill must produce negative reward; got {}",
        r.reward
    );
    // is_contribution should also be ≤ 0 (sum of fill_cost + perm_cost, negated).
    let isc = r
        .info
        .get("is_contribution")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    assert!(
        isc <= 0.0,
        "is_contribution must be ≤ 0 for a sell, got {isc}"
    );
}

#[test]
fn counterfactual_spread_ref_is_rescaled_not_raw_btc_dollar_scale() {
    // Regression guard for a real bug found quantifying the Task-3
    // wait-penalty jump: reset_counterfactual computed spread_ref from the
    // RAW (unrescaled, ~$40k-100k BTC-dollar-scale) snapshot, while every
    // per-step comparison against it uses the RESCALED (~100 simulation-
    // scale) book. That mismatch made state dim s3 collapse toward 0 after
    // the first step, and made the forced-liquidation punitive-price
    // formula floor at ~$0 instead of a genuinely punitive-but-nonzero
    // price. spread_ref must end up small (simulation-scale, comparable to
    // a ~100-scale half-spread), not raw-BTC-scale.
    let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
    if !std::path::Path::new(lob_path).exists() {
        eprintln!("skipping: {lob_path} not present in this checkout");
        return;
    }
    let cfg = EnvConfig {
        mode: RlMode::Counterfactual,
        total_inventory: 1_000_000.0,
        total_steps: 100,
        seed: 42,
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: Some(lob_path.to_string()),
        btc_target: 50_000.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: true,
        fixed_size: true,
        no_time_features: true,
        episode_steps: None,
    };
    let mut env = RlEnv::new(cfg);
    env.reset(Some(1));
    assert!(
        env.spread_ref < 5.0,
        "spread_ref must be simulation-scale (comparable to a ~100-scale \
         half-spread, so well under 5.0), not raw BTC-dollar-scale \
         (which would be tens): got {}",
        env.spread_ref
    );

    // And the state dim that depends on it (s3) must not collapse after the
    // first step -- it should stay in a sane tanh-bounded range throughout.
    let mut last_s3 = f64::NAN;
    for i in 0..5 {
        let r = env.step(0);
        last_s3 = r.state[3];
        assert!(
            r.state[3] > 0.05,
            "s3 (spread ratio) collapsed toward 0 at step {i} -- the \
             spread_ref scale-mismatch bug is back (got {})",
            r.state[3]
        );
    }
    assert!(last_s3.is_finite());
}

#[test]
fn regime_dims_separate_crash_from_calm_in_first_100_steps() {
    // Phase 3 fix regression guard (docs/proposals/research-roadmap.md, sections C/D):
    // before this fix, s10 (vol_regime) was measured to be pinned at
    // LOW/"normal" on every real date, calm or crash, for the entire
    // episode -- because sigma_ref was hardcoded at 0.20 while
    // current_sigma's annualisation was separately wrong by ~19x, so no
    // real date's ratio ever crossed the bucket thresholds. This test
    // is the "measure first-100-step separation before retraining"
    // step Step 0 requires: with CALIBRATED_SIGMA_REF and the corrected
    // annualisation, the first 100 steps of a severe crash date must
    // read as measurably more "stressed" (higher vol_signal) than a
    // held-out calm date -- the exact window in which a policy has to
    // decide how urgently to trade, per the standing-issues doc's
    // issue #6.
    let crash_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-08-05.csv";
    let calm_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-07-10.csv";
    if !std::path::Path::new(crash_path).exists() || !std::path::Path::new(calm_path).exists() {
        eprintln!("skipping: TradeData not present in this checkout");
        return;
    }

    fn mean_vol_signal_first_100_steps(lob_path: &str) -> f64 {
        let cfg = EnvConfig {
            mode: RlMode::Counterfactual,
            total_inventory: 1_000_000.0,
            total_steps: 100,
            seed: 42,
            sigma_ref: CALIBRATED_SIGMA_REF,
            gamma: 0.005,
            adversarial: false,
            lob_data_path: Some(lob_path.to_string()),
            btc_target: 50_000.0,
            adv_btc: 0.0,
            agg_trades_path: None,
            fixed_steps: true,
            fixed_size: true,
            no_time_features: true,
            episode_steps: None,
        };
        let mut env = RlEnv::new(cfg);
        env.reset(Some(0));
        let mut vals = Vec::new();
        for _ in 0..100 {
            // Action 0 (no trade) -- isolate the market-state signal from
            // any effect of the agent's own fills.
            let r = env.step(0);
            vals.push(r.state[10]);
            if r.done {
                break;
            }
        }
        vals.iter().sum::<f64>() / vals.len() as f64
    }

    let crash_mean = mean_vol_signal_first_100_steps(crash_path);
    let calm_mean = mean_vol_signal_first_100_steps(calm_path);

    assert!(
        crash_mean > calm_mean + 0.1,
        "vol_signal (s10) must read meaningfully more stressed on a severe \
         crash date (2024-08-05) than a held-out calm date (2024-07-10) \
         within the first 100 steps -- got crash_mean={crash_mean:.4}, \
         calm_mean={calm_mean:.4}. If this regresses, the sigma_ref \
         calibration or annualisation fix has been undone."
    );
}

#[test]
fn training_horizon_episode_fits_within_day_no_stale_tail() {
    // Start-offset bounding guard (Phase 2R). reset_historical picks
    //   start_offset = seed % (len/3)
    // and a training episode runs up to RANDOM_HORIZON_MAX (1440) snapshots.
    // For the episode to never trade against a stale (frozen) book, we need
    // start_offset + horizon <= len, i.e. len/3 + 1440 <= len -> len >= 2160.
    // Real days have ~2880 snapshots, so the worst-case training episode
    // (offset ~959 + 1440 = 2399) fits. This test guards that property: run a
    // full RANDOM_HORIZON_MAX-length episode from a large start offset with a
    // pure WAIT policy (market advances every step) and assert the book keeps
    // advancing -- no long frozen tail.
    //
    // NOTE (documented, not asserted here): the FIXED eval horizon
    // HISTORICAL_DEFAULT_STEPS (2880) does NOT satisfy this bound for
    // start_offset > 0; a wait-heavy eval episode at large start_offset trades
    // its tail against a frozen book. Evals in this project use small seeds
    // (start_offset 0-19, <1% of a 2880-step episode), so the effect is
    // negligible, but bounding start_offset <= len - total_steps would remove
    // it entirely.
    let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
    if !std::path::Path::new(lob_path).exists() {
        eprintln!("skipping: {lob_path} not present in this checkout");
        return;
    }
    let cfg = EnvConfig {
        mode: RlMode::Historical,
        total_inventory: 1_000_000.0,
        total_steps: RANDOM_HORIZON_MAX, // 1440, the worst-case training horizon
        seed: 900,                       // large start offset (900 % (len/3) ~ 900)
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: Some(lob_path.to_string()),
        btc_target: 50_000.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: true,
        fixed_size: true,
        no_time_features: true,
        episode_steps: None,
    };
    let mut env = RlEnv::new(cfg);
    env.reset(Some(900));
    let mut last_mid = f64::NAN;
    let mut stale = 0usize;
    let mut total = 0usize;
    loop {
        let r = env.step(0); // wait -> market advances each step
        let mid = r
            .info
            .get("book_mid")
            .and_then(|v| v.as_f64())
            .unwrap_or(f64::NAN);
        if (mid - last_mid).abs() < 1e-12 {
            stale += 1;
        }
        last_mid = mid;
        total += 1;
        if r.done {
            break;
        }
    }
    // A tiny number of stale steps (repeated identical mids) can occur naturally;
    // a premature-exhaustion tail would be hundreds. Assert well under 5%.
    assert!(
        (stale as f64) / (total as f64) < 0.05,
        "training-horizon episode ran a stale/frozen book for {stale}/{total} steps \
         -- start_offset + horizon exceeded the day's snapshot count (bounding bug)"
    );
}

#[test]
fn reward_higher_for_sell_fill_closer_to_mid() {
    // Sign-convention guard (Phase 2R). The fill reward is
    //   (fill_price - book_mid) * fill_qty / (arrival_price * total).
    // For a SELL, fill_price <= book_mid, so reward <= 0; a fill that stays
    // near the top of the book (avg price close to mid) must earn a HIGHER
    // (less negative) PER-UNIT reward than one that walks deep into the book.
    // Compared per-unit (reward / fill_qty) to isolate the slippage sign from
    // the quantity scaling, since a larger order fills both more qty AND at a
    // worse average price.
    let per_unit = |action: usize| -> f64 {
        let mut env = make_env();
        env.reset(Some(42));
        let r = env.step(action);
        let fq = r
            .info
            .get("fill_qty")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        assert!(fq > 1e-9, "action {action} must produce a fill");
        r.reward / fq
    };
    // action 5 = 1% of remaining (small, stays near mid);
    // action 16 = 50% of remaining (large, walks deep into the book).
    let shallow = per_unit(5);
    let deep = per_unit(16);
    assert!(
        shallow < 0.0 && deep < 0.0,
        "both sell fills must have negative reward (below mid): shallow={shallow}, deep={deep}"
    );
    assert!(
        shallow > deep,
        "a sell fill closer to mid must earn higher per-unit reward than a deeper \
         one: shallow(1%)={shallow} must be > deep(50%)={deep}"
    );
}

#[test]
fn rl_env_state_all_values_in_valid_range() {
    let mut env = make_env();
    env.reset(Some(99));
    // Walk 20 steps with a mix of actions and assert every state value is in range.
    let actions = [
        0, 9, 0, 5, 1, 0, 12, 3, 0, 7, 11, 0, 2, 6, 0, 8, 0, 4, 10, 0,
    ];
    for a in actions {
        let r = env.step(a);
        assert_eq!(r.state.len(), 12);
        // [0] inventory fraction in [0,1]
        assert!(
            r.state[0] >= 0.0 && r.state[0] <= 1.0,
            "s0 out of range: {}",
            r.state[0]
        );
        // [1] time fraction in [0,1]
        assert!(
            r.state[1] >= 0.0 && r.state[1] <= 1.0,
            "s1 out of range: {}",
            r.state[1]
        );
        // [2..=4] are tanh() of non-negative ratios → in [0, 1)
        for k in 2..=4 {
            assert!(
                r.state[k] >= 0.0 && r.state[k] < 1.0,
                "s{k} (tanh of non-negative ratio) out of range: {}",
                r.state[k]
            );
        }
        // [5] OFI in [-1,1]; always 0.0 in synthetic mode (no agg-trades loaded)
        assert!(
            r.state[5] >= -1.0 && r.state[5] <= 1.0,
            "s5 (OFI) out of range: {}",
            r.state[5]
        );
        // [6..=7] are signed tanh() → in (-1, 1)
        for k in 6..=7 {
            assert!(
                r.state[k] > -1.0 && r.state[k] < 1.0,
                "s{k} (signed tanh) out of range: {}",
                r.state[k]
            );
        }
        // [8..=9] impact dims: 0.0 in synthetic/historical modes
        assert_eq!(
            r.state[8], 0.0,
            "s8 (perm impact) must be 0 in synthetic mode"
        );
        assert_eq!(
            r.state[9], 0.0,
            "s9 (temp impact) must be 0 in synthetic mode"
        );
        // [10..=11] regime dims: tanh(ln(ratio)) → signed, in (-1, 1).
        // Phase 3 fix replaced the old 3-way {0.0, 0.5, 1.0} bucketing —
        // see regime_dims()'s doc comment.
        for k in 10..=11 {
            let v = r.state[k];
            assert!(
                v > -1.0 && v < 1.0,
                "s{k} (regime dim, signed tanh) out of range: {v}"
            );
        }
        if r.done {
            break;
        }
    }
}

fn make_env_with_time_features() -> RlEnv {
    let cfg = EnvConfig {
        mode: RlMode::Synthetic,
        total_inventory: 1_000_000.0,
        total_steps: 100,
        seed: 42,
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: None,
        btc_target: 500.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: false,
        fixed_size: false,
        no_time_features: false,
        episode_steps: None,
    };
    RlEnv::new(cfg)
}

#[test]
fn time_features_produce_16_dims() {
    let mut env = make_env_with_time_features();
    let r = env.reset(Some(42));
    assert_eq!(
        r.state.len(),
        16,
        "with time features active, state must have 16 dimensions (12 v5 + 4 time)"
    );
}

#[test]
fn no_time_features_produces_12_dims() {
    let mut env = make_env();
    let r = env.reset(Some(42));
    assert_eq!(
        r.state.len(),
        12,
        "--no-time-features must keep the v5-compatible 12-dim state vector"
    );
}

#[test]
fn time_of_day_in_range_zero_to_one() {
    let mut env = make_env_with_time_features();
    let r = env.reset(Some(42));
    let time_of_day = r.state[12];
    assert!(
        (0.0..=1.0).contains(&time_of_day),
        "time_of_day must be in [0,1], got {time_of_day}"
    );
}

#[test]
fn intraday_vol_ratio_dim_carries_no_information_beyond_s2_no_lookahead() {
    // Phase 2 acceptance criterion (Global Rule 1 + 7): an EXECUTABLE
    // leakage guard for state dim [13] (intraday_vol_ratio), not just
    // the Phase -1 audit's prose verdict. build_state() computes:
    //   s2  = tanh(current_sigma / sigma_ref)
    //   s13 = tanh(current_sigma / sigma_ref - 1.0)
    // i.e. s13 is a pure deterministic function of the SAME
    // current_sigma/sigma_ref ratio s2 already exposes -- it introduces
    // no new data source, so it cannot leak anything s2 doesn't already
    // (and s2's current_sigma is a bounded rolling-window estimator,
    // never a full-episode/future aggregate -- the classic "daily_vol"
    // lookahead bug this guard exists to catch would show up here as
    // s13 diverging from a pure function of s2 at some step).
    let mut env = make_env_with_time_features();
    let mut r = env.reset(Some(7));
    for _ in 0..20 {
        let s2 = r.state[2];
        let s13 = r.state[13];
        // Recover the ratio from s2, then recompute s13 from it.
        let ratio = s2.clamp(-0.999999, 0.999999).atanh();
        let expected_s13 = (ratio - 1.0).tanh();
        assert!(
            (s13 - expected_s13).abs() < 1e-9,
            "s13 ({s13}) must be a pure function of s2's ratio ({s2} -> ratio {ratio}), \
             got expected {expected_s13} -- any divergence means s13 is drawing on a \
             data source s2 doesn't already causally have"
        );
        if r.done {
            break;
        }
        r = env.step(5);
    }
}

#[test]
fn cyclical_encoding_bounded() {
    let mut env = make_env_with_time_features();
    let r = env.reset(Some(42));
    assert!(
        r.state[14].abs() <= 1.0 + 1e-6,
        "sin(time_of_day) component out of range: {}",
        r.state[14]
    );
    assert!(
        r.state[15].abs() <= 1.0 + 1e-6,
        "cos(time_of_day) component out of range: {}",
        r.state[15]
    );
}

#[test]
fn sin_cos_identity() {
    let mut env = make_env_with_time_features();
    let r = env.reset(Some(42));
    let identity = r.state[14].powi(2) + r.state[15].powi(2);
    assert!(
        (identity - 1.0).abs() < 0.001,
        "sin^2 + cos^2 must be ~1.0, got {identity}"
    );
}

// -----------------------------------------------------------------------
// Batched env (Phase S, ADR-006)
// -----------------------------------------------------------------------

fn synthetic_cfg_template() -> EnvConfig {
    EnvConfig {
        mode: RlMode::Synthetic,
        total_inventory: 1_000_000.0,
        total_steps: 50,
        seed: 42,
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: None,
        btc_target: 500.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: true,
        fixed_size: true,
        no_time_features: true,
        episode_steps: None,
    }
}

#[test]
fn batched_env_reset_and_step_match_n_single_instances() {
    // Correctness guard: BatchedRlEnv must produce EXACTLY the same numbers
    // as N independent RlEnv instances stepped one at a time -- batching is
    // a performance change, not a semantics change.
    let n = 4;
    let seeds: Vec<u64> = vec![1, 2, 3, 4];
    let actions: Vec<usize> = vec![0, 5, 10, 19];

    let mut batched = BatchedRlEnv::new(synthetic_cfg_template(), n);
    let reset_result = batched.reset_batch(&seeds);
    let step_result = batched.step_batch(&actions);

    for i in 0..n {
        let mut single = RlEnv::new(synthetic_cfg_template());
        let single_reset = single.reset(Some(seeds[i]));
        let single_step = single.step(actions[i]);

        let batch_reset_row = &reset_result.states[i * reset_result.d..(i + 1) * reset_result.d];
        let expected_reset: Vec<f32> = single_reset.state.iter().map(|&x| x as f32).collect();
        assert_eq!(
            batch_reset_row,
            expected_reset.as_slice(),
            "instance {i}: batched reset state != single-instance reset state"
        );

        let batch_step_row = &step_result.states[i * step_result.d..(i + 1) * step_result.d];
        let expected_step: Vec<f32> = single_step.state.iter().map(|&x| x as f32).collect();
        assert_eq!(
            batch_step_row,
            expected_step.as_slice(),
            "instance {i}: batched step state != single-instance step state"
        );
        assert_eq!(
            step_result.rewards[i], single_step.reward as f32,
            "instance {i}: batched reward != single-instance reward"
        );
    }
}

#[test]
fn reset_indices_only_resets_specified_instances() {
    // Correctness guard for the SB3 VecEnv auto-reset-on-done contract:
    // reset_indices must reset ONLY the specified instances and leave
    // every other instance's in-progress episode completely untouched
    // (same remaining inventory, same step_idx) -- a batched training
    // wrapper that reset the whole batch when only one instance finished
    // would silently truncate every other instance's episode.
    let n = 4;
    let mut batched = BatchedRlEnv::new(synthetic_cfg_template(), n);
    batched.reset_batch(&[10, 20, 30, 40]);
    // Advance all 4 by a few steps so they have observable in-progress state.
    batched.step_batch(&[5, 5, 5, 5]);
    batched.step_batch(&[5, 5, 5, 5]);
    let before = batched.step_batch(&[5, 5, 5, 5]);

    // Reset only instances 1 and 3.
    let reset_result = batched.reset_indices(&[1, 3], &[99, 100]);
    assert_eq!(reset_result.n, 2);

    // Instances 1 and 3 must match what a fresh reset(seed) produces.
    let mut fresh1 = RlEnv::new(synthetic_cfg_template());
    let fresh1_state = fresh1.reset(Some(99));
    let expected1: Vec<f32> = fresh1_state.state.iter().map(|&x| x as f32).collect();
    assert_eq!(
        &reset_result.states[0..reset_result.d],
        expected1.as_slice()
    );

    let mut fresh3 = RlEnv::new(synthetic_cfg_template());
    let fresh3_state = fresh3.reset(Some(100));
    let expected3: Vec<f32> = fresh3_state.state.iter().map(|&x| x as f32).collect();
    assert_eq!(
        &reset_result.states[reset_result.d..2 * reset_result.d],
        expected3.as_slice()
    );

    // Instances 0 and 2 must be UNTOUCHED: stepping them further must
    // continue their existing (3-steps-in) trajectory, not a fresh episode.
    let after = batched.step_batch(&[5, 5, 5, 5]);
    let mut continued0 = RlEnv::new(synthetic_cfg_template());
    continued0.reset(Some(10));
    continued0.step(5);
    continued0.step(5);
    continued0.step(5);
    let expected0_step4 = continued0.step(5);
    let expected0: Vec<f32> = expected0_step4.state.iter().map(|&x| x as f32).collect();
    assert_eq!(
        &after.states[0 * after.d..1 * after.d],
        expected0.as_slice(),
        "instance 0 (not reset) must continue its pre-existing trajectory unchanged"
    );

    let mut continued2 = RlEnv::new(synthetic_cfg_template());
    continued2.reset(Some(30));
    continued2.step(5);
    continued2.step(5);
    continued2.step(5);
    let expected2_step4 = continued2.step(5);
    let expected2: Vec<f32> = expected2_step4.state.iter().map(|&x| x as f32).collect();
    assert_eq!(
        &after.states[2 * after.d..3 * after.d],
        expected2.as_slice(),
        "instance 2 (not reset) must continue its pre-existing trajectory unchanged"
    );
    let _ = before; // used only to advance state; no direct assertion needed
}

#[test]
fn batch_arrow_round_trip_preserves_values() {
    // Encode a BatchStepResult to Arrow, decode the request columns back,
    // and confirm nothing is lost/reordered in the RecordBatch conversion
    // itself, independent of the socket transport.
    let result = BatchStepResult {
        states: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], // 3 rows x 2 dims
        rewards: vec![-0.5, -0.25, -0.75],
        dones: vec![false, true, false],
        n: 3,
        d: 2,
    };
    let batch = batch_arrow::encode_result(&result);
    assert_eq!(batch.num_rows(), 3);
    assert_eq!(batch.num_columns(), 4); // s0, s1, reward, done

    let seeds = vec![10u64, 20, 30];
    let req = batch_arrow::encode_reset_request(&seeds);
    assert_eq!(batch_arrow::decode_reset_request(&req), seeds);

    let actions = vec![0usize, 5, 19];
    let req = batch_arrow::encode_step_request(&actions);
    assert_eq!(batch_arrow::decode_step_request(&req), actions);
}

#[test]
fn batch_socket_server_round_trip() {
    // End-to-end: spin up run_batch_server on a real Unix socket in a
    // background thread, connect as a client using the same length-prefixed
    // framing a real client (Python's BatchedStrataExecEnv) uses, and
    // confirm reset_batch then step_batch round-trip correctly across
    // MULTIPLE exchanges on one persistent connection. This specific
    // scenario (2nd+ message on a shared connection) is what caught the
    // "pyarrow reads ahead past the stream boundary" bug that motivated
    // the length-prefixed framing in the first place.
    use arrow::ipc::reader::StreamReader;
    use std::io::Cursor;
    use std::os::unix::net::UnixStream;

    let socket_path = std::env::temp_dir().join(format!(
        "rl_env_batch_test_{}_{}.sock",
        std::process::id(),
        42 // fixed per-test suffix; unique per test fn via the temp_dir cleanup below
    ));
    let socket_path_str = socket_path.to_string_lossy().to_string();
    let _ = std::fs::remove_file(&socket_path);

    let server_path = socket_path_str.clone();
    let handle = std::thread::spawn(move || {
        run_batch_server(synthetic_cfg_template(), 3, &server_path);
    });

    // Give the server a moment to bind before connecting.
    let mut stream = None;
    for _ in 0..50 {
        if let Ok(s) = UnixStream::connect(&socket_path_str) {
            stream = Some(s);
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    let mut stream = stream.expect("failed to connect to batch server socket");

    let request_response = |stream: &mut UnixStream, req: arrow::record_batch::RecordBatch| {
        let bytes = encode_batch_to_bytes(&req);
        write_framed_message(stream, &bytes).unwrap();
        let resp_bytes = read_framed_message(stream).expect("server closed connection");
        let reader = StreamReader::try_new(Cursor::new(resp_bytes), None).unwrap();
        let batches: Vec<_> = reader.collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(batches.len(), 1);
        batches.into_iter().next().unwrap()
    };

    // -- reset_batch --
    let seeds = vec![1u64, 2, 3];
    let response = request_response(&mut stream, batch_arrow::encode_reset_request(&seeds));
    assert_eq!(response.num_rows(), 3);
    assert_eq!(
        response.schema().field(0).name(),
        "s0",
        "response columns must be s0..s{{d-1}}, reward, done"
    );

    // -- step_batch (2nd message on the same connection) --
    let actions = vec![0usize, 1, 2];
    let response = request_response(&mut stream, batch_arrow::encode_step_request(&actions));
    assert_eq!(response.num_rows(), 3);

    // -- a 3rd message, to be extra sure framing holds beyond just "2" --
    let actions = vec![5usize, 6, 7];
    let response = request_response(&mut stream, batch_arrow::encode_step_request(&actions));
    assert_eq!(response.num_rows(), 3);

    // -- reset_indices (partial reset over the wire, the SB3 VecEnv
    // auto-reset path) -- 4th message, only 2 of the 3 instances --
    let response = request_response(
        &mut stream,
        batch_arrow::encode_reset_indices_request(&[0, 2], &[111, 222]),
    );
    assert_eq!(
        response.num_rows(),
        2,
        "partial reset must return exactly len(indices) rows"
    );

    drop(stream); // closes the connection; server loop exits on next read
    let _ = handle.join();
    let _ = std::fs::remove_file(&socket_path);
}

// ── Phase 2R regression tests: reward loophole + price_factor rescaling ──

#[test]
fn forced_liquidation_is_not_reward_superior_to_trading() {
    // Regression test for the reward loophole behind the v6 training collapse
    // (Phase 2R incident): forced liquidation used to be priced at a flat
    // `gamma * (forced_liq_qty/total)` penalty (0.005 for liquidating 100%),
    // completely decoupled from the real cost of dumping that size into the
    // book. Against real historical LOB depth with a large-relative-to-depth
    // order, cumulative execution slippage clearly exceeds that flat 0.005 cap
    // -- so "never trade, eat the flat penalty" was reward-superior to actually
    // executing. The RL agent found and converged on exactly this shortcut.
    //
    // Invariant under the fix: spreading execution across the episode must
    // score a HIGHER (less negative) cumulative reward than never trading and
    // being forced to dump 100% of inventory in one shot at the horizon.
    //
    // Uses real historical LOB data (not synthetic) because the synthetic
    // book's depth is generous enough relative to a 1M-unit order that
    // trading slippage never gets close to gamma=0.005 either way -- the loophole
    // only actually bites at realistic size-vs-depth ratios, which is exactly
    // what happened in the real v6 training run this test guards against.
    let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
    if !std::path::Path::new(lob_path).exists() {
        eprintln!("skipping: {lob_path} not present in this checkout");
        return;
    }
    let make_cfg = || EnvConfig {
        mode: RlMode::Historical,
        total_inventory: 5_000_000.0,
        total_steps: HISTORICAL_DEFAULT_STEPS,
        seed: 42,
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: Some(lob_path.to_string()),
        btc_target: 50_000.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: true,
        fixed_size: true,
        no_time_features: true,
        episode_steps: None,
    };

    // Never-trade policy: action 0 every step, forced-liquidated at the horizon.
    let mut never_trade_env = RlEnv::new(make_cfg());
    never_trade_env.reset(Some(1));
    let mut never_trade_reward = 0.0;
    loop {
        let r = never_trade_env.step(0);
        never_trade_reward += r.reward;
        if r.done {
            assert!(
                r.info.get("forced_liquidation_qty").is_some(),
                "never-trade policy must end in forced liquidation of the full size"
            );
            break;
        }
    }

    // Spread-execution policy: sell 15% of *remaining* inventory each step
    // (ACTION_FRACTIONS[12] = 0.15) -- winds down gradually instead of dumping
    // everything at once.
    let mut trading_env = RlEnv::new(make_cfg());
    trading_env.reset(Some(1));
    let mut trading_reward = 0.0;
    loop {
        let r = trading_env.step(12);
        trading_reward += r.reward;
        if r.done {
            break;
        }
    }

    assert!(
        trading_reward > never_trade_reward,
        "trading policy (reward={trading_reward}) must score better than the \
         never-trade policy (reward={never_trade_reward}) -- if this regresses, \
         the forced-liquidation penalty is once again cheaper than real execution"
    );
}

#[test]
fn historical_price_factor_is_fixed_at_episode_start_not_recomputed_per_step() {
    // The bug: advance_market()'s historical branch used to recompute
    // `price_factor = 100.0 / new_raw_mid` fresh on every snapshot, which
    // renormalises the rescaled book's mid back to ≈100 (== arrival) at every
    // step regardless of real price movement. The fix fixes price_factor once
    // at reset_historical and reuses it for the whole episode.
    let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
    if !std::path::Path::new(lob_path).exists() {
        eprintln!("skipping: {lob_path} not present in this checkout");
        return;
    }
    let cfg = EnvConfig {
        mode: RlMode::Historical,
        total_inventory: 1_000_000.0,
        total_steps: HISTORICAL_DEFAULT_STEPS,
        seed: 42,
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: Some(lob_path.to_string()),
        btc_target: 50_000.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: true,
        fixed_size: true,
        no_time_features: true,
        episode_steps: None,
    };
    let mut env = RlEnv::new(cfg);
    env.reset(Some(42));
    let factor_at_reset = env.price_factor;

    for _ in 0..50 {
        let r = env.step(0); // never trade -- pure market observation
        if r.done {
            break;
        }
    }
    assert_eq!(
        env.price_factor, factor_at_reset,
        "price_factor must stay fixed for the whole episode, not be recomputed per snapshot"
    );

    // With the factor fixed, the rescaled book's mid should drift away from the
    // arrival price (100.0) as the real BTC price moves over the day, instead of
    // being renormalised back to ~100 at every step.
    let mid_after = env
        .current_book
        .as_ref()
        .and_then(|b| b.mid_price())
        .unwrap_or(100.0);
    assert!(
        (mid_after - 100.0).abs() > 1e-6,
        "rescaled book mid should reflect real intraday price drift, not stay \
         pinned at 100.0 (got {mid_after})"
    );
}

#[test]
fn forced_liquidation_is_pct_reflects_real_execution_not_pinned_zero() {
    // The bug this guards against: forced liquidation used to price the entire
    // remaining size at `current_book.mid_price()` without ever crossing the
    // book, and (compounded by the price_factor bug) that mid was always ≈
    // arrival -- so a fully-forced-liquidated episode scored total_IS_pct ≈ 0%
    // regardless of real market movement or spread cost. This is exactly what
    // produced the `-0.0000%` rows in the Phase 2 ablation.
    let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
    if !std::path::Path::new(lob_path).exists() {
        eprintln!("skipping: {lob_path} not present in this checkout");
        return;
    }
    let cfg = EnvConfig {
        mode: RlMode::Historical,
        total_inventory: 1_000_000.0,
        total_steps: 10, // short horizon so we hit forced liquidation quickly
        seed: 42,
        sigma_ref: 0.20,
        gamma: 0.005,
        adversarial: false,
        lob_data_path: Some(lob_path.to_string()),
        btc_target: 50_000.0,
        adv_btc: 0.0,
        agg_trades_path: None,
        fixed_steps: true,
        fixed_size: true,
        no_time_features: true,
        episode_steps: None,
    };
    let mut env = RlEnv::new(cfg);
    env.reset(Some(42));

    let mut last = None;
    for _ in 0..10 {
        last = Some(env.step(0)); // never trade -- entire inventory force-liquidated
    }
    let r = last.unwrap();
    assert!(r.done);
    assert!(
        r.info.get("forced_liquidation_qty").is_some(),
        "expected forced liquidation at the horizon"
    );
    let is_pct = r
        .info
        .get("total_IS_pct")
        .and_then(|v| v.as_f64())
        .expect("done step must report total_IS_pct");
    assert!(
        is_pct.abs() > 1e-4,
        "forced-liquidation IS must be non-zero -- dumping 100% of inventory into \
         the book must cross real depth/spread, not price at a static mid \
         (got {is_pct})"
    );
}

mod cli {
    use crate::config::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn accepts_every_flag_python_callers_pass() {
        let cfg = parse_args(&args(&[
            "--rl-mode",
            "Counterfactual",
            "--lob-data",
            "b.csv",
            "--agg-trades",
            "a.csv",
            "--btc-target",
            "50000",
            "--total-steps",
            "400",
            "--fixed-steps",
            "--fixed-size",
            "--no-time-features",
            "--seed",
            "7",
            "--batch-size",
            "4",
            "--socket-path",
            "/tmp/s.sock",
            "--print-calibration",
        ]))
        .expect("valid arguments must parse");
        assert!(matches!(cfg.mode, RlMode::Counterfactual));
        assert_eq!(cfg.total_steps, 400);
        assert_eq!(cfg.btc_target, 50_000.0);
        assert_eq!(cfg.seed, 7);
        assert!(cfg.fixed_steps && cfg.fixed_size && cfg.no_time_features);
    }

    #[test]
    fn replay_modes_keep_their_defaults() {
        let cfg = parse_args(&args(&["--rl-mode", "historical"])).expect("parses");
        assert_eq!(cfg.total_steps, HISTORICAL_DEFAULT_STEPS);
        assert_eq!(cfg.sigma_ref, CALIBRATED_SIGMA_REF);
        let cfg =
            parse_args(&args(&["--rl-mode", "historical", "--sigma-ref", "0.3"])).expect("parses");
        assert_eq!(cfg.sigma_ref, 0.3);
        let cfg = parse_args(&[]).expect("no arguments is valid");
        assert!(matches!(cfg.mode, RlMode::Synthetic));
        assert_eq!(cfg.sigma_ref, DEFAULT_SIGMA);
    }

    #[test]
    fn rejects_unknown_flag() {
        let err = parse_args(&args(&["--jump-diffusion"])).expect_err("must reject");
        assert!(err.contains("--jump-diffusion"), "{err}");
    }

    #[test]
    fn rejects_unknown_mode_instead_of_falling_back_to_synthetic() {
        let err = parse_args(&args(&["--rl-mode", "counterfacual"])).expect_err("must reject");
        assert!(err.contains("counterfacual"), "{err}");
    }

    #[test]
    fn rejects_unparseable_number_instead_of_using_default() {
        assert!(parse_args(&args(&["--btc-target", "abc"])).is_err());
        assert!(parse_args(&args(&["--total-steps", "-5"])).is_err());
    }

    #[test]
    fn rejects_adversarial_in_counterfactual_mode_only() {
        let err = parse_args(&args(&["--rl-mode", "counterfactual", "--adversarial"]))
            .expect_err("must reject");
        assert!(err.contains("counterfactual"), "{err}");
        for mode in ["historical", "synthetic"] {
            let cfg = parse_args(&args(&["--rl-mode", mode, "--adversarial"]))
                .expect("adversarial is supported here");
            assert!(cfg.adversarial);
        }
    }

    #[test]
    fn rejects_missing_value() {
        let err = parse_args(&args(&["--lob-data"])).expect_err("must reject");
        assert!(err.contains("requires a value"), "{err}");
    }
}

#[test]
fn forced_liquidation_at_data_exhaustion_walks_the_last_real_book() {
    // Regression guard: a fixed-horizon counterfactual episode that starts
    // at a non-zero offset runs past the day's last snapshot before its
    // clock ends. The adjusted book is then empty, and the forced dump used
    // to fill nothing and price 100% of the remainder by the punitive
    // fallback (mid - 5 * spread_ref) -- cheap enough that never trading
    // beat competent execution. The dump must consume the last real book.
    let lob_path = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
    if !std::path::Path::new(lob_path).exists() {
        eprintln!("skipping: {lob_path} not present in this checkout");
        return;
    }
    let cfg = EnvConfig {
        mode: RlMode::Counterfactual,
        lob_data_path: Some(lob_path.to_string()),
        btc_target: 50_000.0,
        fixed_steps: true,
        fixed_size: true,
        no_time_features: true,
        episode_steps: None,
        ..EnvConfig::default()
    };
    let mut env = RlEnv::new(cfg);
    env.reset(Some(17)); // start offset 17: data runs out before the clock
    let mut last = env.step(0);
    while !last.done {
        last = env.step(0); // wait every step
    }
    let info = &last.info;
    let filled = info["forced_liq_filled_qty"].as_f64().unwrap_or(0.0);
    assert!(
        info["forced_liquidation_qty"].as_f64().unwrap_or(0.0) > 0.0,
        "all-wait episode must end in a forced liquidation: {info}"
    );
    assert!(
        filled > 0.0,
        "forced dump at data exhaustion must fill against real depth, got {filled}"
    );
}

mod protocol_v3 {
    use crate::config::*;
    use crate::env::*;

    const LOB: &str = "TradeData/BookDepth/BTCUSDT-bookDepth-2024-01-15.csv";
    const H: usize = 1920;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    fn v3_env() -> Option<RlEnv> {
        if !std::path::Path::new(LOB).exists() {
            eprintln!("skipping: {LOB} not present in this checkout");
            return None;
        }
        let cfg = parse_args(&args(&[
            "--rl-mode",
            "counterfactual",
            "--lob-data",
            LOB,
            "--btc-target",
            "50000",
            "--fixed-size",
            "--episode-steps",
            "1920",
        ]))
        .expect("valid v3 arguments");
        Some(RlEnv::new(cfg))
    }

    #[test]
    fn episode_steps_parses_and_validates() {
        let cfg = parse_args(&args(&[
            "--rl-mode",
            "historical",
            "--episode-steps",
            "1920",
        ]))
        .expect("parses");
        assert_eq!(cfg.episode_steps, Some(1920));
        assert_eq!(cfg.total_steps, 1920);
        assert!(cfg.fixed_steps);
        for bad in [
            &["--rl-mode", "historical", "--episode-steps", "0"][..],
            &["--rl-mode", "synthetic", "--episode-steps", "100"][..],
            &[
                "--rl-mode",
                "historical",
                "--episode-steps",
                "100",
                "--total-steps",
                "50",
            ][..],
        ] {
            assert!(parse_args(&args(bad)).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn latest_start_runs_full_horizon_and_ends_on_real_data() {
        // The data-exhaustion bug class: every v3 episode must last exactly H
        // steps and its forced dump must hit a non-empty book.
        let Some(mut env) = v3_env() else { return };
        let r = env.reset_at(Some(0), Some(1.0));
        let max_start = r.info["max_start"].as_u64().unwrap_or(0) as usize;
        assert_eq!(
            r.info["start_offset"].as_u64().map(|v| v as usize),
            Some(max_start)
        );
        let (mut steps, mut last) = (0usize, r);
        while !last.done {
            last = env.step(0);
            steps += 1;
        }
        assert_eq!(steps, H);
        assert!(last.info["adjusted_book_depth"].as_f64().unwrap_or(0.0) > 0.0);
        assert!(last.info["forced_liq_filled_qty"].as_f64().unwrap_or(0.0) > 0.0);
    }

    #[test]
    fn seeds_spread_starts_across_the_window() {
        let Some(mut env) = v3_env() else { return };
        let starts: Vec<u64> = (0..8)
            .map(|s| {
                env.reset(Some(s)).info["start_offset"]
                    .as_u64()
                    .unwrap_or(0)
            })
            .collect();
        let distinct: std::collections::BTreeSet<_> = starts.iter().collect();
        assert!(
            distinct.len() >= 7,
            "seeds must give different starts: {starts:?}"
        );
        assert!(
            starts.iter().any(|&s| s > 480),
            "starts must reach the late window: {starts:?}"
        );
    }

    #[test]
    fn start_frac_is_rejected_outside_protocol_v3_or_range() {
        let Some(mut env) = v3_env() else { return };
        let r = env.reset_at(Some(0), Some(1.5));
        assert!(
            r.done
                && r.info["error"]
                    .as_str()
                    .unwrap_or("")
                    .contains("start_frac")
        );

        let legacy = parse_args(&args(&["--rl-mode", "counterfactual", "--lob-data", LOB]))
            .expect("valid legacy arguments");
        let mut env = RlEnv::new(legacy);
        let r = env.reset_at(Some(0), Some(0.5));
        assert!(
            r.done
                && r.info["error"]
                    .as_str()
                    .unwrap_or("")
                    .contains("--episode-steps")
        );
    }
}
