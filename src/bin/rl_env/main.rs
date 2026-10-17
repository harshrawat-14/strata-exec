// rl_env — Long-running RL environment server.
//
// Python controls this process via line-delimited JSON over stdin/stdout.
// One JSON object per line in both directions.
//
//   reset:  {"reset": true, "seed": 42}
//   action: {"action": <0..=19>}
//
// The response is a JSON object {state, reward, done, info}.

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

/// Timing-instrumented allocator (separate from dhat-heap, which counts
/// allocations/bytes but not wall time spent inside them). Wraps the system
/// allocator, timing every alloc/dealloc/realloc call via `Instant::now()`
/// deltas summed into atomics, so a step-loop benchmark can report what
/// fraction of its own wall time is actually spent allocating -- the
/// follow-up measurement to the dhat-heap allocation-COUNT numbers in
/// docs/performance/baseline-profiling.md. `Instant::now()` itself has
/// measurable overhead (~20-40ns on macOS via mach_absolute_time), which is
/// a source of over-counting on very cheap allocations -- treat this as an
/// order-of-magnitude answer, documented as such wherever it's reported.
#[cfg(feature = "alloc-timing")]
mod alloc_timing {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;

    pub static ALLOC_NANOS: AtomicU64 = AtomicU64::new(0);
    pub static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);

    // Read only by the allocation-timing benchmark test.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn reset() {
        ALLOC_NANOS.store(0, Ordering::SeqCst);
        ALLOC_CALLS.store(0, Ordering::SeqCst);
    }

    // Read only by the allocation-timing benchmark test.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn snapshot() -> (u64, u64) {
        (ALLOC_NANOS.load(Ordering::SeqCst), ALLOC_CALLS.load(Ordering::SeqCst))
    }

    pub struct TimingAlloc;

    unsafe impl GlobalAlloc for TimingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let t0 = Instant::now();
            let ptr = System.alloc(layout);
            ALLOC_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ptr
        }
        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            let t0 = Instant::now();
            System.dealloc(ptr, layout);
            ALLOC_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let t0 = Instant::now();
            let p = System.realloc(ptr, layout, new_size);
            ALLOC_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            p
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            let t0 = Instant::now();
            let ptr = System.alloc_zeroed(layout);
            ALLOC_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
            ptr
        }
    }
}

#[cfg(feature = "alloc-timing")]
#[global_allocator]
static ALLOC_TIMING: alloc_timing::TimingAlloc = alloc_timing::TimingAlloc;

mod batch_arrow;
mod batched;
mod config;
mod env;
mod market_cache;
mod transport;
#[cfg(test)]
mod tests;

use std::io::{self, BufRead};

use serde_json::json;


use crate::config::*;
use crate::env::*;
use crate::transport::*;

fn main() {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    // Single source of truth for the action->fraction mapping. Python's
    // classical-baseline policies (TWAP/Heuristic/AC-Optimal/AdaptiveAC) used
    // to hand-maintain their OWN copies of this table (V5_FRACTIONS/
    // V6_FRACTIONS in rl/evaluate.py) to convert a target sell fraction into
    // an action index -- which silently diverged from this real table (17/20
    // indices wrong for one of the two mirrors), so those policies were
    // picking action indices under an assumption the environment never
    // honored. Fixed by having Python query this exact array instead of
    // maintaining its own copy. See docs/experiments/02-reward-objective-audit.md.
    if std::env::args().any(|a| a == "--print-action-fractions") {
        println!("{}", json!(ACTION_FRACTIONS));
        return;
    }

    // Same single-source-of-truth mechanism as --print-action-fractions above,
    // for the scalar constants Python-side policies need. Added Phase 3 after
    // finding instance #5 of this project's canonical-value bug class: the
    // Step 0 fix recalibrated sigma_ref in Rust (0.20 -> CALIBRATED_SIGMA_REF)
    // while `rl/evaluate.py`'s AdaptiveACPolicy and `rl/batched_environment.py`
    // each kept their own hardcoded `sigma_ref: float = 0.20` default that no
    // call site ever overrode -- so the Python baseline was reconstructing
    // volatility against a reference the environment had stopped using. Any
    // future scalar Python needs from this binary belongs here, not in a
    // hand-copied Python default.
    if std::env::args().any(|a| a == "--print-env-constants") {
        println!(
            "{}",
            json!({
                // sigma_ref actually in force, BY MODE -- these differ, and the
                // caller must pick the one matching the mode it is running.
                "calibrated_sigma_ref": CALIBRATED_SIGMA_REF,
                "default_sigma": DEFAULT_SIGMA,
                "trading_steps_per_year": TRADING_STEPS_PER_YEAR,
            })
        );
        return;
    }

    let cfg = parse_cli();

    // Step 3 (sqrt-impact baseline): the per-date calibrated impact_coefficient
    // (Y), sigma, and adv_btc used by counterfactual mode are computed inside
    // reset_counterfactual (via CounterfactualLobConfig::calibrated) and, until
    // now, only ever printed to stderr as a diagnostic -- no caller could read
    // them back. A numerically-optimized baseline needs these exact fitted
    // values (not a second, hand-copied estimate -- see "Standing rules" in
    // docs/proposals/research-roadmap.md on canonical values). Reuses reset_counterfactual
    // unmodified, so there is exactly one place that computes calibration.
    if std::env::args().any(|a| a == "--print-calibration") {
        let mut env = RlEnv::new(cfg);
        let _ = env.reset(Some(0));
        let cf = env.cf_lob.as_ref().expect(
            "--print-calibration requires counterfactual mode with --lob-data \
             (and ideally --agg-trades, or calibration falls back to defaults)",
        );
        let c = cf.config();
        println!(
            "{}",
            json!({
                "impact_coefficient": c.impact_coefficient,
                "sigma": c.sigma,
                "adv_btc": c.adv_btc,
                "rho": c.rho,
                "permanent_fraction": c.permanent_fraction,
                "temporary_fraction": c.temporary_fraction,
                "dt": c.dt,
                "btc_scale": c.btc_scale,
            })
        );
        return;
    }

    if let Some((batch_size, socket_path)) = parse_batch_args() {
        run_batch_server(cfg, batch_size, &socket_path);
        return;
    }

    let mut env = RlEnv::new(cfg);

    let stdin = io::stdin();
    let mut stdin = stdin.lock();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    // Reused across iterations (cleared, not reallocated) instead of
    // BufRead::lines(), which allocates a fresh String per line -- one fewer
    // allocation per step on the input side of the hot path (measured via
    // dhat-heap, docs/performance/baseline-profiling.md).
    let mut line_buf = String::with_capacity(256);
    loop {
        line_buf.clear();
        match stdin.read_line(&mut line_buf) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(_) => break,
        }
        let trimmed = line_buf.trim();
        if trimmed.is_empty() {
            continue;
        }
        let msg: InMsg = match serde_json::from_str(trimmed) {
            Ok(m) => m,
            Err(e) => {
                let err = OutMsg {
                    state: vec![],
                    reward: 0.0,
                    done: false,
                    info: json!({"error": format!("invalid JSON: {e}")}),
                };
                let _ = write_msg(&mut stdout, &err);
                continue;
            }
        };

        let result = if msg.reset {
            env.reset(msg.seed)
        } else if let Some(a) = msg.action {
            env.step(a)
        } else {
            StepResult {
                state: vec![],
                reward: 0.0,
                done: false,
                info: json!({"error": "expected 'reset' or 'action'"}),
            }
        };

        if write_msg(&mut stdout, &step_result_to_msg(result)).is_err() {
            break;
        }
    }
}
