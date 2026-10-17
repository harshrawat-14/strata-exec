//! Batched env (Phase S, ADR-006): N instances per process sharing one
//! MarketDataCache, driven over Arrow instead of JSON-over-stdio.

use std::cell::RefCell;
use std::rc::Rc;




use crate::config::*;
use crate::market_cache::MarketDataCache;
use crate::env::*;

// ---------------------------------------------------------------------------
// Batched env (Phase S, ADR-006) — N instances per process, sharing one
// MarketDataCache, transported via Arrow RecordBatch over a Unix domain
// socket instead of JSON-over-stdio. Additive: the default (no --batch-size)
// code path above is completely unchanged.
// ---------------------------------------------------------------------------

/// N `RlEnv` instances stepped together, sharing one `MarketDataCache` so a
/// date replayed by multiple instances in the batch is parsed once, not N
/// times (see ADR-006 and `MarketDataCache`'s doc comment).
pub struct BatchedRlEnv {
    envs: Vec<RlEnv>,
    n_state_dims: usize,
}

/// Column-oriented result of one `reset_batch`/`step_batch` call: `states` is
/// row-major N×D (state dims cast to f32 -- the wire/observation precision the
/// Python/torch side trains at; internal computation stays f64), `rewards`/
/// `dones` are length N.
pub struct BatchStepResult {
    pub states: Vec<f32>,
    pub rewards: Vec<f32>,
    pub dones: Vec<bool>,
    pub n: usize,
    pub d: usize,
}

impl BatchedRlEnv {
    pub fn new(cfg_template: EnvConfig, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be >= 1");
        let cache = Rc::new(RefCell::new(MarketDataCache::default()));
        let n_state_dims = if cfg_template.no_time_features { 12 } else { 16 };
        let envs = (0..batch_size)
            .map(|_| RlEnv::new_with_shared_cache(cfg_template.clone(), Rc::clone(&cache)))
            .collect();
        Self { envs, n_state_dims }
    }


    pub fn reset_batch(&mut self, seeds: &[u64]) -> BatchStepResult {
        assert_eq!(seeds.len(), self.envs.len(), "seeds.len() must equal batch_size");
        self.collect(|env, i| env.reset(Some(seeds[i])))
    }

    pub fn step_batch(&mut self, actions: &[usize]) -> BatchStepResult {
        assert_eq!(actions.len(), self.envs.len(), "actions.len() must equal batch_size");
        self.collect(|env, i| env.step(actions[i]))
    }

    /// Reset only the given subset of instances (identified by index into the
    /// batch), each with its own seed, leaving every other instance's
    /// in-progress episode untouched. Required for the SB3 `VecEnv` contract:
    /// when instances finish episodes at different step counts (the normal
    /// case whenever `fixed_steps=False`, i.e. real training with randomised
    /// per-episode horizons -- not the fixed-horizon eval configuration Phase
    /// 0's harness uses), only the just-finished sub-envs auto-reset, not the
    /// whole batch. Returns a result sized to `indices.len()`, not
    /// `batch_size` -- the caller splices these rows into the right batch
    /// positions itself (see `BatchedStrataExecVecEnv.step_wait` in
    /// rl/batched_vec_env.py).
    pub fn reset_indices(&mut self, indices: &[usize], seeds: &[u64]) -> BatchStepResult {
        assert_eq!(indices.len(), seeds.len(), "indices.len() must equal seeds.len()");
        let d = self.n_state_dims;
        let mut states = Vec::with_capacity(indices.len() * d);
        let mut rewards = Vec::with_capacity(indices.len());
        let mut dones = Vec::with_capacity(indices.len());
        for (k, &idx) in indices.iter().enumerate() {
            let r = self.envs[idx].reset(Some(seeds[k]));
            debug_assert_eq!(r.state.len(), d);
            states.extend(r.state.iter().map(|&x| x as f32));
            rewards.push(r.reward as f32);
            dones.push(r.done);
        }
        BatchStepResult { states, rewards, dones, n: indices.len(), d }
    }

    pub(crate) fn collect(&mut self, mut f: impl FnMut(&mut RlEnv, usize) -> StepResult) -> BatchStepResult {
        let n = self.envs.len();
        let d = self.n_state_dims;
        let mut states = Vec::with_capacity(n * d);
        let mut rewards = Vec::with_capacity(n);
        let mut dones = Vec::with_capacity(n);
        for i in 0..n {
            let r = f(&mut self.envs[i], i);
            debug_assert_eq!(r.state.len(), d, "instance {i} produced a {}-dim state, expected {d}", r.state.len());
            states.extend(r.state.iter().map(|&x| x as f32));
            rewards.push(r.reward as f32);
            dones.push(r.done);
        }
        BatchStepResult { states, rewards, dones, n, d }
    }
}
