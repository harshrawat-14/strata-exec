//! Per-process cache of parsed per-date market data.

use std::collections::HashMap;
use std::sync::Arc;


use strata_exec::market::agg_trades::AggTradesDay;
use strata_exec::market::lob_replay::{LobReplay, LobSnapshot};

/// Caches parsed per-date market data (`LobReplay` snapshots, `AggTradesDay`) across
/// repeated `reset()` calls within one `RlEnv`'s process lifetime.
///
/// Phase S baseline measurement: `LobReplay::from_csv` + `AggTradesDay::from_csv`
/// account for ~99% of a historical/counterfactual `reset()`'s allocations and ~100%
/// of its wall time (re-parsing a 1M+-row aggTrades CSV and a 28.8K-row bookDepth CSV
/// from disk), yet the same date is replayed across many episodes both in evaluation
/// (50 episodes/strategy/date) and training (repeated sampling from `--multi-date`).
/// Parsing once per date per process and sharing the immutable parsed data via `Arc`
/// (a cheap refcount bump per reset, not a copy) removes that redundant I/O/parsing
/// entirely after the first reset for a given date. Deliberately process-local, not
/// persisted across `rl-env` invocations: an updated CSV on disk is always picked up
/// by the next process start.
#[derive(Default)]
pub struct MarketDataCache {
    lob: HashMap<String, Arc<Vec<LobSnapshot>>>,
    agg: HashMap<String, Arc<AggTradesDay>>,
}

impl MarketDataCache {
    pub(crate) fn load_lob(&mut self, path: &str) -> Result<LobReplay, String> {
        if let Some(snapshots) = self.lob.get(path) {
            return Ok(LobReplay::from_arc(Arc::clone(snapshots)));
        }
        let replay = LobReplay::from_csv(path)
            .map_err(|e| format!("failed to load LOB data '{path}': {e}"))?;
        let snapshots = replay.snapshots_arc();
        self.lob.insert(path.to_string(), Arc::clone(&snapshots));
        Ok(LobReplay::from_arc(snapshots))
    }

    pub(crate) fn load_agg(&mut self, path: &str) -> Result<Arc<AggTradesDay>, String> {
        if let Some(agg) = self.agg.get(path) {
            return Ok(Arc::clone(agg));
        }
        let agg = Arc::new(AggTradesDay::from_csv(path)?);
        self.agg.insert(path.to_string(), Arc::clone(&agg));
        Ok(agg)
    }
}
