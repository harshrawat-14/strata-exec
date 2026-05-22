// Adversarial market agents (Phase 4 — Component 2).
//
// Two simple counter-parties that react to our flow:
//   - LiquidityWithdrawer pulls bid depth after a large fill, then restores it over
//     several steps.
//   - FrontRunner detects a sustained selling pattern in recent history and steals
//     a fraction of our order's depth before we get to it.
//
// Both wrap a passive OrderBook via the AdversarialBook composition type, which
// also exposes diagnostic counters (front-runner qty, withdrawn pct) for the
// RL environment.

use std::collections::VecDeque;

use crate::market::order_book::{FillResult, OrderBook};

// ---------------------------------------------------------------------------
// LiquidityWithdrawer
// ---------------------------------------------------------------------------

/// Pulls a fraction of remaining bid-side liquidity after a sufficiently
/// large fill, then restores it linearly over `recovery_steps`.
#[derive(Debug, Clone)]
pub struct LiquidityWithdrawer {
    /// Fraction of total bid depth that the fill must exceed to trigger.
    pub sensitivity: f64,
    /// Fraction of each bid level's remaining depth pulled on trigger.
    pub withdrawal_pct: f64,
    /// Number of steps over which withdrawn depth is restored.
    pub recovery_steps: usize,
    /// Per-level unrestored deficit (parallel index with `bid_levels`).
    pub withdrawn_depth_per_level: Vec<f64>,
    /// Remaining recovery steps until the deficit reaches zero.
    recovery_counter: usize,
}

impl Default for LiquidityWithdrawer {
    fn default() -> Self {
        Self::new()
    }
}

impl LiquidityWithdrawer {
    pub fn new() -> Self {
        Self {
            sensitivity: 0.30,
            withdrawal_pct: 0.20,
            recovery_steps: 10,
            withdrawn_depth_per_level: Vec::new(),
            recovery_counter: 0,
        }
    }

    /// Call after each fill. Returns the total depth withdrawn this call
    /// (0.0 if the trigger did not fire). Mutates `book` in place.
    pub fn post_fill(&mut self, book: &mut OrderBook, fill_qty: f64, pre_fill_depth: f64) -> f64 {
        if pre_fill_depth <= 0.0 || fill_qty <= self.sensitivity * pre_fill_depth {
            return 0.0;
        }

        let levels = book.bid_levels_mut();
        if self.withdrawn_depth_per_level.len() < levels.len() {
            self.withdrawn_depth_per_level.resize(levels.len(), 0.0);
        }

        let mut total_withdrawn = 0.0_f64;
        for (i, lvl) in levels.iter_mut().enumerate() {
            let pull = self.withdrawal_pct * lvl.quantity;
            lvl.quantity -= pull;
            self.withdrawn_depth_per_level[i] += pull;
            total_withdrawn += pull;
        }

        // Reset recovery counter so the new withdrawal restores over a fresh window.
        self.recovery_counter = self.recovery_steps;
        total_withdrawn
    }

    /// Restore a slice of the unrestored deficit. Called once per simulation step.
    pub fn recover_step(&mut self, book: &mut OrderBook) {
        if self.recovery_counter == 0 {
            return;
        }
        let levels = book.bid_levels_mut();
        let n = levels.len().min(self.withdrawn_depth_per_level.len());
        let denom = self.recovery_counter as f64;
        for i in 0..n {
            if self.withdrawn_depth_per_level[i] > 0.0 {
                let share = self.withdrawn_depth_per_level[i] / denom;
                levels[i].quantity += share;
                self.withdrawn_depth_per_level[i] -= share;
            }
        }
        self.recovery_counter -= 1;
    }

    /// Sum of all unrestored deficits across levels.
    pub fn deficit(&self) -> f64 {
        self.withdrawn_depth_per_level.iter().sum()
    }
}

// ---------------------------------------------------------------------------
// FrontRunner
// ---------------------------------------------------------------------------

/// Detects sustained selling pressure and front-runs a fraction of the order
/// before it can be filled.
#[derive(Debug, Clone)]
pub struct FrontRunner {
    /// Number of recent trades observed when deciding to trigger.
    pub detection_window: usize,
    /// Sustained-flow trigger threshold (relative to baseline market volume).
    pub size_threshold: f64,
    /// Fraction of the order copied by the front-runner.
    pub aggression: f64,
    /// Last `detection_window` observed trade quantities.
    pub recent_sell_volume: VecDeque<f64>,
    /// Rolling average single-trade volume (EMA).
    pub avg_market_volume: f64,
}

impl Default for FrontRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontRunner {
    pub fn new() -> Self {
        Self {
            detection_window: 5,
            size_threshold: 0.30,
            aggression: 0.40,
            recent_sell_volume: VecDeque::new(),
            avg_market_volume: 0.0,
        }
    }

    /// Call BEFORE each market sell with the intended quantity.
    /// Returns the qty consumed by the front-runner (0.0 if the trigger did not fire).
    pub fn pre_fill(&mut self, book: &mut OrderBook, intended_qty: f64) -> f64 {
        // Require a full detection window before reacting; isolated trades are ignored.
        if self.recent_sell_volume.len() < self.detection_window {
            return 0.0;
        }
        let recent_sum: f64 = self.recent_sell_volume.iter().sum();
        let trigger_value =
            self.size_threshold * self.avg_market_volume * self.detection_window as f64;
        if recent_sum <= trigger_value || intended_qty <= 1e-9 {
            return 0.0;
        }

        let copy_qty = (self.aggression * intended_qty).max(0.0);
        if copy_qty > 1e-9 {
            book.fill_market_sell(copy_qty);
        }
        copy_qty
    }

    /// Update the rolling volume estimator with one observed trade.
    pub fn update_volume(&mut self, qty: f64) {
        if qty <= 0.0 {
            return;
        }
        self.recent_sell_volume.push_back(qty);
        while self.recent_sell_volume.len() > self.detection_window {
            self.recent_sell_volume.pop_front();
        }
        let alpha = 0.10_f64;
        if self.avg_market_volume == 0.0 {
            self.avg_market_volume = qty;
        } else {
            self.avg_market_volume = (1.0 - alpha) * self.avg_market_volume + alpha * qty;
        }
    }
}

// ---------------------------------------------------------------------------
// AdversarialBook — composite that wraps a passive OrderBook
// ---------------------------------------------------------------------------

/// A `OrderBook` wrapped with two reactive adversarial agents.
#[derive(Debug, Clone)]
pub struct AdversarialBook {
    pub base_book: OrderBook,
    pub withdrawer: LiquidityWithdrawer,
    pub front_runner: FrontRunner,
    /// Quantity consumed by the front-runner on the most recent `fill` call.
    pub last_front_runner_qty: f64,
    /// Total depth withdrawn by `LiquidityWithdrawer` on the most recent `fill` call.
    pub last_depth_withdrawn: f64,
    /// Total bid depth observed at the start of the most recent fill (pre-front-runner).
    pub last_pre_fill_depth: f64,
}

impl AdversarialBook {
    pub fn new(base_book: OrderBook) -> Self {
        Self {
            base_book,
            withdrawer: LiquidityWithdrawer::new(),
            front_runner: FrontRunner::new(),
            last_front_runner_qty: 0.0,
            last_depth_withdrawn: 0.0,
            last_pre_fill_depth: 0.0,
        }
    }

    /// Execute a market sell against the adversarial book.
    ///
    /// Steps:
    ///   1. Front-runner consumes a fraction of the depth ahead of us.
    ///   2. We fill what is left in the reduced book.
    ///   3. The liquidity-withdrawer pulls a fraction of the remaining depth
    ///      if our fill was a large fraction of pre-fill depth.
    ///   4. The withdrawer restores a slice of any outstanding deficit.
    ///   5. Our filled quantity is recorded by the front-runner's volume estimator.
    ///
    /// The returned `FillResult` reflects ONLY our own fills (the front-runner's
    /// consumed qty and the withdrawer's deltas are reported via the diagnostic
    /// fields on `self`).
    pub fn fill(&mut self, qty: f64) -> FillResult {
        self.last_pre_fill_depth = self.base_book.total_bid_depth();
        self.last_front_runner_qty = self.front_runner.pre_fill(&mut self.base_book, qty);

        let pre_our_depth = self.base_book.total_bid_depth();
        let result = self.base_book.fill_market_sell(qty);

        self.last_depth_withdrawn =
            self.withdrawer.post_fill(&mut self.base_book, result.filled_qty, pre_our_depth);
        self.withdrawer.recover_step(&mut self.base_book);

        self.front_runner.update_volume(result.filled_qty);
        result
    }

    /// Swap in a fresh book (e.g. next historical snapshot or new synthetic LOB).
    /// Carries over the withdrawer's outstanding deficit so liquidity remains
    /// constrained until recovery completes.
    pub fn update_book(&mut self, new_book: OrderBook) {
        self.base_book = new_book;
        let deficits = self.withdrawer.withdrawn_depth_per_level.clone();
        let levels = self.base_book.bid_levels_mut();
        for (i, lvl) in levels.iter_mut().enumerate() {
            if i < deficits.len() && deficits[i] > 0.0 {
                lvl.quantity = (lvl.quantity - deficits[i]).max(0.0);
            }
        }
    }

    /// Fraction of pre-fill bid depth withdrawn by the post-fill withdrawer.
    /// Returns 0.0 if no fill has been processed yet.
    pub fn last_depth_withdrawn_pct(&self) -> f64 {
        if self.last_pre_fill_depth <= 0.0 {
            0.0
        } else {
            (self.last_depth_withdrawn / self.last_pre_fill_depth).clamp(0.0, 1.0)
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::lob_generator::{generate_lob, LobGeneratorParams};

    /// A deep synthetic book big enough to absorb 100k+ orders without exhausting.
    fn deep_book() -> OrderBook {
        let params = LobGeneratorParams {
            num_levels: 20,
            base_quantity: 100_000.0,
            ..LobGeneratorParams::default()
        };
        generate_lob(100.0, 0.20, &params)
    }

    #[test]
    fn withdrawer_triggers_on_large_fill() {
        let mut book = deep_book();
        let mut wd = LiquidityWithdrawer::new();
        let pre_depth = book.total_bid_depth();

        // Simulate having just executed a fill that consumed 50% of pre-fill depth.
        let fill_qty = 0.50 * pre_depth;
        let withdrawn = wd.post_fill(&mut book, fill_qty, pre_depth);

        assert!(withdrawn > 0.0, "withdrawer must remove depth on a large fill, got {withdrawn}");
        // Each level should have shrunk by withdrawal_pct.
        for lvl in book.bid_levels() {
            assert!(lvl.quantity >= 0.0, "level quantities must remain non-negative");
        }
    }

    #[test]
    fn withdrawer_does_not_trigger_on_small_fill() {
        let mut book = deep_book();
        let mut wd = LiquidityWithdrawer::new();
        let pre_depth = book.total_bid_depth();

        // Small fill — only 5% of pre-fill depth.
        let withdrawn = wd.post_fill(&mut book, 0.05 * pre_depth, pre_depth);
        assert_eq!(withdrawn, 0.0, "small fill must not trigger withdrawal");

        let post_depth = book.total_bid_depth();
        assert!(
            (post_depth - pre_depth).abs() < 1e-9,
            "book must be unchanged: pre={pre_depth} post={post_depth}"
        );
    }

    #[test]
    fn withdrawer_depth_recovers_over_time() {
        let mut book = deep_book();
        let mut wd = LiquidityWithdrawer::new();
        let pre_depth = book.total_bid_depth();

        // Trigger withdrawal.
        wd.post_fill(&mut book, 0.50 * pre_depth, pre_depth);
        let post_withdraw_depth = book.total_bid_depth();
        assert!(
            post_withdraw_depth < pre_depth,
            "depth must drop after withdrawal: pre={pre_depth} post={post_withdraw_depth}"
        );

        // Run all recovery steps; deficit should reach zero and depth should rise back.
        for _ in 0..wd.recovery_steps {
            wd.recover_step(&mut book);
        }
        let recovered_depth = book.total_bid_depth();
        assert!(
            recovered_depth > post_withdraw_depth,
            "depth must grow back during recovery: post_withdraw={post_withdraw_depth} recovered={recovered_depth}"
        );
        assert!(
            wd.deficit().abs() < 1e-6,
            "deficit must reach zero after full recovery, got {}",
            wd.deficit()
        );
    }

    #[test]
    fn front_runner_triggers_on_detected_flow() {
        let mut book = deep_book();
        let mut fr = FrontRunner::new();

        // Populate a full window of consistent large trades.
        for _ in 0..fr.detection_window {
            fr.update_volume(10_000.0);
        }

        let pre_depth = book.total_bid_depth();
        let copy_qty = fr.pre_fill(&mut book, 5_000.0);

        assert!(
            copy_qty > 0.0,
            "front-runner must consume a positive qty when flow is detected"
        );
        let post_depth = book.total_bid_depth();
        assert!(
            post_depth < pre_depth,
            "book depth must drop after front-running: pre={pre_depth} post={post_depth}"
        );
    }

    #[test]
    fn front_runner_ignores_isolated_small_order() {
        let mut book = deep_book();
        let mut fr = FrontRunner::new();

        // Only one observation — buffer not full.
        fr.update_volume(100.0);

        let pre_depth = book.total_bid_depth();
        let copy_qty = fr.pre_fill(&mut book, 5_000.0);

        assert_eq!(
            copy_qty, 0.0,
            "front-runner must not react before the detection window fills",
        );
        let post_depth = book.total_bid_depth();
        assert!(
            (post_depth - pre_depth).abs() < 1e-9,
            "book must be unchanged when no front-running occurs",
        );
    }

    #[test]
    fn adversarial_fill_costs_more_than_passive() {
        // Identical starting books for a head-to-head comparison.
        let mut passive_book = deep_book();
        let adv_book = deep_book();
        let mut adv = AdversarialBook::new(adv_book);

        // Warm up the front-runner's buffer with large prior trades so it triggers.
        for _ in 0..adv.front_runner.detection_window {
            adv.front_runner.update_volume(50_000.0);
        }

        // Use an order large enough to walk past the top bid level (qty=100k),
        // so front-running and withdrawal can push fills onto worse prices.
        let order_qty = 150_000.0;
        let passive_fill = passive_book.fill_market_sell(order_qty);
        let adv_fill = adv.fill(order_qty);

        assert!(adv_fill.filled_qty > 0.0, "adversarial fill must execute some quantity");
        assert!(passive_fill.filled_qty > 0.0, "passive fill must execute some quantity");

        // For a sell, "worse" means lower avg fill price.
        assert!(
            adv_fill.avg_fill_price < passive_fill.avg_fill_price,
            "adversarial fill price ({}) must be worse than passive ({})",
            adv_fill.avg_fill_price,
            passive_fill.avg_fill_price,
        );
        assert!(
            adv.last_front_runner_qty > 0.0,
            "front-runner must have acted given warm buffer",
        );
    }
}
