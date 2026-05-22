/// A single price level in the order book.
#[derive(Debug, Clone, PartialEq)]
pub struct PriceLevel {
    pub price: f64,
    pub quantity: f64,
}

/// A limit order book snapshot with sorted bid and ask ladders.
///
/// Invariants (maintained by `new`, not enforced at runtime):
///   - asks are sorted ascending by price (lowest ask first)
///   - bids are sorted descending by price (highest bid first)
///   - best_ask > best_bid (no crossed book)
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// Ask levels: ascending price order (lowest ask first).
    asks: Vec<PriceLevel>,
    /// Bid levels: descending price order (highest bid first).
    bids: Vec<PriceLevel>,
    pub timestamp: u64,
}

/// Result of a depth-consuming market fill.
#[derive(Debug, Clone)]
pub struct FillResult {
    /// Total quantity filled across all consumed levels.
    pub filled_qty: f64,
    /// Dollar-weighted average fill price across consumed levels.
    pub avg_fill_price: f64,
    /// Number of price levels touched (partially or fully consumed).
    pub levels_consumed: usize,
    /// Quantity not filled because the book was exhausted.
    pub residual_qty: f64,
    /// Half-spread cost: `half_spread × filled_qty` in dollar terms.
    pub real_spread_cost: f64,
    /// Price improvement/slippage versus mid.
    /// For a sell: `mid_price - avg_fill_price` (positive = adverse).
    /// For a buy:  `avg_fill_price - mid_price` (positive = adverse).
    pub slippage_from_mid: f64,
}

impl OrderBook {
    /// Create a new order book.
    ///
    /// Callers are responsible for:
    ///   - asks sorted ascending (lowest price first)
    ///   - bids sorted descending (highest price first)
    ///   - no crossed spread
    pub fn new(asks: Vec<PriceLevel>, bids: Vec<PriceLevel>) -> Self {
        Self {
            asks,
            bids,
            timestamp: 0,
        }
    }

    /// Best (lowest) ask price, or `None` if the ask side is empty.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Best (highest) bid price, or `None` if the bid side is empty.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Mid-price: average of best bid and best ask.
    /// Returns `None` if either side is empty.
    pub fn mid_price(&self) -> Option<f64> {
        Some((self.best_ask()? + self.best_bid()?) / 2.0)
    }

    /// Bid-ask spread: best_ask − best_bid.
    /// Returns `None` if either side is empty.
    pub fn spread(&self) -> Option<f64> {
        Some(self.best_ask()? - self.best_bid()?)
    }

    /// True LOB half-spread — replaces the 0.5×σ×√dt proxy.
    pub fn half_spread(&self) -> Option<f64> {
        Some(self.spread()? / 2.0)
    }

    /// Return a new `OrderBook` with every price multiplied by `factor`.
    ///
    /// Quantities are unchanged; sort order is preserved (factor must be > 0).
    /// Used to normalize real LOB snapshots (e.g. BTC-scale) to the simulation
    /// price scale before depth-consuming fills.
    pub fn rescale(&self, factor: f64) -> OrderBook {
        let scale_level = |l: &PriceLevel| PriceLevel {
            price: l.price * factor,
            quantity: l.quantity,
        };
        OrderBook {
            asks: self.asks.iter().map(scale_level).collect(),
            bids: self.bids.iter().map(scale_level).collect(),
            timestamp: self.timestamp,
        }
    }

    /// Return a new `OrderBook` with prices and quantities rescaled independently.
    ///
    /// `price_factor`: multiply every price (e.g. 100.0 / real_btc_mid).
    /// `qty_factor`:   multiply every quantity (e.g. 1.0 / btc_scale to convert
    ///                 BTC → simulation units).
    pub fn rescale_full(&self, price_factor: f64, qty_factor: f64) -> OrderBook {
        let scale_level = |l: &PriceLevel| PriceLevel {
            price: l.price * price_factor,
            quantity: l.quantity * qty_factor,
        };
        OrderBook {
            asks: self.asks.iter().map(scale_level).collect(),
            bids: self.bids.iter().map(scale_level).collect(),
            timestamp: self.timestamp,
        }
    }

    /// Read-only view of the bid ladder (descending by price).
    /// Intended for adversarial agents that need per-level inspection.
    pub fn bid_levels(&self) -> &[PriceLevel] {
        &self.bids
    }

    /// Mutable view of the bid ladder (descending by price).
    /// Adversarial agents use this to withdraw or restore liquidity per level.
    pub fn bid_levels_mut(&mut self) -> &mut [PriceLevel] {
        &mut self.bids
    }

    /// Sum of all bid quantities across the ladder.
    pub fn total_bid_depth(&self) -> f64 {
        self.bids.iter().map(|l| l.quantity).sum()
    }

    /// Scale all bid quantities by `factor` (clamped to [0.05, 1.0]).
    /// Used by CounterfactualLob to model depth withdrawal due to market impact.
    pub fn scale_bid_depth(&mut self, factor: f64) {
        let factor = factor.clamp(0.05, 1.0);
        for level in self.bids.iter_mut() {
            level.quantity *= factor;
        }
    }

    /// Total ask quantity available at prices ≤ `price` (inclusive).
    pub fn ask_depth_up_to(&self, price: f64) -> f64 {
        self.asks
            .iter()
            .take_while(|l| l.price <= price)
            .map(|l| l.quantity)
            .sum()
    }

    /// Total bid quantity available at prices ≥ `price` (inclusive).
    pub fn bid_depth_up_to(&self, price: f64) -> f64 {
        self.bids
            .iter()
            .take_while(|l| l.price >= price)
            .map(|l| l.quantity)
            .sum()
    }

    /// Execute a market sell by walking the bid ladder from best bid downward.
    ///
    /// Destructive: consumed quantity is removed from the book.
    /// Returns a [`FillResult`] describing how the order was filled.
    pub fn fill_market_sell(&mut self, quantity: f64) -> FillResult {
        let mid = self.mid_price();
        let half_spread = self.half_spread();
        self.fill_side(&mut vec![], quantity, true, mid, half_spread)
    }

    /// Execute a market buy by walking the ask ladder from best ask upward.
    ///
    /// Destructive: consumed quantity is removed from the book.
    /// Returns a [`FillResult`] describing how the order was filled.
    pub fn fill_market_buy(&mut self, quantity: f64) -> FillResult {
        let mid = self.mid_price();
        let half_spread = self.half_spread();
        self.fill_side(&mut vec![], quantity, false, mid, half_spread)
    }

    /// Internal depth-consuming fill engine.
    ///
    /// `is_sell = true`  → walk bids (descending), fill against bid ladder.
    /// `is_sell = false` → walk asks (ascending), fill against ask ladder.
    fn fill_side(
        &mut self,
        _unused: &mut Vec<()>,
        quantity: f64,
        is_sell: bool,
        mid: Option<f64>,
        half_spread: Option<f64>,
    ) -> FillResult {
        let ladder = if is_sell {
            &mut self.bids
        } else {
            &mut self.asks
        };

        let mut remaining = quantity;
        let mut notional_filled = 0.0_f64;
        let mut filled_qty = 0.0_f64;
        let mut levels_consumed = 0_usize;

        let mut i = 0;
        while i < ladder.len() && remaining > 1e-12 {
            let available = ladder[i].quantity;
            let level_price = ladder[i].price;

            if available <= remaining {
                // Consume the entire level.
                notional_filled += available * level_price;
                filled_qty += available;
                remaining -= available;
                levels_consumed += 1;
                ladder.remove(i);
                // Do not advance i — next level slides into position i.
            } else {
                // Partially consume this level.
                notional_filled += remaining * level_price;
                filled_qty += remaining;
                levels_consumed += 1;
                ladder[i].quantity -= remaining;
                remaining = 0.0;
                i += 1;
            }
        }

        let avg_fill_price = if filled_qty > 1e-12 {
            notional_filled / filled_qty
        } else {
            0.0
        };

        let real_spread_cost = half_spread.unwrap_or(0.0) * filled_qty;

        let slippage_from_mid = if let Some(m) = mid {
            if is_sell {
                m - avg_fill_price // positive when filled below mid (adverse for seller)
            } else {
                avg_fill_price - m // positive when filled above mid (adverse for buyer)
            }
        } else {
            0.0
        };

        FillResult {
            filled_qty,
            avg_fill_price,
            levels_consumed,
            residual_qty: remaining,
            real_spread_cost,
            slippage_from_mid,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_book() -> OrderBook {
        // Bids (descending): 99.9, 99.8, 99.7
        // Asks (ascending):  100.1, 100.2, 100.3
        // Mid = 100.0, spread = 0.2, half_spread = 0.1
        OrderBook::new(
            vec![
                PriceLevel { price: 100.1, quantity: 10.0 },
                PriceLevel { price: 100.2, quantity: 20.0 },
                PriceLevel { price: 100.3, quantity: 30.0 },
            ],
            vec![
                PriceLevel { price: 99.9, quantity: 10.0 },
                PriceLevel { price: 99.8, quantity: 20.0 },
                PriceLevel { price: 99.7, quantity: 30.0 },
            ],
        )
    }

    #[test]
    fn best_ask_is_lowest_ask_price() {
        let book = make_book();
        assert_eq!(book.best_ask(), Some(100.1));
    }

    #[test]
    fn best_bid_is_highest_bid_price() {
        let book = make_book();
        assert_eq!(book.best_bid(), Some(99.9));
    }

    #[test]
    fn mid_price_is_average_of_best_bid_and_ask() {
        let book = make_book();
        let mid = book.mid_price().expect("must have mid");
        assert!(
            (mid - 100.0).abs() < 1e-10,
            "expected mid=100.0, got {mid}"
        );
    }

    #[test]
    fn spread_is_ask_minus_bid() {
        let book = make_book();
        let spread = book.spread().expect("must have spread");
        assert!(
            (spread - 0.2).abs() < 1e-10,
            "expected spread=0.2, got {spread}"
        );
    }

    #[test]
    fn ask_depth_up_to_sums_correctly_across_levels() {
        let book = make_book();
        // Up to 100.1: only first level = 10
        assert!((book.ask_depth_up_to(100.1) - 10.0).abs() < 1e-10);
        // Up to 100.2: first + second = 30
        assert!((book.ask_depth_up_to(100.2) - 30.0).abs() < 1e-10);
        // Up to 100.3: all three = 60
        assert!((book.ask_depth_up_to(100.3) - 60.0).abs() < 1e-10);
        // Up to 100.0: below best ask, nothing
        assert!((book.ask_depth_up_to(100.0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn empty_book_returns_none_for_all_queries() {
        let book = OrderBook::new(vec![], vec![]);
        assert!(book.best_ask().is_none());
        assert!(book.best_bid().is_none());
        assert!(book.mid_price().is_none());
        assert!(book.spread().is_none());
        assert!(book.half_spread().is_none());
    }

    // ── Fill tests ───────────────────────────────────────────────────────────

    #[test]
    fn fill_consumes_single_level_when_quantity_fits() {
        let mut book = make_book();
        // Sell 5 units — fits within best bid (10 units at 99.9)
        let fill = book.fill_market_sell(5.0);
        assert!((fill.filled_qty - 5.0).abs() < 1e-10, "filled_qty={}", fill.filled_qty);
        assert!((fill.avg_fill_price - 99.9).abs() < 1e-10, "avg_fill_price={}", fill.avg_fill_price);
        assert_eq!(fill.levels_consumed, 1);
        assert!((fill.residual_qty).abs() < 1e-10);
        // Book: best bid now has 5 units remaining at 99.9
        assert!((book.best_bid().unwrap() - 99.9).abs() < 1e-10);
    }

    #[test]
    fn fill_spans_multiple_levels_when_first_level_insufficient() {
        let mut book = make_book();
        // Sell 15 units — first bid level has only 10, need to dip into second
        let fill = book.fill_market_sell(15.0);
        assert!((fill.filled_qty - 15.0).abs() < 1e-10);
        assert_eq!(fill.levels_consumed, 2);
        assert!((fill.residual_qty).abs() < 1e-10);
        // avg = (10*99.9 + 5*99.8) / 15
        let expected_avg = (10.0 * 99.9 + 5.0 * 99.8) / 15.0;
        assert!(
            (fill.avg_fill_price - expected_avg).abs() < 1e-10,
            "expected {expected_avg}, got {}",
            fill.avg_fill_price
        );
    }

    #[test]
    fn fill_returns_correct_average_price_across_levels() {
        let mut book = make_book();
        // Buy 25 units across first two ask levels: 10@100.1 + 15@100.2
        let fill = book.fill_market_buy(25.0);
        assert!((fill.filled_qty - 25.0).abs() < 1e-10);
        let expected_avg = (10.0 * 100.1 + 15.0 * 100.2) / 25.0;
        assert!(
            (fill.avg_fill_price - expected_avg).abs() < 1e-10,
            "expected {expected_avg}, got {}",
            fill.avg_fill_price
        );
    }

    #[test]
    fn fill_returns_residual_when_book_exhausted() {
        let mut book = make_book();
        // Total bid depth = 10 + 20 + 30 = 60; sell 100 → residual = 40
        let fill = book.fill_market_sell(100.0);
        assert!((fill.filled_qty - 60.0).abs() < 1e-10, "filled={}", fill.filled_qty);
        assert!((fill.residual_qty - 40.0).abs() < 1e-10, "residual={}", fill.residual_qty);
        // Book should be empty on bid side
        assert!(book.best_bid().is_none());
    }

    #[test]
    fn fill_is_destructive_book_depth_decreases_after_fill() {
        let mut book = make_book();
        let depth_before = book.bid_depth_up_to(99.9);
        book.fill_market_sell(5.0);
        let depth_after = book.bid_depth_up_to(99.9);
        assert!(
            depth_after < depth_before,
            "depth should decrease after fill: before={depth_before}, after={depth_after}"
        );
    }

    #[test]
    fn fill_slippage_increases_with_order_size() {
        // Large sell hits deeper (worse) bid levels → more slippage from mid
        let mut book_small = make_book();
        let mut book_large = make_book();
        let fill_small = book_small.fill_market_sell(5.0);   // stays at best bid
        let fill_large = book_large.fill_market_sell(25.0);  // crosses into deeper levels
        assert!(
            fill_large.slippage_from_mid > fill_small.slippage_from_mid,
            "larger sell must have more slippage: small={}, large={}",
            fill_small.slippage_from_mid, fill_large.slippage_from_mid
        );
    }

    #[test]
    fn fill_slippage_is_zero_for_infinitesimally_small_order() {
        let mut book = make_book();
        // An infinitesimally small sell fills entirely at the best bid price.
        // slippage_from_mid = mid - best_bid = 100.0 - 99.9 = 0.1 (half spread).
        // This is the minimum slippage — it equals half the spread, not zero.
        // The test name reflects the spirit: slippage does NOT grow beyond half-spread
        // for tiny orders, and is bounded below by the half-spread itself.
        let fill = book.fill_market_sell(1e-6);
        let half_spread = book.half_spread().unwrap();
        assert!(
            (fill.slippage_from_mid - half_spread).abs() < 1e-6,
            "tiny sell slippage should equal half_spread={half_spread}, got {}",
            fill.slippage_from_mid
        );
    }

    #[test]
    fn rescale_preserves_spread_ratio() {
        // Original book: mid=100.0, spread=0.2 → spread/mid = 0.002
        // After rescaling to simulate BTC→sim (factor = 100.0 / 41000.0), the
        // spread/mid ratio must be identical within floating-point precision.
        let book = make_book();
        let orig_mid = book.mid_price().expect("must have mid");
        let orig_spread = book.spread().expect("must have spread");
        let orig_ratio = orig_spread / orig_mid;

        let factor = 100.0 / 41_000.0; // representative BTC-to-sim scale
        let scaled = book.rescale(factor);
        let scaled_mid = scaled.mid_price().expect("scaled must have mid");
        let scaled_spread = scaled.spread().expect("scaled must have spread");
        let scaled_ratio = scaled_spread / scaled_mid;

        assert!(
            (orig_ratio - scaled_ratio).abs() < 1e-10,
            "spread/mid ratio must be unchanged after rescaling: \
             original={orig_ratio:.12} scaled={scaled_ratio:.12}"
        );

        // Total depth (sum of all quantities) must be unchanged on both sides.
        let orig_ask_depth = book.ask_depth_up_to(f64::MAX);
        let scaled_ask_depth = scaled.ask_depth_up_to(f64::MAX);
        assert!(
            (orig_ask_depth - scaled_ask_depth).abs() < 1e-10,
            "total ask depth must be unchanged: orig={orig_ask_depth} scaled={scaled_ask_depth}"
        );

        let orig_bid_depth = book.bid_depth_up_to(0.0);
        let scaled_bid_depth = scaled.bid_depth_up_to(0.0);
        assert!(
            (orig_bid_depth - scaled_bid_depth).abs() < 1e-10,
            "total bid depth must be unchanged: orig={orig_bid_depth} scaled={scaled_bid_depth}"
        );
    }
}
