// ── Phase 2 Track B Calibration Gap (Angle 1 Finding) ───────────────────────
//
// Table A (formula-based fill): exec = effective_price × (1 − impact_fraction)
// Table B (LOB-based fill):     exec = depth-consuming avg fill price
// Measured over 50 GARCH paths, gamma=0.005, impact_coeff=0.5
//
// Strategy         | Formula IS% | LOB IS%  | Delta%
// ─────────────────────────────────────────────────────
// TWAP             |   2.2084%   | 2.2012%  | −0.0072%
// Heuristic        |   2.0390%   | 2.0319%  | −0.0071%
// Optimal          |   1.1326%   | 1.1270%  | −0.0056%
// AdaptiveOptimal  |   1.5340%   | 1.5281%  | −0.0060%
//
// Finding: Under synthetic LOB conditions, the formula-based impact model
// slightly OVERESTIMATES slippage (~0.006–0.007% of notional) compared to
// the depth-consuming LOB fill. The delta is small because:
//   (a) the transient impact fraction still shifts the LOB mid, overlapping
//       with the LOB depth-slippage effect;
//   (b) the synthetic LOB has ample depth (base_qty=1000 × 10 levels per side),
//       so depth-consuming fills rarely spill into worse levels.
//
// This gap is the Phase 2 Track B baseline. When real Binance L2 data replaces
// the synthetic LOB, the delta is expected to grow substantially — real LOBs
// have sparse deep levels and clustered thin depth during high-vol periods,
// producing larger depth-consumption slippage than the formula predicts.
//
// TODO Phase 2 Track B WebSocket — re-measure this delta with real L2 snapshots.

use crate::market::order_book::{OrderBook, PriceLevel};

/// Parameters controlling the synthetic LOB shape.
pub struct LobGeneratorParams {
    /// Number of price levels on each side of the book. Default: 10.
    pub num_levels: usize,
    /// Fractional price increment between levels (e.g. 0.0001 = 0.01%). Default: 0.0001.
    pub level_spacing: f64,
    /// Quantity available at the best bid/ask level. Default: 1000.0.
    pub base_quantity: f64,
    /// Quantity multiplier per additional level away from mid. Default: 0.85.
    /// Deeper levels have less liquidity (level k has base_quantity × depth_decay^k).
    pub depth_decay: f64,
    /// Bid-ask spread in basis points (1 bps = 0.01%). Default: 2.0.
    pub spread_bps: f64,
}

impl Default for LobGeneratorParams {
    fn default() -> Self {
        Self {
            num_levels: 10,
            level_spacing: 0.0001,
            base_quantity: 1000.0,
            depth_decay: 0.85,
            spread_bps: 2.0,
        }
    }
}

/// Generate a synthetic limit order book centred on `mid_price`.
///
/// The spread widens with volatility:
///   effective_spread = spread_bps × (1 + vol_scalar) × mid_price / 10_000
/// where `vol_scalar = volatility / 0.20` (normalised to 20% annual vol).
///
/// Level quantities decay geometrically away from the best level:
///   qty_k = base_quantity × depth_decay^k   (k = 0 is best level)
///
/// Ask levels are sorted ascending; bid levels sorted descending.
///
/// # Marker
/// TODO Phase 2 Track B WebSocket — replace with real L2 snapshot
pub fn generate_lob(mid_price: f64, volatility: f64, params: &LobGeneratorParams) -> OrderBook {
    // Effective spread scales with volatility relative to a 20%-vol baseline.
    let vol_scalar = volatility / 0.20;
    let effective_spread_bps = params.spread_bps * (1.0 + vol_scalar);
    let half_spread = mid_price * effective_spread_bps / 20_000.0; // bps → fraction, then half

    let level_step = mid_price * params.level_spacing;

    let mut asks = Vec::with_capacity(params.num_levels);
    let mut bids = Vec::with_capacity(params.num_levels);

    for k in 0..params.num_levels {
        let qty = params.base_quantity * params.depth_decay.powi(k as i32);
        let offset = k as f64 * level_step;

        asks.push(PriceLevel {
            price: mid_price + half_spread + offset,
            quantity: qty,
        });

        bids.push(PriceLevel {
            price: mid_price - half_spread - offset,
            quantity: qty,
        });
    }

    // asks: already ascending (offset grows with k)
    // bids: already descending (price shrinks as offset grows with k)
    OrderBook::new(asks, bids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> LobGeneratorParams {
        LobGeneratorParams::default()
    }

    #[test]
    fn generated_book_has_correct_number_of_levels() {
        let params = LobGeneratorParams { num_levels: 5, ..default_params() };
        let book = generate_lob(100.0, 0.20, &params);
        // Verify via depth queries: ask at price of 5th level should include all 5 levels.
        // We can check indirectly via ask_depth_up_to a very high price capturing all.
        let total_ask = book.ask_depth_up_to(f64::MAX);
        let total_bid = book.bid_depth_up_to(0.0);

        // 5 levels each side; qty_k = 1000 × 0.85^k
        let expected_total: f64 = (0..5).map(|k| 1000.0 * 0.85_f64.powi(k)).sum();
        assert!(
            (total_ask - expected_total).abs() < 1e-6,
            "ask total={total_ask}, expected={expected_total}"
        );
        assert!(
            (total_bid - expected_total).abs() < 1e-6,
            "bid total={total_bid}, expected={expected_total}"
        );
    }

    #[test]
    fn generated_spread_scales_with_volatility() {
        let params = default_params();
        let book_low = generate_lob(100.0, 0.05, &params);
        let book_high = generate_lob(100.0, 0.40, &params);

        let spread_low = book_low.spread().expect("must have spread (low vol)");
        let spread_high = book_high.spread().expect("must have spread (high vol)");
        assert!(
            spread_high > spread_low,
            "spread should widen with volatility: low={spread_low:.6}, high={spread_high:.6}"
        );
    }

    #[test]
    fn generated_ask_prices_are_ascending() {
        let params = default_params();
        let book = generate_lob(100.0, 0.20, &params);

        // We can verify by doing successive depth queries and confirming each level adds qty.
        // More directly: check that ask_depth_up_to grows monotonically with price.
        let best_ask = book.best_ask().expect("must have best ask");
        let step = 100.0 * params.level_spacing;

        let mut prev_depth = 0.0_f64;
        for k in 0..params.num_levels {
            let price_limit = best_ask + k as f64 * step;
            let depth = book.ask_depth_up_to(price_limit);
            assert!(
                depth >= prev_depth,
                "ask depth must be non-decreasing: k={k}, prev={prev_depth}, now={depth}"
            );
            prev_depth = depth;
        }
    }

    #[test]
    fn generated_bid_prices_are_descending() {
        let params = default_params();
        let book = generate_lob(100.0, 0.20, &params);

        // bid_depth_up_to(price) counts bids >= price; grows as price decreases.
        // Equivalently: best bid > next bid > ... (descending by construction).
        let best_bid = book.best_bid().expect("must have best bid");
        let step = 100.0 * params.level_spacing;

        let mut prev_depth = 0.0_f64;
        for k in 0..params.num_levels {
            let price_limit = best_bid - k as f64 * step;
            let depth = book.bid_depth_up_to(price_limit);
            assert!(
                depth >= prev_depth,
                "bid depth_up_to must grow as price limit falls: k={k}, prev={prev_depth}, now={depth}"
            );
            prev_depth = depth;
        }
    }

    #[test]
    fn generated_depth_decays_with_distance_from_mid() {
        let params = default_params();
        let book = generate_lob(100.0, 0.20, &params);

        // The best-level quantity should exceed all subsequent levels.
        // Indirectly: depth at first level < depth at all levels combined.
        let best_ask = book.best_ask().expect("must have best ask");
        let step = 100.0 * params.level_spacing;

        let depth_first_level = book.ask_depth_up_to(best_ask);
        let depth_all_levels = book.ask_depth_up_to(best_ask + (params.num_levels as f64) * step);

        assert!(
            depth_all_levels > depth_first_level,
            "total depth must exceed single-level depth: first={depth_first_level}, all={depth_all_levels}"
        );

        // Confirm decay: qty at level 0 > qty at level 1 (base_qty × decay^0 > base_qty × decay^1).
        let qty_level_0 = depth_first_level;
        let qty_level_1 = book.ask_depth_up_to(best_ask + step) - qty_level_0;
        assert!(
            qty_level_0 > qty_level_1,
            "quantity must decay away from best: level0={qty_level_0:.4}, level1={qty_level_1:.4}"
        );
    }
}
