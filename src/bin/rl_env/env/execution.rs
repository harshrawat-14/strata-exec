//! One environment step: fills, reward, forced liquidation, episode summary.

use serde_json::{json, Value};

use strata_exec::market::order_book::{FillResult, OrderBook};

use crate::config::*;
use crate::env::*;

/// Outcome of one voluntary sell against a book.
pub(crate) struct Fill {
    qty: f64,
    price: f64,
    /// exec − mid; negative is adverse for a sell.
    slippage: f64,
}

impl Fill {
    pub(crate) fn none(book_mid: f64) -> Self {
        Self {
            qty: 0.0,
            price: book_mid,
            slippage: 0.0,
        }
    }

    pub(crate) fn from_result(fr: &FillResult, book_mid: f64) -> Self {
        if fr.filled_qty > 1e-12 {
            Self {
                qty: fr.filled_qty,
                price: fr.avg_fill_price,
                slippage: fr.avg_fill_price - book_mid,
            }
        } else {
            Self::none(book_mid)
        }
    }
}

/// End-of-episode dump of the remaining inventory.
///
/// The dump walks the book with the same `fill_market_sell` used for
/// voluntary trades, so a last-minute liquidation is priced by genuinely
/// consumed depth instead of a flat, size-independent penalty. Invariant:
/// never-trading must not be reward-superior to competent execution -- the
/// old flat penalty was cheap enough that the agent learned to always wait.
pub(crate) struct ForcedLiquidation {
    qty: f64,
    /// Quantity absorbed by real (impact-adjusted) depth.
    filled_qty: f64,
    /// Quantity the book could not absorb.
    residual_qty: f64,
    filled_notional: f64,
    /// Notional of `residual_qty`, priced punitively below the worst level hit.
    punitive_notional: f64,
}

impl ForcedLiquidation {
    /// Price a dump of `qty`. Without a book, the whole size fills at `mid`.
    pub(crate) fn price(book: Option<&mut OrderBook>, qty: f64, mid: f64, spread_ref: f64) -> Self {
        let Some(book) = book else {
            return Self {
                qty,
                filled_qty: qty,
                residual_qty: 0.0,
                filled_notional: qty * mid,
                punitive_notional: 0.0,
            };
        };
        let fr = book.fill_market_sell(qty);
        // Depth exhausted before absorbing the full size: price the residual
        // worse than the worst level actually consumed, never at mid.
        let punitive_notional = if fr.residual_qty > 1e-9 {
            let worst_price = if fr.filled_qty > 1e-12 {
                fr.avg_fill_price
            } else {
                mid
            };
            let punitive_price = (worst_price - spread_ref.max(1e-6) * 5.0).max(1e-6);
            fr.residual_qty * punitive_price
        } else {
            0.0
        };
        Self {
            qty,
            filled_qty: fr.filled_qty,
            residual_qty: fr.residual_qty,
            filled_notional: fr.filled_qty * fr.avg_fill_price,
            punitive_notional,
        }
    }

    pub(crate) fn notional(&self) -> f64 {
        self.filled_notional + self.punitive_notional
    }
}

impl RlEnv {
    /// Execute one action and advance the environment by one step.
    pub fn step(&mut self, action: usize) -> StepResult {
        if matches!(self.cfg.mode, RlMode::Counterfactual) {
            return self.step_counterfactual(action);
        }
        let action = action.min(ACTION_FRACTIONS.len() - 1);

        let book_mid = self
            .current_book
            .as_ref()
            .and_then(|b| b.mid_price())
            .unwrap_or(self.current_mid);

        // Quantity to trade this step.
        let intended = ACTION_FRACTIONS[action] * self.remaining;

        let mut fill = Fill::none(book_mid);
        let mut step_fr_qty = 0.0_f64;
        let mut step_withdrawn = 0.0_f64;
        let mut step_pre_fill_depth = 0.0_f64;

        if intended > 1e-9 {
            if let Some(book) = self.current_book.as_mut() {
                // Adversarial path: front-runner steals depth, then we fill, then withdrawer pulls liquidity.
                if let Some(adv) = self.adversarial.as_mut() {
                    step_pre_fill_depth = book.total_bid_depth();
                    step_fr_qty = adv.front_runner.pre_fill(book, intended);
                    let pre_our_depth = book.total_bid_depth();
                    let fr_result = book.fill_market_sell(intended);
                    step_withdrawn =
                        adv.withdrawer
                            .post_fill(book, fr_result.filled_qty, pre_our_depth);
                    adv.withdrawer.recover_step(book);
                    adv.front_runner.update_volume(fr_result.filled_qty);

                    adv.last_front_runner_qty = step_fr_qty;
                    adv.last_depth_withdrawn = step_withdrawn;
                    adv.last_pre_fill_depth = step_pre_fill_depth;

                    fill = Fill::from_result(&fr_result, book_mid);
                } else {
                    fill = Fill::from_result(&book.fill_market_sell(intended), book_mid);
                }
            }
        }

        self.record_fill(&fill);
        // Scored before the market advances: the variance penalty uses this
        // step's volatility estimate.
        let (reward, is_contribution) = self.step_reward(&fill, book_mid);

        // Advance time.
        self.step_idx += 1;
        self.advance_market();

        // Termination check.
        let inv_done = self.remaining <= 0.001 * self.cfg.total_inventory;
        let time_done = self.step_idx >= self.cfg.total_steps;
        let mut done = inv_done || time_done;
        let mut forced: Option<ForcedLiquidation> = None;
        let mut total_reward_adj = 0.0;

        if time_done && self.remaining > 0.001 * self.cfg.total_inventory {
            let mid = self
                .current_book
                .as_ref()
                .and_then(|b| b.mid_price())
                .unwrap_or(self.arrival_price);
            let liq = ForcedLiquidation::price(
                self.current_book.as_mut(),
                self.remaining,
                mid,
                self.spread_ref,
            );
            total_reward_adj -= self.settle_forced_liquidation(&liq, mid);
            forced = Some(liq);
            done = true;
        }

        let mut info = json!({
            "step": self.step_idx,
            "remaining": self.remaining,
            "fill_qty": fill.qty,
            "fill_price": fill.price,
            "book_mid": book_mid,
            "interval_volume": self.interval_volume_at_step(),
            "is_contribution": is_contribution,
            "regime": self.regime_label(),
            // Exposed so Python-side policies (AdaptiveACPolicy) and diagnostics
            // can read the volatility estimate DIRECTLY instead of inverting it
            // out of state dim s2 = tanh(current_sigma / sigma_ref). That
            // inversion is lossy at the ratios real crash dates reach (tanh
            // saturates: ratio 2.8 -> 0.9926, ratio 4.1 -> 0.99945, so the
            // difference between a moderate and a severe crash lives in the
            // 4th decimal) and it was additionally being done with a wrong
            // formula. Emitting both values makes the state-vector encoding
            // verifiable against ground truth rather than assumed.
            "current_sigma": self.current_sigma,
            "sigma_ref": self.cfg.sigma_ref,
        });

        if self.adversarial.is_some() {
            let withdrawn_pct = if step_pre_fill_depth > 0.0 {
                (step_withdrawn / step_pre_fill_depth).clamp(0.0, 1.0)
            } else {
                0.0
            };
            if let Some(obj) = info.as_object_mut() {
                obj.insert("front_runner_qty".to_string(), json!(step_fr_qty));
                obj.insert("depth_withdrawn_pct".to_string(), json!(withdrawn_pct));
            }
        }

        if done {
            self.insert_episode_summary(&mut info, forced.as_ref());
        }

        let state = self.build_state();
        StepResult {
            state,
            reward: reward + total_reward_adj,
            done,
            info,
        }
    }

    /// Execute one step in counterfactual mode.
    ///
    /// Adversarial agents are not applied here: fills walk the impact-adjusted
    /// book directly, so `--adversarial` has no effect in this mode.
    pub(crate) fn step_counterfactual(&mut self, action: usize) -> StepResult {
        let action = action.min(ACTION_FRACTIONS.len() - 1);

        let intended = ACTION_FRACTIONS[action] * self.remaining;

        // Get the adjusted book (real Binance − accumulated impact).
        let mut adjusted_book = self
            .cf_lob
            .as_ref()
            .map(|cf| cf.current_adjusted_book())
            .unwrap_or_else(|| {
                self.current_book
                    .clone()
                    .unwrap_or(OrderBook::new(vec![], vec![]))
            });

        let book_mid = adjusted_book.mid_price().unwrap_or(self.current_mid);
        let real_depth = self
            .cf_lob
            .as_ref()
            .map(|cf| cf.real_total_bid_depth())
            .unwrap_or(0.0);

        let fill = if intended > 1e-9 {
            Fill::from_result(&adjusted_book.fill_market_sell(intended), book_mid)
        } else {
            Fill::none(book_mid)
        };

        // Record impact and advance cf_lob to next snapshot.
        if let Some(cf) = self.cf_lob.as_mut() {
            cf.record_trade_and_advance(intended.max(0.0));
        }

        self.record_fill(&fill);

        // Sync current_book/mid from the updated cf_lob.
        if let Some(cf) = &self.cf_lob {
            let new_book = cf.current_adjusted_book();
            let new_mid = new_book.mid_price().unwrap_or(self.current_mid);
            let prev_mid = self.current_mid.max(1e-12);
            let log_ret = (new_mid / prev_mid).ln();
            self.push_return(log_ret);
            self.current_mid = new_mid;
            self.current_sigma = self.realised_vol();
            self.current_book = Some(new_book);
            self.price_history.push_back(new_mid);
            while self.price_history.len() > 16 {
                self.price_history.pop_front();
            }
        }

        self.step_idx += 1;

        // Scored after the book advances: the variance penalty uses the
        // post-trade volatility estimate (differs from historical mode).
        let (reward, is_contribution) = self.step_reward(&fill, book_mid);

        // Termination.
        let inv_done = self.remaining <= 0.001 * self.cfg.total_inventory;
        let time_done = self.step_idx >= self.cfg.total_steps;
        let cf_exhausted = self
            .cf_lob
            .as_ref()
            .map(|cf| cf.is_exhausted())
            .unwrap_or(false);
        let mut done = inv_done || time_done || cf_exhausted;
        let mut forced: Option<ForcedLiquidation> = None;
        let mut total_reward_adj = 0.0;

        if (time_done || cf_exhausted) && self.remaining > 0.001 * self.cfg.total_inventory {
            let mid = self.current_mid.max(1e-12);
            // Once the replay has run past its last snapshot the adjusted book
            // is empty, so a dump against it fills nothing and the whole
            // remainder would be priced by the punitive fallback. Dump into the
            // last real book instead: this step's, after our own fill.
            let mut book = if cf_exhausted {
                Some(adjusted_book)
            } else {
                self.cf_lob.as_ref().map(|cf| cf.current_adjusted_book())
            };
            let liq = ForcedLiquidation::price(book.as_mut(), self.remaining, mid, self.spread_ref);
            total_reward_adj -= self.settle_forced_liquidation(&liq, mid);
            forced = Some(liq);
            done = true;
        }

        let depth_reduction_pct = self
            .cf_lob
            .as_ref()
            .map(|cf| cf.total_residual_impact() * 100.0)
            .unwrap_or(0.0);
        let permanent_impact_pct = self
            .cf_lob
            .as_ref()
            .map(|cf| cf.permanent_impact() * 100.0)
            .unwrap_or(0.0);
        let adjusted_depth = self
            .cf_lob
            .as_ref()
            .map(|cf| cf.current_adjusted_book().total_bid_depth())
            .unwrap_or(0.0);

        let mut info = json!({
            "step": self.step_idx,
            "remaining": self.remaining,
            "fill_qty": fill.qty,
            "fill_price": fill.price,
            "book_mid": book_mid,
            "interval_volume": self.interval_volume_at_step(),
            "btc_scale": self.cfg.btc_target / self.cfg.total_inventory.max(1.0),
            "is_contribution": is_contribution,
            "regime": self.regime_label(),
            // See the matching comment in step().
            "current_sigma": self.current_sigma,
            "sigma_ref": self.cfg.sigma_ref,
            "depth_reduction_pct": depth_reduction_pct,
            "permanent_impact_pct": permanent_impact_pct,
            "adjusted_book_depth": adjusted_depth,
            "real_book_depth": real_depth,
        });

        if done {
            self.insert_episode_summary(&mut info, forced.as_ref());
        }

        let state = self.build_state();
        StepResult {
            state,
            reward: reward + total_reward_adj,
            done,
            info,
        }
    }

    /// Add end-of-episode fields to `info`: total implementation shortfall
    /// and, if the episode ended in a forced liquidation, how it was priced
    /// (real depth vs the punitive residual).
    pub(crate) fn insert_episode_summary(
        &self,
        info: &mut Value,
        forced: Option<&ForcedLiquidation>,
    ) {
        let ideal = self.arrival_price * self.cfg.total_inventory;
        let is_dollar = ideal - self.total_notional_received;
        let is_pct = if ideal.abs() > 1e-12 {
            is_dollar / ideal * 100.0
        } else {
            0.0
        };
        let Some(obj) = info.as_object_mut() else {
            return;
        };
        obj.insert("total_IS_pct".to_string(), json!(is_pct));
        obj.insert("total_is_dollar".to_string(), json!(is_dollar));
        if let Some(liq) = forced {
            obj.insert("forced_liquidation_qty".to_string(), json!(liq.qty));
            obj.insert("forced_liq_filled_qty".to_string(), json!(liq.filled_qty));
            obj.insert(
                "forced_liq_residual_qty".to_string(),
                json!(liq.residual_qty),
            );
            obj.insert(
                "forced_liq_filled_notional".to_string(),
                json!(liq.filled_notional),
            );
            obj.insert(
                "forced_liq_punitive_notional".to_string(),
                json!(liq.punitive_notional),
            );
        }
    }

    /// Apply a voluntary fill to inventory and episode totals.
    pub(crate) fn record_fill(&mut self, fill: &Fill) {
        self.remaining = (self.remaining - fill.qty).max(0.0);
        self.total_filled += fill.qty;
        self.total_notional_received += fill.qty * fill.price;
        self.last_slippage = fill.slippage;
    }

    /// Dimensionless IS-normalised step reward and the step's IS contribution.
    ///
    /// Fill step: signed slippage × fill fraction (negative when sell < mid).
    /// Wait step: small penalty for remaining exposure (keeps wait rewards tiny).
    /// Must be called after `record_fill`, since it reads the updated inventory.
    pub(crate) fn step_reward(&self, fill: &Fill, book_mid: f64) -> (f64, f64) {
        let total = self.cfg.total_inventory.max(1.0);
        let q_frac = self.remaining / total;
        let sigma_ref = self.cfg.sigma_ref.max(1e-9);

        let fill_cost = if fill.qty > 0.0 && book_mid > 1e-12 {
            (book_mid - fill.price) / book_mid // positive = adverse for a seller
        } else {
            0.0
        };
        let perm_cost = self.cfg.gamma * (fill.qty / total);

        let var_penalty = 0.01 * (self.current_sigma / sigma_ref).powi(2) * q_frac.powi(2);
        let inv_pressure = 0.0002 * q_frac.powi(2);

        let reward = if fill.qty > 1e-9 {
            (fill.price - book_mid) * fill.qty / (self.arrival_price.max(1e-12) * total)
        } else {
            -(var_penalty + inv_pressure) * 0.0001
        };
        (reward, -(fill_cost + perm_cost))
    }

    /// Book a forced liquidation into the episode totals, zero the inventory,
    /// and return its reward cost (>= 0): the same slippage formula as a normal
    /// fill plus the permanent-impact term on the full dumped quantity.
    pub(crate) fn settle_forced_liquidation(&mut self, liq: &ForcedLiquidation, mid: f64) -> f64 {
        let total = self.cfg.total_inventory.max(1.0);
        let notional = liq.notional();
        let liquidated_at = if liq.qty > 1e-12 {
            notional / liq.qty
        } else {
            mid
        };

        self.total_filled += liq.qty;
        self.total_notional_received += notional;
        self.remaining = 0.0;

        let slippage_cost =
            (mid - liquidated_at) * liq.qty / (self.arrival_price.max(1e-12) * total);
        let forced_perm = self.cfg.gamma * (liq.qty / total);
        slippage_cost + forced_perm
    }
}
