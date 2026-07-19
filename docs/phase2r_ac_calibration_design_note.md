# Design note: mapping the simulator's calibrated impact model onto AC's (σ, η) — item 2, audit only

Scope per explicit instruction: **design note only, no code.** This exists to surface the modeling
choice and its tradeoffs before anyone commits to a specific calibration, not to implement one.

## The two models, precisely

**Simulator (Task 2, `CounterfactualLob::record_trade_and_advance`, `counterfactual_lob.rs:213-217`):**

```
impact_fraction = Y · σ_cal · sqrt(participation)      participation = trade_btc / adv_btc
permanent_component = impact_fraction · permanent_fraction     (never decays, additive, capped 95%)
temporary_component = impact_fraction · temporary_fraction     (decays as exp(-ρ·Δt))
```

`Y` (`impact_coefficient`) and `σ_cal` are fit per date by `CalibrationEngine::calibrate` from real
aggTrades: `Y = |Δprice/price| / (σ_cal · sqrt(Q/V))`, i.e. Y is whatever constant makes the
square-root law match observed historical price-impact-per-trade on that date.

**AC closed form (`ACOptimalPolicy`, `rl/evaluate.py:379-420`):**

```
κ = sqrt(λ · σ² / η)
schedule = sinh(κ·(T-t)) / sinh(κ·T)          -- from Almgren-Chriss (2000/2001)
```

`η` here is a *linear* temporary-impact coefficient: AC's underlying cost model assumes marginal
cost per unit trading rate is constant (`cost_rate = η · v`). There is no `γ` (permanent impact) term
in this implementation at all (confirmed directly — see the close-out gate report). `σ` is the
price-volatility input to the risk-aversion term, structurally the same *concept* as the simulator's
`σ_cal`, just currently hardcoded to 0.20 instead of using it.

## σ: no modeling ambiguity, straightforward to fix

`σ` is a measurement of realized price volatility, not an impact-model parameter — it means the same
thing in both models. `realized_volatility()` (`impact_fit.rs:149`) already computes a properly
annualized value from real LOB snapshot mid-returns, in the same units (`0.20` = 20%/year) as the
hardcoded default. **There is no design choice to make here** — exposing this per-date value to
Python and using it in place of the hardcoded `sigma=0.20`/`sigma_ref=0.20` is a straightforward,
low-risk fix whenever plumbing is approved. This part of item 2 does not need to wait on η.

## η: genuinely ambiguous — the sqrt-vs-linear mismatch has no single principled fix

`Y·σ_cal` calibrates the *magnitude* of a **square-root** law. AC's `η` is the slope of a **linear**
law. These are different functional forms, not just different units of the same quantity — there is
no value of `η` that makes AC's linear model equal the sqrt model except at one specific participation
rate. Three candidate ways to pick that rate, each a different, defensible-but-incomplete modeling
commitment:

**Candidate A — local linearization at the schedule's own typical participation rate.**
Take the derivative of the sqrt law at some reference participation `p*` (e.g., TWAP's uniform
`1/total_steps` per step, or the eventual AC schedule's own average participation) and use that slope
as `η`. Problem: the sqrt law's marginal cost *decreases* as participation grows (economies of scale
in the sqrt functional form), so the "right" `η` depends on what `p*` you evaluate at — pick a small
`p*` and `η` is large (AC becomes overly cautious, spreads out more than the sqrt model actually
requires); pick a large `p*` and `η` is small (AC becomes overly aggressive). Using the eventual
schedule's own participation is circular (the schedule depends on `η`, `η` depends on the schedule) —
solvable via fixed-point iteration, but that's implementation complexity for what's still an
approximation, not a principled answer.

**Candidate B — match total cost at a reference execution (TWAP).**
Run TWAP through the real simulator, measure its actual total temporary-impact cost, then solve for
the `η` that makes AC's closed-form cost formula predict that same total cost under a TWAP-like
schedule. This anchors AC to be "right on average" specifically at TWAP, but will systematically
misprice any schedule that deviates from TWAP in either direction (too fast or too patient), because
the two cost functions have different curvature — the calibration only agrees at the one point it was
fit to.

**Candidate C — match the low-participation limit.**
Ill-posed: the sqrt law's relative marginal cost (`d(impact)/dQ`) diverges as participation → 0, so
there is no finite `η` this converges to. Ruled out, not a real candidate.

**None of A/B/C is "the" calibration — they are three different modeling commitments with different
blind spots**, and picking one without saying so would just be a fourth instance of this phase's
recurring bug class (a value asserted rather than derived), now one level more subtle because the
number would *look* calibrated (fit to real data) while still encoding an arbitrary structural choice
(which participation rate, or which reference schedule, to match).

## The actually-principled fix is item 5, not a calibration of η

The honest way to remove the sqrt-vs-linear mismatch is not to force AC's linear closed form to
agree with the sqrt model at one point — it's to stop using the linear closed form and solve the
execution-cost optimization directly under the real sqrt cost function (numerically, since sqrt
breaks the closed-form derivation). That is exactly item 5's scope. **Recommendation: calibrate σ
now (cheap, unambiguous, no dependency on item 5); treat η calibration as blocked on the item 5
go/no-go rather than picking one of A/B/C as a stopgap** — a stopgap η would produce a number that
looks more rigorous than the current hardcoded 0.10 while still being an uninspected modeling
assumption, which is the wrong direction to move in during a phase explicitly about removing exactly
that kind of unstated assumption.

## What this means for the currently-running λ-grid (item 1)

The λ-grid sweep does not touch η — it only varies AC's risk-aversion `λ`, holding the flawed
linear-vs-sqrt structure fixed. So the λ-grid answers "how good can AC get within its own (currently
mismatched) model," not "how good could a properly-specified baseline get." Both numbers are worth
having and are not in conflict — the λ-grid result stands as a lower bound on what a well-tuned
*linear-model* AC can do; item 5's number, if pursued, would be the ceiling for what any
closed-form-family execution baseline could do against the real sqrt-impact simulator.
