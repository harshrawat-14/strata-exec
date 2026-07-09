.PHONY: test build evaluate unified-eval precompute demo dev docker-build docker-run

# Build all Rust binaries (strata-exec, research-sim, rl-env) in release mode.
build:
	cargo build --release

# Run the Rust test suite (415+ tests, no RPC/network needed).
test:
	cargo test --release

# Evaluate the default RL model against all historical dates.
evaluate:
	python rl/evaluate.py \
		--passive-model rl/models/ppo_lstm_v5_adaptive_best/best_model \
		--episodes 50 \
		--n-state-dims 12

# Run all 5 strategies (TWAP, Heuristic, AC Optimal, Adaptive AC, RL) through
# the same counterfactual LOB / OW-impact pipeline for a fair comparison.
unified-eval:
	python rl/evaluate.py \
		--passive-model rl/models/ppo_lstm_v5_adaptive_best/best_model \
		--episodes 50 \
		--n-state-dims 12 \
		--include-test-dates \
		--unified-comparison

# Regenerate demo_results/ from the current model + simulation presets.
# NOTE: only run this deliberately — demo_results/ backs the live DEMO_MODE
# deployment and must not be overwritten as a side effect of other work.
precompute:
	python scripts/precompute_demo_data.py

# Serve the backend in demo mode (pre-computed JSON, no live compute).
demo:
	DEMO_MODE=true uvicorn web.main:app --host 0.0.0.0 --port 8000 --reload

# Serve the backend in live mode (real Rust subprocess compute).
dev:
	DEMO_MODE=false uvicorn web.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t strataexec .

docker-run:
	docker run --rm -p 8000:8000 -e DEMO_MODE=true strataexec
