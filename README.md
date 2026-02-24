# StrataExec

Production-grade adaptive execution engine for DeFi — monitors on-chain liquidity and volatility to size order chunks through a risk-gated scheduler.

## Prerequisites

- **Rust** 1.70+ (`rustup` recommended)
- An Ethereum JSON-RPC endpoint (local node, Alchemy, Infura, etc.)

## Setup

```bash
# 1. Clone the repo
git clone <repo-url> && cd StrataExec

# 2. Create your .env from the example
cp .env.example .env

# 3. Edit .env with your values
#    At minimum, set RPC_URL and PAIR_ADDRESS
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `RPC_URL` | ✅ | — | Ethereum JSON-RPC endpoint |
| `PAIR_ADDRESS` | ✅ | — | Uniswap V2 pair address (checksummed) |
| `TOTAL_NOTIONAL` | ✅ | — | Total notional to execute (> 0) |
| `BASE_CHUNK` | ✅ | — | Base chunk size (> 0) |
| `MAX_SLIPPAGE_PCT` | ✅ | — | Max slippage fraction, (0, 1) |
| `EMERGENCY_RESERVE_FRAC` | ✅ | — | Emergency reserve fraction, [0, 1) |
| `VOL_WINDOW` | ✅ | — | Volatility window size (≥ 2) |
| `SIGMA_REF` | ✅ | — | Reference volatility for chunk scaling |
| `LIQUIDITY_REF` | ✅ | — | Reference liquidity for chunk scaling |
| `LOG_LEVEL` | | `info` | Tracing log level |
| `HTTP_PORT` | | `3000` | HTTP server port |

## Run

```bash
cargo run
```

The engine starts polling blocks and the HTTP control plane listens on `0.0.0.0:<HTTP_PORT>`.

## HTTP API

### Health check

```bash
curl http://localhost:3000/health
# {"status":"ok"}
```

### Engine status

```bash
curl http://localhost:3000/status
# {"is_running":true,"remaining_notional":10000.0,"last_volatility":0.0}
```

### Metrics

```bash
curl http://localhost:3000/metrics
# {"blocks_processed":42,"chunks_executed":10,"risk_denials":2}
```

### Start / Stop engine

```bash
# Pause execution (engine keeps polling but skips scheduling)
curl -X POST http://localhost:3000/stop
# {"is_running":false}

# Resume execution
curl -X POST http://localhost:3000/start
# {"is_running":true}
```

## Tests

```bash
cargo test
```

## Architecture

```
src/
├── api/          # Axum HTTP control plane (AppState, routes)
├── blockchain/   # RPC client with retry + exponential backoff
├── config/       # Environment-based configuration (dotenvy)
├── core/
│   ├── engine    # Block-polling orchestrator
│   ├── scheduler # Adaptive chunk-sizing
│   ├── risk      # Multi-constraint slippage risk manager
│   └── state     # Engine lifecycle states
├── features/     # Volatility, liquidity, drift estimators
├── ml/           # (future) ML-based predictions
├── types/        # Shared type definitions
└── utils/        # Logging setup
```

## License

Private — all rights reserved.
