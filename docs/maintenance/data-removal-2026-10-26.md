# Data removal — 2026-10-26

Era 1's raw market data was deleted at the user's request, after era 1 was recorded
([../history/era-1-binance-bookdepth.md](../history/era-1-binance-bookdepth.md)) and tagged
`era-1-binance-bookdepth`. Written before deleting.

| path | contents | size | files |
|---|---|---:|---:|
| `TradeData/AggTrades/` | Binance BTCUSDT perpetual aggTrades, 25 dates | 3.2 GB | 25 |
| `TradeData/BookDepth/` | Binance BTCUSDT perpetual bookDepth (±1–5% bands), 25 dates | 38 MB | 25 |
| `TradeData/Klines/` | Binance BTCUSDT perpetual daily candles, 2023-12 → 2024-12 | 132 KB | 26 |

All files are gitignored (never in the repository) and re-downloadable from
`data.binance.vision` — URLs and commands in the era-1 record. Kept: committed derived
tables (`rl/results/daily_volume.csv`, `rl/results/adv20.csv`, every era-1 results CSV),
model checkpoints (`rl/models/`), and `TradeData/BybitOB/` (era 2).

Effect on the code: tests and the golden regression guard that need these files skip
cleanly until era-2 data and fixtures replace them; CI never had the data.
