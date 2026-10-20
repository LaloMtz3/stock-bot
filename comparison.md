# Strategy comparison (2026-10-20)

Unseen test period: 2025-04-22 to today. Costs of 0.1% per trade included.
Per-stock tests only used data from before the test period.

| Version | Stocks allowed | Trades | Per month | Win rate | Avg trade | Avg win | Avg loss | Profit factor | Median days held | Buy & hold, same stocks |
|---|---|---|---|---|---|---|---|---|---|---|
| A. Old rules (v4) | 198 | 1378 | 76.8 | 24.0% | 0.66% | 14.82% | -3.82% | 1.23 | 7 | 69.1% |
| B. Old entry, no per-stock test | 898 | 6643 | 370.0 | 24.5% | 0.73% | 13.02% | -3.26% | 1.3 | 7 | 59.1% |
| C. Calmer entry (RSI<60, not stretched) | 898 | 5578 | 310.7 | 23.5% | 0.46% | 12.2% | -3.15% | 1.19 | 7 | 59.1% |
| D. Calmer entry + market filter | 898 | 5370 | 299.1 | 22.8% | 0.18% | 11.64% | -3.2% | 1.07 | 7 | 59.1% |
| E. D + loose per-stock test | 409 | 2488 | 138.6 | 23.0% | 0.29% | 12.13% | -3.26% | 1.12 | 7 | 60.4% |
| F. D + strict per-stock test | 182 | 1034 | 57.6 | 24.3% | 0.84% | 14.49% | -3.53% | 1.32 | 7 | 85.4% |

Profit factor = total gains / total losses. Above 1 makes money; 1.5+ is solid.
Buy & hold = average return from simply holding the same allowed stocks through the test period.
