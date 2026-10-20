# Strategy comparison (2026-10-20)

Unseen test period: 2025-04-22 to today. Costs of 0.1% per trade included.
The model was trained only on data up to 2025-03-10.

| Version | Trades | Per month | Win rate | Avg trade | Avg win | Avg loss | Profit factor | Median days |
|---|---|---|---|---|---|---|---|---|
| A. Old rules (v4) | 1378 | 76.8 | 24.0% | 0.66% | 14.82% | -3.82% | 1.23 | 7 |
| B. Rule only (current rule) | 6644 | 370.1 | 24.5% | 0.72% | 13.02% | -3.26% | 1.29 | 7 |
| C. Calmer entry (RSI<60, not stretched) | 5579 | 310.8 | 23.5% | 0.45% | 12.2% | -3.15% | 1.19 | 7 |
| D. Calmer entry + market filter | 5371 | 299.2 | 22.8% | 0.18% | 11.64% | -3.2% | 1.07 | 7 |
| F. D + strict per-stock test | 1035 | 57.7 | 24.3% | 0.84% | 14.49% | -3.53% | 1.31 | 7 |
| G. Rule + model (top half) | 3025 | 168.5 | 25.4% | 0.7% | 12.35% | -3.25% | 1.29 | 7 |
| H. Rule + model (top 30%) | 1645 | 91.6 | 26.2% | 0.72% | 12.13% | -3.33% | 1.29 | 8 |
| I. Rule + model (top 10%) | 409 | 22.8 | 26.9% | 1.09% | 14.47% | -3.83% | 1.39 | 9 |

## Prediction model (AUC 0.495; 0.5 = coin flip)

Stocks ranked each day by the model, split into 10 groups (10 = most confident).
Next-month result vs the S&P 500:

| Group | Beat the market | Avg vs market | Samples |
|---|---|---|---|
| 1 | 42.4% | -0.78% | 31773 |
| 2 | 43.5% | -0.70% | 32130 |
| 3 | 44.5% | -0.43% | 32014 |
| 4 | 45.0% | -0.38% | 32040 |
| 5 | 46.1% | -0.15% | 31863 |
| 6 | 46.3% | -0.07% | 32130 |
| 7 | 45.8% | -0.19% | 32040 |
| 8 | 46.3% | +0.04% | 32014 |
| 9 | 46.0% | -0.09% | 32130 |
| 10 | 47.2% | +0.40% | 32130 |
