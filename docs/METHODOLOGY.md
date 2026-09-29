# Methodology and reproduction notes

Methodology choices, data-source constraints, and validation checks for the
reproduction of Gomez-Cram, Guo, Jensen and Kung (2026).

## Data sources

- GAMMA API (`gamma-api.polymarket.com`): market and event metadata (condition
  ids, outcome token ids, outcomes, resolution prices, categories, dates,
  neg-risk flag, event grouping).
- Goldsky orderbook subgraph: on-chain `OrderFilled` events (maker, taker, asset
  ids, amounts filled), decoded into per-account trades with a maker/taker flag.

Prices are derived from trades (forward-filled last-trade price) rather than the
CLOB quote series. Bin prices use the index-0 (YES) basis.

## Operator-leg cleaning

The CTF Exchange emits an `OrderFilled` event for both the maker order and the
taker order, and routes one leg of every trade through operator/matching wallets.
Summing raw fills therefore double-counts share volume (about 2x), scrambles the
maker/taker (limit-order) tag, and adds noise to signed order flow.

Ten operator addresses were identified (high-frequency wallets with zero trades
in Polymarket's own data-api). Dropping every fill in which an operator is the
maker or the taker reconstructs the canonical one-sided volume (ratio about 1.00
versus the data-api, across binary and neg-risk markets) and restores correct
maker/taker semantics for genuine trader-to-trader fills.

Implementation: `config.OPERATORS` and `src/build/clean_trades.py`. The analysis
reads the cleaned trades through `config.TRADES_GLOB`, which prefers the cleaned
shards when present.

Effect on the headline numbers (reproduced on cleaned data vs the paper):

| Metric | Reproduced | Paper |
| --- | --- | --- |
| Market makers | 1,080 | 1,660 |
| Skilled share | 3.23% | 3.16% |
| Anti-skilled share | 5.92% | 6.15% |
| Lucky share | 29.38% | 28.93% |
| Unlucky share | 61.41% | 61.66% |
| Total one-sided volume | $12.6B | $13.76B |
| Persistence (p-value correlation) | 0.34 | 0.35 |
| Table 4A (delta-p, skilled OIB) | +1.13 | +2.05 |
| Table 4B (outcome, skilled OIB) | +4.04 | +7.77 |
| Table 5 (accuracy, skilled VOL) | -5.90 | -4.95 |
| Table 5 (final 20% window) | -11.7 | -9.53 |

## Source data ceiling

The Goldsky orderbook subgraph stops at 2026-04-28; it appears to have stopped
indexing around that date. April 2026 is therefore the furthest month reachable
from this source (January to March 2026 complete, April partial). Data past that
point would require a different feed and is out of scope.

## Paper window vs extended universe

The paper window is markets resolved on or before 2025-12-31. The collection
horizon (`config.COLLECT_END`) extends to 2026-04-30 to include the additional
months. Rebuilding the market universe produces a superset (the original resolved
markets plus the 2026-resolving markets).

A snapshot of the paper-window universe (markets, token_map, events) is kept in
`data/raw/_repro_2023_2025/`. Because the analysis inner-joins trades to
`token_map`, the paper-window result is recovered exactly by joining against this
snapshot: tokens that exist only in the extended universe drop out of the join.

## Deviations from the paper

- Prices are derived from trades (forward-filled last-trade price) rather than
  the CLOB quote series.
- Before applying the "at least one trade" market filter, the market universe is
  larger than the paper's; the filter brings the counts into line.

## Validation checks

- Total PnL across all accounts sums to zero (confirms correct maker/taker
  decoding).
- The p-value distribution shows tail spikes and a roughly uniform middle
  (matches the paper's Figure 2 shape).
- Table 4 order-imbalance coefficients: positive for skilled, negative for
  anti-skilled and unlucky, insignificant for lucky.
- Table 5 accuracy: the skilled volume effect grows toward resolution and is
  strongest in the final 20% of a market's life.

## Known limitations

- Skilled-to-skilled persistence reproduces at about 0.34 versus the paper's
  0.46, while the p-value correlation matches (0.34 vs 0.35). The difference is
  most likely in how the skilled/lucky boundary moves between the train and test
  splits rather than a data issue.
- The pooled Table 4B coefficient is specification-sensitive. It is strongly
  positive near resolution (raw, last 5%), and the sign can flip when a
  lagged-price control is added, because near resolution the price is close to
  the outcome and absorbs it. The economic conclusion (skilled order flow
  predicts outcomes, strongly near resolution) reproduces.
