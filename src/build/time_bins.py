"""Build the 50-bin market panel for the price-discovery tests (Sections 3.4-3.5).

Each market's lifespan [start, resolution] is standardized to the unit interval
and split into 50 equal bins. Everything is put on the YES (outcome-index-0)
basis: buying the index-1 token is selling index-0, and its price is 1 - price.

For each (market, bin), per account group g (train-set classification):
  OIB^g = sum_g(size * d_yes) / sum_all(size)      (net YES buy share, [-.5,.5])
  VOL^g = sum_g(size)         / sum_all(size)       (volume share, [0,1])
and the bin-end YES price (last trade in the bin, forward-filled across empty
bins). The panel is restricted to TEST-set markets, since accounts are classified
on the training set and results are reported out of sample.

Outputs data/processed/bins.parquet with, per (market, bin): price, next-bin
price change, final outcome, change in |pricing error| (PD), lagged controls,
the bin-midpoint calendar month (for clustering), and OIB_g / VOL_g per group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C

TRADES_GLOB = C.TRADES_GLOB
GROUP_KEYS = {"Market Maker": "mm", "Skilled": "sk", "Lucky": "lk",
              "Unlucky": "ul", "Anti-skilled": "as"}


def _aggregate(con) -> pd.DataFrame:
    # markets with epoch lifespan bounds + index-0 outcome (YES-basis outcome)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE mext AS
        SELECT m.condition_id AS market, m.event_id,
               epoch(m.start)::BIGINT AS start_ts,
               epoch(m.resolved)::BIGINT AS res_ts,
               date_diff('day', m.start, m.resolved) AS duration_days,
               t0.y AS y0
        FROM read_parquet('{C.RAW / 'markets.parquet'}') m
        JOIN read_parquet('{C.RAW / 'token_map.parquet'}') t0
             ON t0.condition_id = m.condition_id AND t0.outcome_index = 0
        WHERE m.resolved > m.start
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE test_ev AS
        SELECT event_id FROM read_parquet('{C.PROC / 'event_split.parquet'}')
        WHERE split = 1
    """)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE ctrain AS
        SELECT account, "group" AS grp
        FROM read_parquet('{C.PROC / 'classification_train.parquet'}')
    """)
    # Per (market,bin) aggregates. The per-trade YES-basis derivation is folded
    # into a subquery that DuckDB pipelines straight into the GROUP BY, so the
    # 544M rows are never materialized, avoiding memory exhaustion from a temp table.
    grp_cols = []
    for g, k in GROUP_KEYS.items():
        grp_cols.append(
            f"SUM(CASE WHEN grp='{g}' THEN size*d_yes ELSE 0 END) AS net_{k}")
        grp_cols.append(
            f"SUM(CASE WHEN grp='{g}' THEN size ELSE 0 END) AS vol_{k}")
    q = f"""
        SELECT market, ANY_VALUE(event_id) AS event_id, bin,
               ANY_VALUE(y0) AS y0, ANY_VALUE(duration_days) AS duration_days,
               SUM(size) AS vol_total,
               arg_max(p_yes, ts) AS bin_price,
               strftime(to_timestamp(CAST(AVG(ts) AS BIGINT)), '%Y-%m') AS month,
               {', '.join(grp_cols)}
        FROM (
            SELECT e.market, e.event_id, e.y0, e.duration_days,
                   CASE WHEN tm.outcome_index=0 THEN t.price ELSE 1-t.price END AS p_yes,
                   CASE WHEN tm.outcome_index=0 THEN t.d    ELSE -t.d   END AS d_yes,
                   t.size,
                   COALESCE(c.grp,'Unclassified') AS grp,
                   t.ts,
                   LEAST(49, GREATEST(0, CAST(
                       floor(50.0*(t.ts - e.start_ts)/(e.res_ts - e.start_ts)) AS INT)))
                       AS bin
            FROM read_parquet('{TRADES_GLOB}') t
            JOIN read_parquet('{C.RAW / 'token_map.parquet'}') tm ON tm.token_id=t.token_id
            JOIN mext e ON e.market = t.condition_id
            JOIN test_ev te ON te.event_id = e.event_id
            LEFT JOIN ctrain c ON c.account = t.account
            WHERE t.ts >= e.start_ts AND t.ts <= e.res_ts
        )
        GROUP BY market, bin
    """
    return con.execute(q).fetchdf()


def _panel(agg: pd.DataFrame) -> pd.DataFrame:
    # Reindex every market onto the full 0..49 bin grid so lags/leads are well
    # defined, then forward/back-fill the price path across empty bins.
    agg = agg.sort_values(["market", "bin"])
    markets = agg["market"].unique()
    full_idx = pd.MultiIndex.from_product([markets, range(C.N_BINS)],
                                          names=["market", "bin"])
    p = agg.set_index(["market", "bin"]).reindex(full_idx).reset_index()

    # carry static market attributes
    for col in ["event_id", "y0", "duration_days"]:
        p[col] = p.groupby("market")[col].transform(
            lambda s: s.ffill().bfill())
    # empty bins: zero volume / zero group activity
    fillzero = ["vol_total"] + [f"net_{k}" for k in GROUP_KEYS.values()] \
        + [f"vol_{k}" for k in GROUP_KEYS.values()]
    p[fillzero] = p[fillzero].fillna(0.0)
    # price path: ffill then bfill within market
    p["bin_price"] = p.groupby("market")["bin_price"].transform(
        lambda s: s.ffill().bfill())
    p["month"] = p.groupby("market")["month"].transform(lambda s: s.ffill().bfill())

    # group OIB / VOL (guard divide-by-zero on empty bins)
    denom = p["vol_total"].replace(0, np.nan)
    for g, k in GROUP_KEYS.items():
        p[f"OIB_{k}"] = (p[f"net_{k}"] / denom).fillna(0.0)
        p[f"VOL_{k}"] = (p[f"vol_{k}"] / denom).fillna(0.0)

    # outcomes and dynamics (dependent vars later scaled x100)
    p["price"] = p["bin_price"]
    p["dp_next"] = p.groupby("market")["price"].shift(-1) - p["price"]
    p["abs_err"] = (p["price"] - p["y0"]).abs()
    p["PD"] = p.groupby("market")["abs_err"].diff()          # |e_t| - |e_{t-1}|

    # lagged controls
    p["price_lag"] = p.groupby("market")["price"].shift(1)
    p["dpchg_lag"] = p.groupby("market")["price"].shift(1) \
        - p.groupby("market")["price"].shift(2)
    p["logvol_lag"] = np.log1p(p.groupby("market")["vol_total"].shift(1))
    p["duration"] = np.log1p(p["duration_days"].clip(lower=0))
    return p


def build() -> None:
    from src.duckutil import connect
    con = connect()
    agg = _aggregate(con)
    con.close()
    panel = _panel(agg)
    panel.to_parquet(C.PROC / "bins.parquet", index=False)
    print(f"bins panel: {len(panel):,} market-bin rows over "
          f"{panel['market'].nunique():,} test markets")
    print(f"non-empty bins: {(panel['vol_total']>0).sum():,}")
    print(f"mean skilled OIB: {panel['OIB_sk'].mean():.4f}  "
          f"mean skilled VOL: {panel['VOL_sk'].mean():.4f}")


if __name__ == "__main__":
    build()
