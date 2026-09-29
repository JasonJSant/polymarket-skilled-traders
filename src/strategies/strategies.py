"""Public-information trading strategies (Section 4.3).

  * calibration: Fig E.2, market price at 90% of lifespan vs realized
    outcome, 50 price bins (drop <$1000-volume markets).
  * longshot: Fig 5, net YES-basis buy volume by group across 5 price
    bins (favorite-longshot bias).
  * arbitrage: Fig 4, net buy volume + PnL of complete No-baskets in
    neg-risk events, by group.
  * dispute_window: Fig E.1, net buy of the eventually-winning token in the
    last 2h before resolution, by group.

Descriptive figures use the full-sample classification (classification.parquet).
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

import config as C
from src.duckutil import connect as _safe_connect

TRADES_GLOB = C.TRADES_GLOB


def _con():
    con = _safe_connect()
    con.execute("PRAGMA threads=6")
    con.execute(f"CREATE OR REPLACE VIEW tm AS SELECT * FROM read_parquet('{C.RAW/'token_map.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW mk AS SELECT * FROM read_parquet('{C.RAW/'markets.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW cls AS SELECT account, \"group\" AS grp FROM read_parquet('{C.PROC/'classification.parquet'}')")
    con.execute(f"CREATE OR REPLACE VIEW tr AS SELECT * FROM read_parquet('{TRADES_GLOB}')")
    return con


def calibration() -> pd.DataFrame:
    """Price at 90% of lifespan vs outcome, 50 bins, markets with >=$1000 vol."""
    con = _con()
    df = con.execute(f"""
        WITH mext AS (
            SELECT m.condition_id AS market, epoch(m.start)::BIGINT s, epoch(m.resolved)::BIGINT e,
                   t0.y AS y0
            FROM mk m JOIN tm t0 ON t0.condition_id=m.condition_id AND t0.outcome_index=0
            WHERE m.resolved>m.start
        ),
        pt AS (
            SELECT t.condition_id AS market, t.ts, t.notional,
                   CASE WHEN tk.outcome_index=0 THEN t.price ELSE 1-t.price END AS p_yes
            FROM tr t JOIN tm tk ON tk.token_id=t.token_id
        ),
        j AS (
            SELECT p.market, p.ts, p.p_yes, p.notional, e.s, e.e, e.y0,
                   e.s + 0.9*(e.e-e.s) AS cut
            FROM pt p JOIN mext e ON e.market=p.market
        )
        SELECT market, ANY_VALUE(y0) y0, SUM(notional) vol,
               arg_max(p_yes, CASE WHEN ts<=cut THEN ts ELSE NULL END) AS price90
        FROM j GROUP BY market
    """).fetchdf()
    con.close()
    df = df.dropna(subset=["price90"])
    df = df[df["vol"] >= C.CALIB_MIN_VOLUME]
    df["bin"] = np.clip((df["price90"] * C.CALIB_BINS).astype(int), 0, C.CALIB_BINS - 1)
    out = df.groupby("bin").agg(price=("price90", "mean"),
                                outcome=("y0", "mean"),
                                n=("y0", "size")).reset_index()
    return out


def longshot() -> pd.DataFrame:
    """Net YES-basis buy volume (million shares) by group across 5 price bins."""
    con = _con()
    edges = C.LONGSHOT_BINS
    case = "CASE " + " ".join(
        f"WHEN p_yes>={lo} AND p_yes<{hi + (hi==1.0)*1e-9} THEN '{lo}-{hi}'"
        for lo, hi in edges) + " END"
    df = con.execute(f"""
        WITH pt AS (
            SELECT COALESCE(c.grp,'Unclassified') grp,
                   CASE WHEN tk.outcome_index=0 THEN t.price ELSE 1-t.price END AS p_yes,
                   CASE WHEN tk.outcome_index=0 THEN t.d ELSE -t.d END * t.size AS net
            FROM tr t JOIN tm tk ON tk.token_id=t.token_id
            LEFT JOIN cls c ON c.account=t.account
        )
        SELECT {case} AS price_bin, grp, SUM(net)/1e6 AS net_mshares
        FROM pt WHERE {case} IS NOT NULL GROUP BY price_bin, grp
    """).fetchdf()
    con.close()
    return df


def arbitrage() -> pd.DataFrame:
    """Net buy volume + PnL of complete No-baskets in neg-risk events, by group.

    A trader's trades are grouped into blocks (new block when the gap exceeds
    10s). A block that buys the No token of every market in one neg-risk event
    is an executed basket arbitrage; its size is the min No quantity bought.
    """
    con = _con()
    # neg-risk events and their market counts
    con.execute("""
        CREATE OR REPLACE TEMP TABLE nr AS
        SELECT m.event_id, COUNT(*) AS n_out
        FROM mk m WHERE m.neg_risk GROUP BY m.event_id HAVING COUNT(*)>=2
    """)
    # No-token buys (index-1 buys) in neg-risk events, with group + block id
    con.execute("""
        CREATE OR REPLACE TEMP TABLE nobuys AS
        WITH b AS (
            SELECT t.account, t.event_id, t.condition_id AS market, t.ts, t.size,
                   t.price, COALESCE(c.grp,'Unclassified') grp
            FROM tr t
            JOIN tm tk ON tk.token_id=t.token_id
            JOIN nr n ON n.event_id=t.event_id
            LEFT JOIN cls c ON c.account=t.account
            WHERE tk.outcome_index=1 AND t.d=1          -- buying No
        ),
        g AS (
            SELECT *,
                   CASE WHEN ts - LAG(ts) OVER (PARTITION BY account ORDER BY ts) > 10
                        OR LAG(ts) OVER (PARTITION BY account ORDER BY ts) IS NULL
                        THEN 1 ELSE 0 END AS newblock
            FROM b
        )
        SELECT *, SUM(newblock) OVER (PARTITION BY account ORDER BY ts) AS block
        FROM g
    """)
    # a block is an arb basket if it covers all N outcomes of one event
    df = con.execute("""
        WITH per_block AS (
            SELECT account, grp, event_id, block,
                   COUNT(DISTINCT market) AS mkts_hit,
                   MIN(size) AS basket_qty,
                   SUM(size*price) AS no_cost,
                   ANY_VALUE(n_out) AS n_out
            FROM nobuys nb JOIN nr n USING (event_id)
            GROUP BY account, grp, event_id, block
        )
        SELECT grp,
               SUM(CASE WHEN mkts_hit=n_out THEN basket_qty*n_out ELSE 0 END)/1e6
                   AS net_buy_mshares,
               SUM(CASE WHEN mkts_hit=n_out
                        THEN basket_qty*((n_out-1) - no_cost/NULLIF(basket_qty,0)*0)
                        ELSE 0 END)/1e6 AS pnl_m
        FROM per_block GROUP BY grp
    """).fetchdf()
    con.close()
    return df


def dispute_window() -> pd.DataFrame:
    """Net buy of the eventually-winning token in the last 2h before resolution."""
    con = _con()
    df = con.execute(f"""
        WITH mext AS (
            SELECT condition_id AS market, epoch(resolved)::BIGINT e FROM mk
        ),
        pt AS (
            SELECT COALESCE(c.grp,'Unclassified') grp,
                   t.d * t.size AS net_shares, t.d*t.size*t.price AS net_usd,
                   tk.y AS token_win, t.ts, e.e
            FROM tr t JOIN tm tk ON tk.token_id=t.token_id
            JOIN mext e ON e.market=t.condition_id
            LEFT JOIN cls c ON c.account=t.account
            WHERE t.ts >= e.e - 7200 AND t.ts <= e.e
        )
        SELECT grp,
               SUM(CASE WHEN token_win=1 THEN net_shares ELSE 0 END)/1e6 AS net_win_mshares,
               SUM(CASE WHEN token_win=1 THEN net_usd ELSE 0 END)/1e6 AS pnl_m
        FROM pt GROUP BY grp
    """).fetchdf()
    con.close()
    return df


def build() -> None:
    for name, fn in [("calibration", calibration), ("longshot", longshot),
                     ("arbitrage", arbitrage), ("dispute", dispute_window)]:
        df = fn()
        df.to_parquet(C.PROC / f"strat_{name}.parquet", index=False)
        print(f"=== {name} ===")
        print(df.to_string(index=False)[:800])
        print()


if __name__ == "__main__":
    build()
