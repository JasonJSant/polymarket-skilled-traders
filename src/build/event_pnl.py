"""Build the per-(account, event) PnL contributions and per-account features.

The sign-randomization skill test only needs, for each account, the vector of its
event-level PnL contributions  S_e = sum_{k in K_e} q_k * d_k * (Y_k - p_k)
plus a handful of account features for the market-maker rule and reporting.

Everything is computed with DuckDB straight off the sharded trade parquet joined
to token payoffs (token_map.y), so nothing large is ever held in pandas.

Outputs (data/processed/):
  * account_event_pnl.parquet  (account, event_id, S_e): skill-test input
  * account_features.parquet   per-account: n_events, n_markets, PnL, volume,
                               maker_volume_share, both_sides_share (MM rule)
"""
from __future__ import annotations

import argparse

import config as C
from src.duckutil import connect

TRADES_GLOB = C.TRADES_GLOB
TM = str(C.RAW / "token_map.parquet")


def build() -> None:
    con = connect()

    # S_e: account x event PnL contribution (the skill-test unit). The join is
    # folded directly into the GROUP BY so the 544M rows stream through without
    # ever being materialized, avoiding the memory exhaustion of a temp table.
    con.execute(f"""
        COPY (
            SELECT t.account, t.event_id,
                   SUM(t.size * t.d * (tm.y - t.price)) AS s_e,
                   COUNT(*) AS n_trades,
                   SUM(t.notional) AS volume
            FROM read_parquet('{TRADES_GLOB}') t
            JOIN read_parquet('{TM}') tm ON tm.token_id = t.token_id
            WHERE t.event_id IS NOT NULL
            GROUP BY t.account, t.event_id
        ) TO '{C.PROC / 'account_event_pnl.parquet'}' (FORMAT parquet)
    """)

    # Per (account, market) aggregates on disk (much smaller than the raw rows),
    # used to build the market-maker features.
    con.execute(f"""
        COPY (
            SELECT t.account, t.condition_id AS market_cid,
                   SUM(t.notional) AS vol,
                   SUM(CASE WHEN t.is_maker THEN t.notional ELSE 0 END) AS maker_vol,
                   MAX(CASE WHEN t.d=1 THEN 1 ELSE 0 END) AS has_buy,
                   MAX(CASE WHEN t.d=-1 THEN 1 ELSE 0 END) AS has_sell
            FROM read_parquet('{TRADES_GLOB}') t
            JOIN read_parquet('{TM}') tm ON tm.token_id = t.token_id
            GROUP BY t.account, t.condition_id
        ) TO '{C.PROC / 'per_mkt.parquet'}' (FORMAT parquet)
    """)

    # Per-account features (n_events, n_markets, PnL, volume, maker/both-sides
    # shares) from the two small intermediates.
    con.execute(f"""
        COPY (
            SELECT a.account, a.n_events, m.n_markets, a.pnl, m.volume,
                   m.maker_volume_share, m.both_sides_share
            FROM (
                SELECT account, COUNT(*) AS n_events, SUM(s_e) AS pnl
                FROM read_parquet('{C.PROC / 'account_event_pnl.parquet'}')
                GROUP BY account
            ) a
            JOIN (
                SELECT account, COUNT(*) AS n_markets, SUM(vol) AS volume,
                       SUM(maker_vol) / NULLIF(SUM(vol),0) AS maker_volume_share,
                       AVG(CASE WHEN has_buy=1 AND has_sell=1 THEN 1.0 ELSE 0.0 END)
                           AS both_sides_share
                FROM read_parquet('{C.PROC / 'per_mkt.parquet'}')
                GROUP BY account
            ) m USING (account)
        ) TO '{C.PROC / 'account_features.parquet'}' (FORMAT parquet)
    """)

    ne = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{C.PROC / 'account_features.parquet'}')"
    ).fetchone()[0]
    npairs = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{C.PROC / 'account_event_pnl.parquet'}')"
    ).fetchone()[0]
    tot = con.execute(
        f"SELECT SUM(pnl), SUM(volume) FROM "
        f"read_parquet('{C.PROC / 'account_features.parquet'}')"
    ).fetchone()
    print(f"accounts: {ne:,} | account-event pairs: {npairs:,}")
    print(f"sum PnL (should be ~0, zero-sum): {tot[0]:,.0f}")
    print(f"sum account volume (2-sided): ${tot[1]/1e9:.2f}B  "
          f"(1-sided ~= ${tot[1]/2/1e9:.2f}B)")
    con.close()


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    build()
