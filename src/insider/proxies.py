"""Flag suspected insider accounts by three proxies (Section 4.2, Table 7A).

  record: per-share win rate >=90%, profit/share >=0.1, total profit >=$1000,
      and >=90% of profit in insider-prone markets.
  timing: first trade <=7d before the (single, insider-prone) event it trades,
      no trades after it resolves, trade >=$1000, profit >=$1000.
  conviction: a price-insensitive trade (moves price >=0.2) in an insider-prone
      market it wins, more than half of profit from such markets, profit >=$1000.

Uses the LLM insider-prone labels (data/processed/insider_labels.parquet). A
"win" share is a position that pays off (buy a winning token or sell a losing one).
"""
from __future__ import annotations

import duckdb
import pandas as pd

import config as C
from src.duckutil import connect as _safe_connect

TRADES_GLOB = C.TRADES_GLOB


def _con():
    con = _safe_connect()
    con.execute("PRAGMA threads=6")
    con.execute(f"CREATE VIEW tm AS SELECT * FROM read_parquet('{C.RAW/'token_map.parquet'}')")
    con.execute(f"CREATE VIEW mk AS SELECT * FROM read_parquet('{C.RAW/'markets.parquet'}')")
    con.execute(f"CREATE VIEW tr AS SELECT * FROM read_parquet('{TRADES_GLOB}')")
    con.execute(f"CREATE VIEW lab AS SELECT market_id, condition_id, label FROM read_parquet('{C.PROC/'insider_labels.parquet'}')")
    con.execute("CREATE VIEW prone AS SELECT DISTINCT condition_id FROM lab WHERE label IN ('yes','unclear')")
    return con


def build() -> None:
    if not (C.PROC / "insider_labels.parquet").exists():
        raise SystemExit("Run src.insider.llm_classify first (needs insider labels).")
    con = _con()

    # per-account, per-market building blocks
    con.execute("""
        CREATE OR REPLACE TEMP TABLE am AS
        SELECT t.account, t.condition_id AS market, t.event_id,
               (p.condition_id IS NOT NULL) AS prone,
               SUM(t.size) AS shares,
               -- winning shares: buy a winner (d=1,y=1) or sell a loser (d=-1,y=0)
               SUM(CASE WHEN (t.d=1 AND tk.y=1) OR (t.d=-1 AND tk.y=0)
                        THEN t.size ELSE 0 END) AS win_shares,
               SUM(t.size*t.d*(tk.y-t.price)) AS pnl,
               SUM(t.notional) AS volume,
               MIN(t.ts) AS first_ts, MAX(t.ts) AS last_ts,
               -- price-insensitive trade that wins: |price jump| >= 0.2 & market won by this side
               MAX(CASE WHEN abs(t.price - COALESCE(
                          lag(t.price) OVER (PARTITION BY t.token_id ORDER BY t.ts), t.price))
                          >= 0.2 THEN 1 ELSE 0 END) AS had_bigmove
        FROM tr t
        JOIN tm tk ON tk.token_id=t.token_id
        LEFT JOIN prone p ON p.condition_id=t.condition_id
        GROUP BY t.account, t.condition_id, t.event_id, prone
    """)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE acc AS
        SELECT account,
               SUM(shares) shares, SUM(win_shares) win_shares,
               SUM(pnl) pnl, SUM(volume) volume,
               COUNT(DISTINCT event_id) n_events,
               COUNT(DISTINCT market) n_markets,
               MIN(first_ts) first_ts, MAX(last_ts) last_ts,
               SUM(CASE WHEN prone THEN GREATEST(pnl,0) ELSE 0 END) prone_pos_pnl,
               SUM(GREATEST(pnl,0)) pos_pnl,
               SUM(CASE WHEN had_bigmove=1 AND pnl>0 AND prone THEN pnl ELSE 0 END) conv_pnl
        FROM am GROUP BY account
    """)
    acc = con.execute("""
        SELECT *, win_shares/NULLIF(shares,0) AS win_rate,
               pnl/NULLIF(shares,0) AS profit_per_share,
               prone_pos_pnl/NULLIF(pos_pnl,0) AS prone_profit_share
        FROM acc
    """).fetchdf()

    # event-level detail for the timing rule (single insider-prone event)
    ev = con.execute("""
        SELECT a.account, a.event_id, a.first_ts, a.last_ts, a.volume, a.pnl, a.prone,
               epoch(m.start)::BIGINT AS ev_start, epoch(m.resolved)::BIGINT AS ev_res
        FROM am a JOIN mk m ON m.condition_id = a.market
    """).fetchdf()
    con.close()

    # record
    rec = acc[(acc.win_rate >= C.REC_WINRATE)
              & (acc.profit_per_share >= C.REC_PROFIT_PER_SHARE)
              & (acc.pnl >= C.REC_MIN_PROFIT)
              & (acc.prone_profit_share >= C.REC_PRONE_PROFIT_SHARE)]

    # timing: single insider-prone event, opened <=7d before it, quiet after
    single = acc[acc.n_events == 1].account
    et = ev[ev.account.isin(single) & ev.prone]
    tim = et[(et.first_ts >= et.ev_start - C.TIM_MAX_DAYS_BEFORE * 86400)
             & (et.last_ts <= et.ev_res)
             & (et.volume >= C.TIM_MIN_VOLUME)
             & (et.pnl >= C.TIM_MIN_PROFIT)]

    # conviction
    conv = acc[(acc.conv_pnl > 0)
               & (acc.conv_pnl >= C.CONV_PROFIT_SHARE * acc.pos_pnl)
               & (acc.pnl >= C.CONV_MIN_PROFIT)]

    flags = pd.DataFrame({"account": pd.concat([rec.account, pd.Series(tim.account),
                                                conv.account]).unique()})
    flags["record"] = flags.account.isin(rec.account)
    flags["timing"] = flags.account.isin(tim.account)
    flags["conviction"] = flags.account.isin(conv.account)
    flags.to_parquet(C.PROC / "insider_flags.parquet", index=False)

    print(f"record: {len(rec):,}  timing: {tim.account.nunique():,}  "
          f"conviction: {len(conv):,}  union: {len(flags):,}")


if __name__ == "__main__":
    build()
