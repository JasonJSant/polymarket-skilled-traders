"""Maduro insider case study (Section 4.2 / Appendix C.1, Table C.1 + Fig C.1).

Finds the "Maduro out by January 31, 2026?" market, ranks the accounts that
traded it by realized PnL, and reports the top three (the paper's suspected
insiders) with their first/last trade times, PnL, and volume. Also renders the
price path with large trades split into the three flagged accounts vs. the rest.
"""
from __future__ import annotations

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C
from src.duckutil import connect as _safe_connect

TRADES_GLOB = C.TRADES_GLOB


def _find_market(con) -> pd.DataFrame:
    return con.execute(f"""
        SELECT condition_id, market_id, question, event_id, start, resolved
        FROM read_parquet('{C.RAW/'markets.parquet'}')
        WHERE lower(question) LIKE '%maduro%'
        ORDER BY volume DESC
    """).fetchdf()


def build() -> None:
    con = _safe_connect()
    mkts = _find_market(con)
    if mkts.empty:
        print("Maduro market not found (may be outside the sample window).")
        con.close()
        return
    cid = mkts.iloc[0]["condition_id"]
    print(f"market: {mkts.iloc[0]['question']}  ({cid[:12]}...)")

    acct = con.execute(f"""
        SELECT t.account,
               MIN(t.ts) first_ts, MAX(t.ts) last_ts,
               SUM(t.size*t.d*(tk.y-t.price)) pnl,
               SUM(t.notional) volume
        FROM read_parquet('{TRADES_GLOB}') t
        JOIN read_parquet('{C.RAW/'token_map.parquet'}') tk ON tk.token_id=t.token_id
        WHERE t.condition_id = '{cid}'
        GROUP BY t.account ORDER BY pnl DESC
    """).fetchdf()
    top3 = acct.head(3).copy()
    for c in ("first_ts", "last_ts"):
        top3[c] = pd.to_datetime(top3[c], unit="s")
    top3["account_short"] = top3["account"].str[:6] + "..." + top3["account"].str[-4:]
    tableC1 = top3[["account_short", "first_ts", "last_ts", "pnl", "volume"]]
    tableC1.to_csv(C.TABLES / "tableC1_maduro.csv", index=False)
    print(tableC1.to_string(index=False))

    # Fig C.1: price path + large trades (flagged vs other)
    flagged = set(top3["account"])
    trades = con.execute(f"""
        SELECT t.ts, t.size,
               CASE WHEN tk.outcome_index=0 THEN t.price ELSE 1-t.price END p_yes,
               t.account
        FROM read_parquet('{TRADES_GLOB}') t
        JOIN read_parquet('{C.RAW/'token_map.parquet'}') tk ON tk.token_id=t.token_id
        WHERE t.condition_id='{cid}' ORDER BY t.ts
    """).fetchdf()
    con.close()
    trades["dt"] = pd.to_datetime(trades["ts"], unit="s")
    big = trades[trades["size"] >= 1000]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trades["dt"], trades["p_yes"], color="#00008B", lw=1, label="Market price")
    ax2 = ax.twinx()
    fb = big[big.account.isin(flagged)]; ob = big[~big.account.isin(flagged)]
    ax2.bar(ob["dt"], ob["size"], width=0.005, color="gray", alpha=0.5, label="Other large trades")
    ax2.bar(fb["dt"], fb["size"], width=0.005, color="red", label="Flagged large trades")
    ax.set_ylabel("Market Price"); ax2.set_ylabel("Trade Size (shares)")
    ax.set_title("Polymarket price for Maduro out"); ax.legend(loc="upper left")
    fig.savefig(C.FIGURES / "figC1_maduro.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote figC1_maduro")


if __name__ == "__main__":
    build()
