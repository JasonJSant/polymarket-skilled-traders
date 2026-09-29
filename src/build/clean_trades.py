"""Remove Polymarket operator/matching legs from the raw subgraph fills.

The CTF Exchange emits an OrderFilled for BOTH the maker's and the taker's order,
routing one leg of every trade through an operator wallet (config.OPERATORS).
Summing raw fills therefore double-counts share volume (~2x) and scrambles the
maker/taker (limit-order) tag. Dropping every fill in which an operator is the
maker or taker reconstructs Polymarket's canonical one-sided volume exactly
(verified ratio 1.00 vs data-api across binary and neg-risk markets) and leaves
genuine trader-to-trader fills with correct maker/taker semantics.

Reads data/raw/trades/YYYY-MM/*.parquet, writes the cleaned equivalents to
data/processed/trades_clean/YYYY-MM/. Processes one month at a time (both legs of
a trade share a timestamp, so per-month cleaning is exact) to bound memory.
Idempotent: months already written are skipped.

Usage:  python -m src.build.clean_trades
"""
from __future__ import annotations

import time

import config as C
from src.build.time_bins import GROUP_KEYS  # noqa: F401
from src.collect.trades import months
from src.duckutil import connect


def clean_month(con, ym: str) -> tuple[int, int] | None:
    src_glob = str(C.TRADES_DIR / ym / "*.parquet")
    import glob
    if not glob.glob(src_glob):
        return None
    out_dir = C.CLEAN_TRADES_DIR / ym
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "part-00000.parquet"
    if out.exists():
        return None  # already cleaned
    ops = C.operator_sql_list()
    # anti-join: drop every row whose fill touched an operator on either leg
    con.execute(f"""
        COPY (
            SELECT t.*
            FROM read_parquet('{src_glob}') t
            WHERE t.fill_id NOT IN (
                SELECT DISTINCT fill_id FROM read_parquet('{src_glob}')
                WHERE account IN ({ops})
            )
        ) TO '{out}' (FORMAT parquet)
    """)
    raw_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{src_glob}')").fetchone()[0]
    clean_n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out}')").fetchone()[0]
    return raw_n, clean_n


def build() -> None:
    con = connect()
    con.execute("PRAGMA threads=6")
    tot_raw = tot_clean = 0
    for ym in months():
        t0 = time.time()
        res = clean_month(con, ym)
        if res is None:
            print(f"[{ym}] skipped (no source or already clean)", flush=True)
            continue
        raw_n, clean_n = res
        tot_raw += raw_n
        tot_clean += clean_n
        print(f"[{ym}] {raw_n:>11,} -> {clean_n:>11,} rows "
              f"({100*clean_n/raw_n:.0f}% kept) in {time.time()-t0:.0f}s", flush=True)
    con.close()
    if tot_raw:
        print(f"\nTOTAL rows {tot_raw:,} -> {tot_clean:,} "
              f"({100*tot_clean/tot_raw:.0f}% kept)")
    print(f"clean trades in {C.CLEAN_TRADES_DIR}")


if __name__ == "__main__":
    build()
