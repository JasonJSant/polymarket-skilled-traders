"""Memory-safe DuckDB connections for the full-scale data (544M trade rows).

The full sweep is far larger than RAM, so every connection must:
  * disable insertion-order preservation (streaming aggregations, big memory win),
  * spill to a temp dir on the large data drive (not the default),
  * cap memory so it spills early instead of running out of memory,
  * use a moderate thread count to bound peak memory.

Queries must also avoid materializing the full row set into a TEMP TABLE; fold
joins straight into the GROUP BY so DuckDB pipelines them.
"""
from __future__ import annotations

import duckdb

import config as C

_TMP = C.DATA / "duckdb_tmp"
_TMP.mkdir(parents=True, exist_ok=True)


def connect(memory_limit: str = "20GB", threads: int = 6) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET temp_directory='{_TMP}'")
    con.execute("SET max_temp_directory_size='250GB'")
    con.execute(f"PRAGMA threads={threads}")
    return con
