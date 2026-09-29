"""Top-level data-collection launcher.

Runs the two collection stages in order; each is resumable:
  1. markets: GAMMA metadata for the full in-window universe (fast; skipped
              if markets.parquet already exists).
  2. trades: the universe of on-chain fills, streamed to sharded parquet.

A top-level supervisor restarts the trades stage if it crashes outright
(e.g. the machine slept), since all progress is checkpointed. Intended to be
launched under `caffeinate` so an idle Mac does not pause it:

    caffeinate -i .venv/bin/python -u run_collect.py --workers 4 > logs/trades.log 2>&1 &

Watch progress with:  tail -f logs/trades.log
"""
from __future__ import annotations

import argparse
import time

import config as C
from src.collect import markets as markets_mod
from src.collect import trades as trades_mod


def main(workers: int) -> None:
    if not (C.RAW / "markets.parquet").exists():
        print("[stage] markets metadata ...", flush=True)
        markets_mod.collect(max_pages=None)
    else:
        print("[stage] markets.parquet present, skipping metadata", flush=True)

    print("[stage] trades sweep ...", flush=True)
    while True:
        try:
            trades_mod.run_all(workers=workers,
                               log=lambda m: print(m, flush=True))
            break
        except Exception as e:  # noqa: BLE001
            print(f"[supervisor] trades crashed: {str(e)[:200]}; "
                  f"restarting from checkpoints in 60s", flush=True)
            time.sleep(60)
    print("[done] collection complete", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=C.SUBGRAPH_WORKERS)
    main(ap.parse_args().workers)
