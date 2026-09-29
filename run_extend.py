"""Extend the Polymarket dataset into 2026 (through the subgraph's 2026-04-28 ceiling).

Same methodology as the original run (streaming, sharded, checkpointed,
resumable), but processed ONE MONTH AT A TIME (sweep, then operator-clean, then
reclaim disk) so RAM and disk stay bounded even as monthly volume grows (Dec 2025
alone was 68M fills / 12GB). Safe to kill and relaunch: every step resumes from
its checkpoint.

Pipeline (all idempotent / resumable):
  0. reproduction universe already snapshotted to data/raw/_repro_2023_2025/.
  1. ensure the extended universe exists (markets.collect through COLLECT_END);
     re-run only if markets.parquet is still paper-scoped.
  2. reset the three re-swept months (2025-10/11/12) ONCE (guarded by a marker):
     delete their raw + clean shards + checkpoint so they re-collect against the
     new superset universe.
  3. for each target month in chronological order:
       a. sweep_month (resumable; retries transient subgraph errors)
       b. operator-clean the month (clean_trades.clean_month; idempotent)
       c. if free disk < DISK_MIN_GB, prune that month's RAW shard (clean is the
          analysis input; raw is re-downloadable immutable on-chain data)
     A live progress line (fills, rows, elapsed, free-GB) is emitted throughout.

Launch (survives sleep/logout):
    caffeinate -i .venv/bin/python -u run_extend.py > logs/extend.log 2>&1 &
Watch:  tail -f logs/extend.log
"""
from __future__ import annotations

import shutil
import time
from datetime import datetime, timezone

import pandas as pd

import config as C
from src.build import clean_trades
from src.collect import markets as markets_mod
from src.collect import trades as trades_mod
from src.duckutil import connect as _duck_connect
from src.http_util import load_ckpt, save_ckpt

# Months to (re)collect: re-sweep the 2025 boundary tail (so markets resolving in
# 2026 keep their late-2025 trades) plus the new 2026 months.
RESWEEP = ["2025-10", "2025-11", "2025-12"]
NEW = ["2026-01", "2026-02", "2026-03", "2026-04"]
TARGETS = RESWEEP + NEW

DISK_MIN_GB = 60.0          # prune cleaned months' raw shards below this headroom
RESET_MARKER = "extend_reset_v1.json"
MONTH_MAX_ATTEMPTS = 40     # per-month retry passes (matches run_all)


def _log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z | {msg}", flush=True)


def _free_gb() -> float:
    return shutil.disk_usage(C.DATA).free / 1e9


def ensure_universe() -> None:
    """Rebuild the superset universe unless markets.parquet already extends past
    the paper window (resolved > SAMPLE_END present)."""
    mk_path = C.RAW / "markets.parquet"
    if mk_path.exists():
        mx = pd.read_parquet(mk_path, columns=["resolved"])["resolved"].max()
        if pd.notna(mx) and pd.Timestamp(mx).tz_convert("UTC") > pd.Timestamp(C.SAMPLE_END):
            _log(f"universe already extended (max resolved {pd.Timestamp(mx).date()}); skip rebuild")
            return
    _log("rebuilding extended universe via GAMMA (markets.collect) ...")
    markets_mod.collect(max_pages=None)
    _log("extended universe rebuilt")


def reset_resweep_once() -> None:
    """Delete the 2025 boundary months so they re-collect against the new universe.
    Guarded by a marker so a relaunch never wipes in-progress work."""
    if load_ckpt(RESET_MARKER, {}).get("done"):
        return
    for ym in RESWEEP:
        shutil.rmtree(C.TRADES_DIR / ym, ignore_errors=True)
        shutil.rmtree(C.CLEAN_TRADES_DIR / ym, ignore_errors=True)
        (C.CKPT / f"trades_{ym}.json").unlink(missing_ok=True)
        _log(f"reset {ym} (raw + clean + checkpoint) for re-sweep")
    save_ckpt(RESET_MARKER, {"done": True, "months": RESWEEP})


def _sweep_with_retry(ym: str, token_meta: dict) -> None:
    """Fully collect one month, retrying transient subgraph/network errors."""
    for attempt in range(1, MONTH_MAX_ATTEMPTS + 1):
        try:
            trades_mod.sweep_month(ym, token_meta, log=_log)
            return
        except Exception as e:  # noqa: BLE001
            back = min(900, 120 * attempt)
            _log(f"[{ym}] sweep error (attempt {attempt}): {str(e)[:160]}; retry in {back}s")
            time.sleep(back)
    raise RuntimeError(f"{ym}: exhausted {MONTH_MAX_ATTEMPTS} sweep attempts")


def _clean_month(ym: str) -> None:
    """Operator-clean one month (idempotent: clean_month skips if already done)."""
    con = _duck_connect()
    con.execute("PRAGMA threads=6")
    try:
        res = clean_trades.clean_month(con, ym)
    finally:
        con.close()
    if res is None:
        _log(f"[{ym}] clean: skipped (no source or already clean)")
    else:
        raw_n, clean_n = res
        pct = 100 * clean_n / raw_n if raw_n else 0
        _log(f"[{ym}] clean: {raw_n:,} -> {clean_n:,} rows ({pct:.0f}% kept)")


def _prune_raw_if_low(cleaned: list[str]) -> None:
    """When disk headroom is low, drop raw shards of already-cleaned months."""
    for ym in cleaned:
        if _free_gb() >= DISK_MIN_GB:
            return
        raw_dir = C.TRADES_DIR / ym
        clean_dir = C.CLEAN_TRADES_DIR / ym
        if raw_dir.exists() and clean_dir.exists() and any(clean_dir.glob("*.parquet")):
            shutil.rmtree(raw_dir, ignore_errors=True)
            _log(f"[{ym}] pruned raw shard to reclaim disk (clean retained); "
                 f"free now {_free_gb():.0f} GB")


def main() -> None:
    t0 = time.time()
    _log(f"=== extend run start | targets: {TARGETS} | free {_free_gb():.0f} GB ===")

    ensure_universe()
    reset_resweep_once()

    token_meta = trades_mod._load_token_meta()
    _log(f"loaded {len(token_meta):,} universe tokens (extended)")

    cleaned: list[str] = []
    for i, ym in enumerate(TARGETS, 1):
        _log(f"--- month {i}/{len(TARGETS)}: {ym} | free {_free_gb():.0f} GB ---")
        _sweep_with_retry(ym, token_meta)
        _clean_month(ym)
        cleaned.append(ym)
        _prune_raw_if_low(cleaned)
        _log(f"[{ym}] done | {i}/{len(TARGETS)} months | {(time.time()-t0)/60:.1f} min | "
             f"free {_free_gb():.0f} GB")

    _log(f"=== extend run COMPLETE | {len(TARGETS)} months | "
         f"{(time.time()-t0)/60:.1f} min | free {_free_gb():.0f} GB ===")


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except Exception as e:  # noqa: BLE001
            _log(f"[supervisor] run crashed: {str(e)[:200]}; restarting from checkpoints in 60s")
            time.sleep(60)
