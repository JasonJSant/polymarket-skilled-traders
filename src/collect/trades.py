"""Collector for the universe of Polymarket trades.

  * STREAMING: never hold more than a bounded buffer in RAM. The buffer flushes
    to a sharded parquet part file every FLUSH_ROWS rows, then is cleared.
  * SHARDED: one directory per calendar month (data/raw/trades/YYYY-MM/), with
    numbered part files. Months are independent units of work.
  * RESUMABLE: each month has its own checkpoint {cursor, part, complete}. The
    sweep cursors on the (frozen, immutable) event `id` with timestamp bounds, so
    re-running continues exactly where it stopped with no duplication.
  * RETRYING: run_all() retries any month that errors out, sleeping between
    passes, until every month is complete. A transient subgraph/network outage
    therefore does not abort the run.

Each on-chain OrderFilledEvent is decoded into TWO account-trades (the maker's
side and the taker's side), each tagged is_maker, giving direction, price, size,
account, and limit-order status: everything the skill test and MM rule need.
Only fills whose traded token is in the market universe (token_map) are kept.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd

import config as C
from src.http_util import gql, load_ckpt, save_ckpt

FLUSH_ROWS = 200_000            # ~100k fills buffered before a disk flush
_QUERY = """query($lo:BigInt!,$hi:BigInt!,$cur:String!){
  orderFilledEvents(first:1000, orderBy:id, orderDirection:asc,
    where:{timestamp_gte:$lo, timestamp_lte:$hi, id_gt:$cur}){
    id timestamp maker taker makerAssetId takerAssetId
    makerAmountFilled takerAmountFilled
  }
}"""


def months() -> list[str]:
    """All YYYY-MM strings in the collection window, inclusive.

    Spans SAMPLE_START to COLLECT_END (COLLECT_END >= SAMPLE_END extends
    past the paper's 2025-12 cutoff to the subgraph's data ceiling).
    """
    out = []
    y, m = C.SAMPLE_START.year, C.SAMPLE_START.month
    while (y, m) <= (C.COLLECT_END.year, C.COLLECT_END.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _month_bounds(ym: str) -> tuple[int, int]:
    y, m = map(int, ym.split("-"))
    lo = datetime(y, m, 1, tzinfo=timezone.utc)
    hi = datetime(y + (m == 12), (m % 12) + 1, 1, tzinfo=timezone.utc)
    # clamp to the collection window and use an inclusive lower / exclusive upper bound
    lo_ts = max(int(lo.timestamp()), C.SAMPLE_START_TS)
    hi_ts = min(int(hi.timestamp()) - 1, C.COLLECT_END_TS)
    return lo_ts, hi_ts


def _load_token_meta() -> dict[str, tuple]:
    tm = pd.read_parquet(C.RAW / "token_map.parquet",
                         columns=["token_id", "condition_id", "event_id", "y", "is_yes"])
    return {
        r.token_id: (r.condition_id, r.event_id, float(r.y), bool(r.is_yes))
        for r in tm.itertuples(index=False)
    }


def _decode(e: dict, token_meta: dict) -> list[dict] | None:
    """Turn one OrderFilledEvent into two account-trades, or None if off-universe.

    Convention: the maker gives makerAsset and receives takerAsset. assetId "0"
    is USDC collateral; the other side is the outcome token.
    """
    mzero = e["makerAssetId"] == "0"
    tzero = e["takerAssetId"] == "0"
    if mzero == tzero:            # both/neither collateral -> not a plain trade
        return None
    if mzero:
        token = e["takerAssetId"]
        usdc = int(e["makerAmountFilled"])
        shares = int(e["takerAmountFilled"])
        maker_buys = True         # maker pays USDC, receives token
    else:
        token = e["makerAssetId"]
        usdc = int(e["takerAmountFilled"])
        shares = int(e["makerAmountFilled"])
        maker_buys = False        # maker delivers token, receives USDC
    meta = token_meta.get(token)
    if meta is None or shares <= 0:
        return None
    condition_id, event_id, _y, _is_yes = meta
    price = usdc / shares                 # 1e6 units cancel
    size = shares / C.SCALE               # shares in whole units
    notional = usdc / C.SCALE             # USDC in dollars
    ts = int(e["timestamp"])
    fill_id = e["id"]
    maker = {
        "fill_id": fill_id, "ts": ts, "account": e["maker"], "token_id": token,
        "condition_id": condition_id, "event_id": event_id, "is_maker": True,
        "d": 1 if maker_buys else -1, "size": size, "price": price, "notional": notional,
    }
    taker = {
        "fill_id": fill_id, "ts": ts, "account": e["taker"], "token_id": token,
        "condition_id": condition_id, "event_id": event_id, "is_maker": False,
        "d": -1 if maker_buys else 1, "size": size, "price": price, "notional": notional,
    }
    return [maker, taker]


def _flush(buf: list[dict], ym: str, part: int) -> int:
    if not buf:
        return part
    d = C.TRADES_DIR / ym
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"part-{part:05d}.parquet.tmp"
    final = d / f"part-{part:05d}.parquet"
    pd.DataFrame(buf).to_parquet(tmp, index=False)
    tmp.rename(final)            # atomic-ish: appears only when fully written
    return part + 1


def sweep_month(ym: str, token_meta: dict, log=print) -> None:
    """Fully collect one month, resuming from its checkpoint. Idempotent."""
    ckname = f"trades_{ym}.json"
    ck = load_ckpt(ckname, {"cursor": "", "part": 0, "fills": 0, "rows": 0, "complete": False})
    if ck.get("complete"):
        return
    lo, hi = _month_bounds(ym)
    cursor, part = ck["cursor"], ck["part"]
    fills, rows_written = ck["fills"], ck["rows"]
    buf: list[dict] = []
    t0 = time.time()
    while True:
        batch = gql(C.SUBGRAPH, _QUERY, {"lo": str(lo), "hi": str(hi), "cur": cursor})
        evs = batch["orderFilledEvents"]
        if not evs:
            break
        for e in evs:
            dec = _decode(e, token_meta)
            if dec:
                buf.extend(dec)
                fills += 1
        cursor = evs[-1]["id"]
        if len(buf) >= FLUSH_ROWS:
            part = _flush(buf, ym, part)
            rows_written += len(buf)
            buf = []
            save_ckpt(ckname, {"cursor": cursor, "part": part, "fills": fills,
                               "rows": rows_written, "complete": False})
            log(f"[{ym}] {fills:,} fills, {rows_written:,} rows, {time.time()-t0:.0f}s")
        if len(evs) < 1000:
            break
    # final flush + mark complete
    part = _flush(buf, ym, part)
    rows_written += len(buf)
    save_ckpt(ckname, {"cursor": cursor, "part": part, "fills": fills,
                       "rows": rows_written, "complete": True})
    log(f"[{ym}] DONE {fills:,} fills, {rows_written:,} rows in {time.time()-t0:.0f}s")


def _overall_status() -> tuple[int, int, int]:
    """(#complete months, total months, total fills so far) from checkpoints."""
    all_months = months()
    done = fills = 0
    for ym in all_months:
        ck = load_ckpt(f"trades_{ym}.json", {})
        fills += ck.get("fills", 0)
        done += 1 if ck.get("complete") else 0
    return done, len(all_months), fills


def run_all(workers: int = C.SUBGRAPH_WORKERS, log=print, max_attempts: int = 40) -> None:
    """Collect every month, retrying failures until all complete.

    Uses escalating backoff between retry passes so a subgraph throttle is
    waited out rather than hammered. Each month resumes from its checkpoint, so
    no work is repeated. Concurrency defaults to 2 workers.
    """
    token_meta = _load_token_meta()
    log(f"loaded {len(token_meta):,} universe tokens; workers={workers}")
    t_start = time.time()
    todo = [ym for ym in months()
            if not load_ckpt(f"trades_{ym}.json", {}).get("complete")]
    attempt = 0
    while todo and attempt < max_attempts:
        attempt += 1
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(sweep_month, ym, token_meta, log): ym for ym in todo}
            for fut in as_completed(futs):
                ym = futs[fut]
                try:
                    fut.result()
                except Exception as e:  # noqa: BLE001
                    log(f"[{ym}] ERROR (attempt {attempt}): {str(e)[:160]}")
                done, total, fills = _overall_status()
                el = time.time() - t_start
                log(f"[progress] {done}/{total} months | {fills:,} fills | "
                    f"{el/60:.1f}min elapsed | {fills/el if el else 0:.0f} fills/s")
        todo = [ym for ym in months()
                if not load_ckpt(f"trades_{ym}.json", {}).get("complete")]
        if todo:
            backoff = min(900, 120 * attempt)   # escalate 120s, 240s, ... cap 15min
            log(f"retrying {len(todo)} months after {backoff}s (attempt {attempt}): {todo}")
            time.sleep(backoff)
    done, total, fills = _overall_status()
    if todo:
        log(f"STOPPED after {attempt} attempts with {len(todo)} months incomplete: {todo}")
    else:
        log(f"ALL MONTHS COMPLETE: {fills:,} fills in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--month", default=None, help="collect a single month (debug)")
    a = ap.parse_args()
    if a.month:
        sweep_month(a.month, _load_token_meta())
    else:
        run_all(a.workers)
