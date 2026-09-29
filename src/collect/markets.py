"""Collect Polymarket market/event metadata from the GAMMA API.

Produces three tables in data/raw/:
  * markets.parquet    one row per binary market (conditionId), with lifespan,
                       volume, negRisk flag, event grouping, and per-token outcome.
  * events.parquet     unique events (the grouping unit for the skill test's
                       event-level sign randomization), with topic tags.
  * token_map.parquet  one row per outcome token: tokenId -> (market, event,
                       outcome index/label, resolved payoff Y in {0,1}).

Universe filter (paper): markets created after 2023-01-01 and resolved by
2025-12-31. Always pulls the complete universe via GAMMA keyset pagination
(the regular offset endpoint caps at offset 2500); metadata is small and needed
by every downstream stage. Trade collection selects which markets to pull
*trades* for separately (see src/collect/trades.py).

Usage:
    python -m src.collect.markets                 # full universe
    python -m src.collect.markets --max-pages 5   # limited page run
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import pandas as pd
from tqdm import tqdm

import config as C
from src.http_util import get_json, load_ckpt, save_ckpt


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _jload(s, default):
    if isinstance(s, list):
        return s
    if not s:
        return default
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return default


def _in_window(created: datetime | None, resolved: datetime | None) -> bool:
    """Created after SAMPLE_START and resolved by COLLECT_END.

    COLLECT_END (>= paper SAMPLE_END) extends the universe to the subgraph's data
    ceiling; the paper window is recovered later by subsetting on `resolved`.
    """
    if created is None or created < C.SAMPLE_START:
        return False
    if resolved is None or resolved > C.COLLECT_END:
        return False
    return True


def _resolution(outcome_prices: list) -> list[float] | None:
    """Map settlement prices to per-token payoffs in {0,1}; None if not clean."""
    try:
        vals = [float(x) for x in outcome_prices]
    except (ValueError, TypeError):
        return None
    if not vals:
        return None
    # A resolved binary/multi market has exactly one payoff == 1 (rest 0).
    ys = [1 if abs(v - 1.0) < 1e-6 else (0 if abs(v) < 1e-6 else -1) for v in vals]
    if -1 in ys or sum(ys) != 1:
        return None
    return [float(y) for y in ys]


def _row(m: dict) -> dict | None:
    created = _parse_dt(m.get("createdAt"))
    start = _parse_dt(m.get("startDate")) or created
    end = _parse_dt(m.get("endDate"))
    closed_time = _parse_dt(m.get("closedTime"))
    resolved = closed_time or end  # actual resolution time preferred
    if not _in_window(created, resolved):
        return None
    tokens = _jload(m.get("clobTokenIds"), [])
    outcomes = _jload(m.get("outcomes"), [])
    prices = _jload(m.get("outcomePrices"), [])
    if len(tokens) < 2 or len(tokens) != len(outcomes):
        return None
    ys = _resolution(prices)
    if ys is None or len(ys) != len(tokens):
        return None
    events = m.get("events") or []
    ev = events[0] if events else {}
    return {
        "market_id": str(m.get("id")),
        "condition_id": m.get("conditionId"),
        "question": m.get("question"),
        "slug": m.get("slug"),
        "category": m.get("category"),
        "neg_risk": bool(m.get("negRisk") or m.get("negRiskOther")),
        "created": created,
        "start": start,
        "end": end,
        "closed_time": closed_time,
        "resolved": resolved,
        "volume": float(m.get("volumeNum") or 0.0),
        "event_id": str(ev.get("id")) if ev else None,
        "event_slug": ev.get("slug") if ev else None,
        "event_title": ev.get("title") if ev else None,
        "event_ticker": ev.get("ticker") if ev else None,
        "tokens": [str(t) for t in tokens],
        "outcomes": [str(o) for o in outcomes],
        "Y": ys,
    }


def collect(max_pages: int | None) -> None:
    """Enumerate the full closed-market universe via gamma keyset pagination.

    gamma /markets caps `offset` at 2500 and ignores `limit`/`id_gt` on the
    closed view, so /markets/keyset is used with the opaque `after_cursor` token
    (100 rows/page). A server-side `end_date_max` prunes markets resolving past
    the collection horizon, and `_row` applies the created>=2023 lower bound.
    Resumable via a cursor checkpoint + partial-rows parquet.
    """
    partial = C.CKPT / "markets_partial.parquet"
    ck = load_ckpt("markets.json", {})
    cursor = ck.get("after_cursor")
    pages = ck.get("pages", 0)
    rows: list[dict] = (
        pd.read_parquet(partial).to_dict("records") if partial.exists() else []
    )
    end_max = C.COLLECT_END.strftime("%Y-%m-%dT%H:%M:%SZ")

    pbar = tqdm(desc="markets", unit="mkt", initial=len(rows))
    while True:
        params = {"limit": 500, "closed": "true", "order": "id",
                  "ascending": "true", "end_date_max": end_max}
        if cursor:
            params["after_cursor"] = cursor
        resp = get_json(f"{C.GAMMA}/markets/keyset", params)
        batch = resp.get("markets") or []
        cursor = resp.get("next_cursor")
        if not batch:
            break
        for m in batch:
            r = _row(m)
            if r:
                rows.append(r)
                pbar.update(1)
        pages += 1
        if pages % 50 == 0:
            pd.DataFrame(rows).to_parquet(partial, index=False)
            save_ckpt("markets.json", {"after_cursor": cursor, "pages": pages})
        if not cursor:
            break
        if max_pages and pages >= max_pages:
            break
    pbar.close()

    if not rows:
        raise SystemExit("No markets collected. Check filters/endpoint.")

    mdf = pd.DataFrame(rows)
    # Explode to token_map: one row per outcome token.
    tm = mdf.explode(["tokens", "outcomes", "Y"], ignore_index=True)
    tm = tm.rename(columns={"tokens": "token_id", "outcomes": "outcome", "Y": "y"})
    tm["outcome_index"] = tm.groupby("market_id").cumcount()
    tm["is_yes"] = tm["outcome"].str.strip().str.lower().eq("yes")
    token_map = tm[[
        "token_id", "condition_id", "market_id", "event_id",
        "outcome", "outcome_index", "is_yes", "y", "neg_risk",
    ]].copy()
    token_map["y"] = token_map["y"].astype(float)

    markets = mdf.drop(columns=["tokens", "outcomes", "Y"]).copy()

    events = (
        markets.dropna(subset=["event_id"])
        .groupby("event_id")
        .agg(
            event_slug=("event_slug", "first"),
            event_title=("event_title", "first"),
            event_ticker=("event_ticker", "first"),
            n_markets=("market_id", "nunique"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )

    markets.to_parquet(C.RAW / "markets.parquet", index=False)
    token_map.to_parquet(C.RAW / "token_map.parquet", index=False)
    events.to_parquet(C.RAW / "events.parquet", index=False)

    # Clean up resumable state on successful completion.
    (C.CKPT / "markets_partial.parquet").unlink(missing_ok=True)
    (C.CKPT / "markets.json").unlink(missing_ok=True)

    print(f"markets: {len(markets):,}  tokens: {len(token_map):,}  events: {len(events):,}")
    print(f"total volume: ${markets['volume'].sum()/1e9:.2f}B")
    print(f"neg-risk markets: {markets['neg_risk'].sum():,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pages", type=int, default=None,
                    help="stop after N keyset pages")
    a = ap.parse_args()
    collect(a.max_pages)
