"""Shared HTTP helpers: SSL fix, retries with backoff, GraphQL, JSON checkpoints.

The trusted TLS root for these APIs lives in the macOS keychain (used by curl)
but not in certifi, so the OS trust store is injected into Python's ssl via
`truststore`. All network access goes through these helpers for consistent
retry and rate-limit behaviour.
"""
from __future__ import annotations

import json
import time
from typing import Any

import truststore

truststore.inject_into_ssl()

import requests  # noqa: E402

import config as C  # noqa: E402

_session = requests.Session()
_session.headers.update({"User-Agent": "pm-repro/1.0"})


def get_json(url: str, params: dict | None = None, *, timeout: int | None = None) -> Any:
    """GET returning parsed JSON, with retry/backoff on transient failures."""
    last = None
    for attempt in range(C.HTTP_RETRIES):
        try:
            r = _session.get(url, params=params, timeout=timeout or C.HTTP_TIMEOUT)
            if r.status_code == 429:
                time.sleep(C.HTTP_BACKOFF ** attempt * 2)
                continue
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:  # noqa: PERF203
            last = e
            time.sleep(C.HTTP_BACKOFF ** attempt)
    raise RuntimeError(f"GET failed after {C.HTTP_RETRIES} tries: {url} :: {last}")


def gql(endpoint: str, query: str, variables: dict | None = None) -> dict:
    """POST a GraphQL query, returning the `data` field.

    Rate-limit (429) and connection errors back off up to about 2 minutes, so a
    transient throttle from the subgraph does not abort a month's sweep.
    """
    last = None
    for attempt in range(C.HTTP_RETRIES):
        try:
            r = _session.post(
                endpoint,
                json={"query": query, "variables": variables or {}},
                timeout=C.HTTP_TIMEOUT,
            )
            if r.status_code == 429:
                time.sleep(min(120, C.HTTP_BACKOFF ** attempt * 3))
                continue
            r.raise_for_status()
            payload = r.json()
            if "errors" in payload and payload["errors"]:
                raise ValueError(str(payload["errors"])[:300])
            return payload["data"]
        except (requests.RequestException, ValueError, KeyError) as e:  # noqa: PERF203
            last = e
            # connection errors: back off hard (subgraph may be throttling)
            time.sleep(min(120, C.HTTP_BACKOFF ** attempt))
    raise RuntimeError(f"GraphQL failed after {C.HTTP_RETRIES} tries: {last}")


def load_ckpt(name: str, default: Any = None) -> Any:
    p = C.CKPT / name
    if p.exists():
        return json.loads(p.read_text())
    return default


def save_ckpt(name: str, obj: Any) -> None:
    (C.CKPT / name).write_text(json.dumps(obj))
