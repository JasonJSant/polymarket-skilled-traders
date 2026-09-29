"""Classify every account into one of five groups (Section 3.1).

Order matters: the market-maker rule is behavioural and is applied FIRST, then
the remaining accounts are sorted by their sign-randomization p-value.

  Market Maker : n_markets >= 100 AND maker_volume_share >= 0.70
                 AND both_sides_share >= 0.70
  Skilled      : E >= filter AND p < 0.05
  Anti-skilled : E >= filter AND p > 0.95
  Lucky        : otherwise, PnL > 0
  Unlucky      : otherwise, PnL <= 0

The >=10-event filter is the benchmark; classify() accepts any filter so Table 1
can be reproduced for >=10, >=4, and none. Benjamini-Hochberg FDR control is
provided for Table 1 Panel B.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def market_maker_mask(feat: pd.DataFrame) -> pd.Series:
    return (
        (feat["n_markets"] >= C.MM_MIN_MARKETS)
        & (feat["maker_volume_share"] >= C.MM_LIMIT_SHARE)
        & (feat["both_sides_share"] >= C.MM_BOTHSIDES_SHARE)
    )


def _bh_reject(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg: return boolean reject mask at level alpha."""
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    below = ranked <= thresh
    k = np.max(np.where(below)[0]) + 1 if below.any() else 0
    reject = np.zeros(n, dtype=bool)
    if k > 0:
        reject[order[:k]] = True
    return reject


def classify(pvals: pd.DataFrame, feat: pd.DataFrame,
             event_filter: int | None = C.BENCHMARK_EVENT_FILTER,
             bh: bool = False) -> pd.DataFrame:
    """Return per-account classification. `pvals` has account,E,pnl,pvalue;
    `feat` has account,n_markets,maker_volume_share,both_sides_share,volume."""
    df = pvals.merge(feat, on="account", how="left", suffixes=("", "_f"))
    df["pnl"] = df["pnl"].fillna(df.get("pnl_f"))
    df["n_markets"] = df["n_markets"].fillna(0)
    df["maker_volume_share"] = df["maker_volume_share"].fillna(0)
    df["both_sides_share"] = df["both_sides_share"].fillna(0)

    is_mm = market_maker_mask(df)
    eligible = df["E"] >= (event_filter or 1)

    if bh:
        # Two one-sided tails, BH-controlled separately (paper Table 1 Panel B).
        skilled = np.zeros(len(df), dtype=bool)
        anti = np.zeros(len(df), dtype=bool)
        elig_idx = np.where(eligible & ~is_mm)[0]
        p = df["pvalue"].to_numpy()
        skilled[elig_idx] = _bh_reject(p[elig_idx], 0.05)
        anti[elig_idx] = _bh_reject(1.0 - p[elig_idx], 0.05)
    else:
        skilled = eligible & ~is_mm & (df["pvalue"] < C.SKILL_P)
        anti = eligible & ~is_mm & (df["pvalue"] > C.ANTISKILL_P)

    group = np.full(len(df), "Unlucky", dtype=object)
    group[df["pnl"] > 0] = "Lucky"
    group[np.asarray(skilled)] = "Skilled"
    group[np.asarray(anti)] = "Anti-skilled"
    group[is_mm.to_numpy()] = "Market Maker"
    df["group"] = group
    return df[["account", "E", "pnl", "pvalue", "n_markets", "volume", "group"]]


def build() -> None:
    pvals = pd.read_parquet(C.PROC / "account_pvalues.parquet")
    feat = pd.read_parquet(C.PROC / "account_features.parquet")
    cls = classify(pvals, feat, event_filter=C.BENCHMARK_EVENT_FILTER, bh=False)
    cls.to_parquet(C.PROC / "classification.parquet", index=False)

    n = len(cls)
    counts = cls["group"].value_counts()
    print(f"total accounts classified: {n:,}")
    for g in C.GROUPS:
        c = int(counts.get(g, 0))
        print(f"  {g:14s}: {c:>9,}  ({c/n*100:5.2f}%)")


if __name__ == "__main__":
    build()
