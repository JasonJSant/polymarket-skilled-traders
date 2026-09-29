"""News-reaction test (Section 4.3.1, Table 8).

Restricts to markets with clearly scheduled information releases (FOMC rate
decisions and corporate earnings) identified from the question text. For each,
the surprise is (outcome - price just before the announcement), and the test
checks whether each group's order imbalance shifts in the surprise direction:

    dOIB^g = alpha + beta^g * Surprise + e.

Uses the 50-bin panel: "before" = second-to-last bin, "after" = last bin. beta>0
means the group trades in the direction of the news. Reported per group with SE
clustered on event and month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyfixest as pf

import config as C
from src.build.time_bins import GROUP_KEYS

FOMC_KW = ["fed ", "fomc", "basis point", "rate hike", "rate cut", "powell",
           "interest rate", "rate decision", "bps"]
EARN_KW = ["earnings", "beat q", "beat estimates", "eps", "quarterly revenue",
           "beat quarterly"]


def _mask(q: pd.Series, kws) -> pd.Series:
    ql = q.str.lower().fillna("")
    m = pd.Series(False, index=q.index)
    for k in kws:
        m |= ql.str.contains(k, regex=False)
    return m


def _panel_news(kind: str) -> pd.DataFrame:
    panel = pd.read_parquet(C.PROC / "bins.parquet")
    mk = pd.read_parquet(C.RAW / "markets.parquet",
                         columns=["condition_id", "question"])
    kws = FOMC_KW if kind == "fomc" else EARN_KW
    mk = mk[_mask(mk["question"], kws)]
    p = panel[panel["market"].isin(set(mk["condition_id"]))].copy()
    if p.empty:
        return p
    # before = bin 48, after = bin 49
    last = p[p["bin"] == C.N_BINS - 1].set_index("market")
    prev = p[p["bin"] == C.N_BINS - 2].set_index("market")
    rows = last.join(prev, rsuffix="_prev", how="inner")
    rows["surprise"] = (rows["y0"] - rows["price_prev"]) * C.PP
    for k in GROUP_KEYS.values():
        rows[f"dOIB_{k}"] = rows[f"OIB_{k}"] - rows[f"OIB_{k}_prev"]
    return rows.reset_index()


def table8() -> pd.DataFrame:
    rows = []
    for kind, label in [("fomc", "FOMC"), ("earnings", "Earnings")]:
        df = _panel_news(kind)
        for g, k in GROUP_KEYS.items():
            if df.empty:
                rows.append({"panel": label, "group": g, "coef": np.nan,
                             "t": np.nan, "n": 0, "r2": np.nan})
                continue
            d = df[[f"dOIB_{k}", "surprise", "event_id", "month"]].dropna()
            if len(d) < 10 or d["surprise"].std() == 0:
                rows.append({"panel": label, "group": g, "coef": np.nan,
                             "t": np.nan, "n": len(d), "r2": np.nan})
                continue
            try:
                m = pf.feols(f"dOIB_{k} ~ surprise", data=d,
                             vcov={"CRV1": "event_id+month"})
                t = m.tidy()
                rows.append({"panel": label, "group": g,
                             "coef": t.loc["surprise", "Estimate"],
                             "t": t.loc["surprise", "t value"],
                             "n": int(m._N), "r2": float(m._r2)})
            except Exception:  # noqa: BLE001
                rows.append({"panel": label, "group": g, "coef": np.nan,
                             "t": np.nan, "n": len(d), "r2": np.nan})
    return pd.DataFrame(rows)


def build() -> None:
    t8 = table8()
    t8.to_parquet(C.PROC / "table8.parquet", index=False)
    for _, r in t8.iterrows():
        print(f"  {r['panel']:9s} {r['group']:14s} beta={r['coef']:8.2f} "
              f"t={r['t']:7.2f} n={r['n']}")


if __name__ == "__main__":
    build()
