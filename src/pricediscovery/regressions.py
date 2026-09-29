"""Price-discovery and accuracy regressions (Tables 4 and 5).

Table 4: regress next-bin price change and the final outcome on each group's
         order imbalance (OIB).
Table 5: regress the change in absolute pricing error (PD) on each group's
         volume share (VOL), full sample and by lifespan window.

Dependent variables are in percentage points (x100); OIB/VOL are raw. Controls
are lagged price, lagged price change, lagged log volume, and log duration.
Standard errors are two-way clustered on event and calendar month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyfixest as pf

import config as C
from src.build.time_bins import GROUP_KEYS

CONTROLS = ["price_lag", "dpchg_lag", "logvol_lag", "duration"]


def _reg(df: pd.DataFrame, dep: str, x: str) -> dict:
    d = df[[dep, x, *CONTROLS, "event_id", "month"]].dropna()
    if len(d) < 100 or d[x].std() == 0:
        return {"coef": np.nan, "t": np.nan, "n": len(d), "r2": np.nan}
    fml = f"{dep} ~ {x} + " + " + ".join(CONTROLS)
    m = pf.feols(fml, data=d, vcov={"CRV1": "event_id+month"})
    t = m.tidy()
    return {"coef": t.loc[x, "Estimate"], "t": t.loc[x, "t value"],
            "n": int(m._N), "r2": float(m._r2)}


def table4(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["dp_next_pp"] = df["dp_next"] * C.PP
    df["outcome_pp"] = df["y0"] * C.PP
    rows = []
    for dep, label in [("dp_next_pp", "Delta p_{t+1}"), ("outcome_pp", "outcome")]:
        for g, k in GROUP_KEYS.items():
            r = _reg(df, dep, f"OIB_{k}")
            rows.append({"panel": label, "group": g, **r})
    return pd.DataFrame(rows)


def table5(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["PD_pp"] = df["PD"] * C.PP
    rows = []
    # Panel A: full sample
    for g, k in GROUP_KEYS.items():
        r = _reg(df, "PD_pp", f"VOL_{k}")
        rows.append({"window": "Full", "group": g, **r})
    # Panel B: by lifespan window (bin/50 in [lo,hi))
    df["frac"] = df["bin"] / C.N_BINS
    for lo, hi in C.LIFESPAN_WINDOWS:
        sub = df[(df["frac"] >= lo) & (df["frac"] < hi + (hi == 1.0) * 1e-9)]
        for g, k in GROUP_KEYS.items():
            r = _reg(sub, "PD_pp", f"VOL_{k}")
            rows.append({"window": f"{int(lo*100)}-{int(hi*100)}%", "group": g, **r})
    return pd.DataFrame(rows)


def build() -> None:
    panel = pd.read_parquet(C.PROC / "bins.parquet")
    t4 = table4(panel)
    t5 = table5(panel)
    t4.to_parquet(C.PROC / "table4.parquet", index=False)
    t5.to_parquet(C.PROC / "table5.parquet", index=False)

    print("=== Table 4 (OIB) ===")
    for _, r in t4.iterrows():
        print(f"  {r['panel']:16s} {r['group']:14s} beta={r['coef']:8.3f} "
              f"t={r['t']:7.2f} n={r['n']:,}")
    print("\n=== Table 5 Panel A (VOL, full) ===")
    for _, r in t5[t5.window == "Full"].iterrows():
        print(f"  {r['group']:14s} beta={r['coef']:8.3f} t={r['t']:7.2f} n={r['n']:,}")
    print("Table 5 skilled by window:")
    for _, r in t5[(t5.group == "Skilled")].iterrows():
        print(f"  {r['window']:8s} beta={r['coef']:8.3f} t={r['t']:7.2f}")


if __name__ == "__main__":
    build()
