"""Insider-vs-skill price discovery (Tables 6, E.6, E.7).

Table 6 re-estimates the price-discovery regressions for the skilled group
separately on markets the LLM flags as insider-prone vs not, showing the skilled
drive price discovery even where insider information is unavailable. E.6/E.7
report all groups within each market type.

Reuses the 50-bin test panel (data/processed/bins.parquet) and the LLM labels
(data/processed/insider_labels.parquet).
"""
from __future__ import annotations

import pandas as pd

import config as C
from src.build.time_bins import GROUP_KEYS
from src.pricediscovery.regressions import _reg


def _label_map() -> pd.DataFrame:
    lab = pd.read_parquet(C.PROC / "insider_labels.parquet")
    lab["prone"] = lab["label"].isin(["yes", "unclear"])
    return lab[["condition_id", "prone"]].drop_duplicates("condition_id")


def build() -> None:
    if not (C.PROC / "insider_labels.parquet").exists():
        raise SystemExit("Run src.insider.llm_classify first.")
    panel = pd.read_parquet(C.PROC / "bins.parquet")
    lab = _label_map().rename(columns={"condition_id": "market"})
    panel = panel.merge(lab, on="market", how="left")
    panel["prone"] = panel["prone"].fillna(False)
    panel["dp_next_pp"] = panel["dp_next"] * C.PP
    panel["outcome_pp"] = panel["y0"] * C.PP
    panel["PD_pp"] = panel["PD"] * C.PP

    rows = []
    for prone, mtype in [(False, "not-prone"), (True, "prone")]:
        sub = panel[panel["prone"] == prone]
        for g, k in GROUP_KEYS.items():
            r_dp = _reg(sub, "dp_next_pp", f"OIB_{k}")
            r_out = _reg(sub, "outcome_pp", f"OIB_{k}")
            r_pd = _reg(sub, "PD_pp", f"VOL_{k}")
            rows.append({
                "market_type": mtype, "group": g,
                "oib_dp": r_dp["coef"], "t_dp": r_dp["t"],
                "oib_out": r_out["coef"], "t_out": r_out["t"],
                "vol_pd": r_pd["coef"], "t_pd": r_pd["t"], "n": r_dp["n"],
            })
    out = pd.DataFrame(rows)
    out.to_parquet(C.PROC / "table6_e6_e7.parquet", index=False)
    print("Table 6 / E.6 / E.7 (skilled highlighted):")
    for _, r in out[out.group == "Skilled"].iterrows():
        print(f"  {r['market_type']:10s} Skilled: OIB->dp {r['oib_dp']:.2f} "
              f"OIB->out {r['oib_out']:.2f} VOL->PD {r['vol_pd']:.2f}")


if __name__ == "__main__":
    build()
