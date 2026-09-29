"""Regenerate the paper's tables from the processed data (CSV + LaTeX).

Covers Table 1 (skill shares across filters x BH), Table 2 (activity/performance
by group), Table 3 (persistence), Tables 4-5 (price discovery/accuracy), and the
appendix summary tables E.8 (market stats), E.10 (PnL distribution), E.11 (PnL
concentration). Each table is written to outputs/tables/<name>.{csv,tex}.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

import config as C
from src.duckutil import connect as _safe_connect
from src.skill.classify import classify

TRADES_GLOB = C.TRADES_GLOB


def _save(df: pd.DataFrame, name: str, float_fmt: str = "%.3f") -> None:
    df.to_csv(C.TABLES / f"{name}.csv", index=False)
    try:
        df.to_latex(C.TABLES / f"{name}.tex", index=False, float_format=float_fmt)
    except Exception:  # noqa: BLE001
        pass
    print(f"  wrote {name} ({len(df)} rows)")


# Table 1
def table1() -> pd.DataFrame:
    pv = pd.read_parquet(C.PROC / "account_pvalues.parquet")
    ft = pd.read_parquet(C.PROC / "account_features.parquet")
    n_all = len(pv)
    rows = []
    for panel, bh in [("A: No Adjustment", False), ("B: Benjamini-Hochberg", True)]:
        for filt in C.EVENT_FILTERS:
            cls = classify(pv, ft, event_filter=filt, bh=bh)
            filt_n = (pv["E"] >= (filt or 1)).sum()
            for grp in ("Skilled", "Anti-skilled"):
                k = int((cls["group"] == grp).sum())
                rows.append({
                    "panel": panel, "filter": filt or "None", "group": grp,
                    "number": k,
                    "pct_filtered": round(100 * k / filt_n, 2) if filt_n else np.nan,
                    "pct_all": round(100 * k / n_all, 2),
                })
    return pd.DataFrame(rows)


# Table 2
def _concentration(aep: pd.DataFrame) -> pd.DataFrame:
    """Per-account: top-1 event profit share and profit HHI (positive profits)."""
    a = aep.copy()
    a["pos"] = a["s_e"].clip(lower=0)
    g = a.groupby("account")
    tot = g["pos"].sum().rename("pos_tot")
    top1 = g["pos"].max().rename("pos_top1")
    hhi = a.assign(tot=a["account"].map(tot))
    hhi["sh"] = np.where(hhi["tot"] > 0, hhi["pos"] / hhi["tot"], 0)
    hhi_v = hhi.groupby("account").apply(
        lambda d: float((d["sh"] ** 2).sum()), include_groups=False).rename("hhi")
    out = pd.concat([tot, top1, hhi_v], axis=1).reset_index()
    out["top1_share"] = np.where(out["pos_tot"] > 0, out["pos_top1"] / out["pos_tot"], np.nan)
    return out[["account", "top1_share", "hhi"]]


def table2() -> tuple[pd.DataFrame, pd.DataFrame]:
    cls = pd.read_parquet(C.PROC / "classification.parquet")
    aep = pd.read_parquet(C.PROC / "account_event_pnl.parquet")
    conc = _concentration(aep)
    cls = cls.merge(conc, on="account", how="left")

    tot_gain = cls.loc[cls.pnl > 0, "pnl"].sum()
    tot_loss = -cls.loc[cls.pnl < 0, "pnl"].sum()
    tot_vol = cls["volume"].sum()
    n = len(cls)

    panelA, panelB = [], []
    for g in C.GROUPS:
        d = cls[cls.group == g]
        pnl = d["pnl"].sum()
        vol = d["volume"].sum()
        panelA.append({
            "group": g, "n_thousand": round(len(d) / 1e3, 2),
            "acct_share_pct": round(100 * len(d) / n, 2),
            "pnl_million": round(pnl / 1e6, 2),
            "pnl_share_pct": round(100 * (pnl / tot_gain if pnl >= 0 else pnl / tot_loss), 2),
            "vol_billion": round(vol / 1e9, 2),
            "vol_share_pct": round(100 * vol / tot_vol, 2),
        })
        panelB.append({
            "group": g,
            "n_events": round(d["E"].mean(), 2),
            "n_markets": round(d["n_markets"].mean(), 2),
            "pnl": round(d["pnl"].mean(), 2),
            "volume": round(d["volume"].mean(), 2),
            "top1_event_share_pct": round(100 * d["top1_share"].mean(), 2),
            "profit_hhi": round(d["hhi"].mean(), 2),
        })
    return pd.DataFrame(panelA), pd.DataFrame(panelB)


# Table 3
def table3() -> tuple[pd.DataFrame, pd.DataFrame]:
    a = pd.read_parquet(C.PROC / "persistence_panelA.parquet").reset_index(names="stat")
    b = pd.read_parquet(C.PROC / "persistence_panelB.parquet").reset_index(names="train")
    return a, b


# Tables 4-5
def tables45() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (pd.read_parquet(C.PROC / "table4.parquet"),
            pd.read_parquet(C.PROC / "table5.parquet"))


# Table E.8
def tableE8() -> pd.DataFrame:
    con = _safe_connect()
    stats = con.execute(f"""
        WITH m AS (
            SELECT t.condition_id market, COUNT(DISTINCT t.account) n_acct,
                   SUM(t.notional)/2 vol,
                   date_diff('day', ANY_VALUE(mk.start), ANY_VALUE(mk.resolved)) dur
            FROM read_parquet('{TRADES_GLOB}') t
            JOIN read_parquet('{C.RAW/'markets.parquet'}') mk ON mk.condition_id=t.condition_id
            GROUP BY t.condition_id
        )
        SELECT n_acct, vol, vol/NULLIF(n_acct,0) AS vol_per_acct, dur FROM m
    """).fetchdf()
    con.close()
    pcts = [0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99]
    rows = {"Obs": [len(stats)] * 4, "Mean": stats.mean().round(2).tolist(),
            "Std": stats.std().round(2).tolist()}
    df = pd.DataFrame({
        "stat": ["Number of Accounts", "Total Volume ($)", "Avg Vol/Account ($)", "Duration (days)"],
        "mean": stats.mean().round(2).values, "std": stats.std().round(2).values,
        **{f"p{int(p*100)}": stats.quantile(p).round(2).values for p in pcts},
        "max": stats.max().round(2).values,
    })
    return df


# Table E.10
def tableE10() -> pd.DataFrame:
    cls = pd.read_parquet(C.PROC / "classification.parquet")
    pcts = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    rows = []
    for g in C.GROUPS:
        d = cls.loc[cls.group == g, "pnl"]
        r = {"group": g, "obs": len(d), "mean": round(d.mean(), 2), "std": round(d.std(), 2)}
        for p in pcts:
            r[f"p{int(p*100)}"] = round(d.quantile(p), 2)
        rows.append(r)
    return pd.DataFrame(rows)


# Table E.11
def tableE11() -> pd.DataFrame:
    cls = pd.read_parquet(C.PROC / "classification.parquet")[["account", "group"]]
    aep = pd.read_parquet(C.PROC / "account_event_pnl.parquet")
    a = aep.merge(cls, on="account", how="left")
    a["pos"] = a["s_e"].clip(lower=0)
    rows = []
    for g in C.GROUPS:
        d = a[a.group == g]
        per = []
        for _, grp in d.groupby("account"):
            s = np.sort(grp["pos"].values)[::-1]
            tot = s.sum()
            if tot <= 0:
                continue
            sh = s / tot
            per.append({
                "top1": sh[:1].sum(), "top3": sh[:3].sum(), "top5": sh[:5].sum(),
                "hhi": (sh ** 2).sum(),
                "rpc": float((np.arange(1, len(sh) + 1) * sh).sum()),
                "eff": 1.0 / (sh ** 2).sum(),
            })
        if per:
            pdf = pd.DataFrame(per).mean()
            rows.append({"group": g, "top1": round(pdf.top1, 2), "top3": round(pdf.top3, 2),
                         "top5": round(pdf.top5, 2), "rpc": round(pdf.rpc, 2),
                         "hhi": round(pdf.hhi, 2), "eff_num": round(pdf.eff, 2)})
    return pd.DataFrame(rows)


def build() -> None:
    print("building tables ...")
    _save(table1(), "table1_skill_shares")
    a, b = table2()
    _save(a, "table2A_group_totals"); _save(b, "table2B_group_averages")
    pa, pb = table3()
    _save(pa, "table3A_correlations"); _save(pb, "table3B_transition")
    t4, t5 = tables45()
    _save(t4, "table4_oib"); _save(t5, "table5_accuracy")
    _save(tableE8(), "tableE8_market_summary")
    _save(tableE10(), "tableE10_pnl_distribution")
    _save(tableE11(), "tableE11_concentration")
    if (C.PROC / "table8.parquet").exists():
        _save(pd.read_parquet(C.PROC / "table8.parquet"), "table8_news")
    if (C.PROC / "table6_e6_e7.parquet").exists():
        _save(pd.read_parquet(C.PROC / "table6_e6_e7.parquet"), "table6_insider_markettype")


if __name__ == "__main__":
    build()
