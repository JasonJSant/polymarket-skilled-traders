"""Regenerate the paper's figures (PNG + PDF) into outputs/figures/.

Fig 1  Polymarket trading activity (volume, active accounts, volume percentiles)
Fig 2  distribution of skill-test p-values (skilled / anti-skilled / others)
Fig 3  order imbalance before resolution by group (resolved Yes vs No)
Fig 4  arbitrage activity by group
Fig 5  net buy volume across the price distribution by group
Fig E.2 prediction-market calibration
"""
from __future__ import annotations

import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from src.duckutil import connect as _safe_connect

TRADES_GLOB = C.TRADES_GLOB
# paper-like group colours
GC = {"Market Maker": "#888888", "Skilled": "#00008B", "Lucky": "#4A90D9",
      "Unlucky": "#F08080", "Anti-skilled": "#8B0000"}


def _save(fig, name: str) -> None:
    fig.savefig(C.FIGURES / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(C.FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}")


def fig1() -> None:
    con = _safe_connect()
    mon = con.execute(f"""
        SELECT strftime(to_timestamp(ts), '%Y-%m') AS ym,
               SUM(notional)/2 AS vol, COUNT(DISTINCT account) AS accts
        FROM read_parquet('{TRADES_GLOB}') GROUP BY ym ORDER BY ym
    """).fetchdf()
    pct = con.execute(f"""
        WITH am AS (
            SELECT strftime(to_timestamp(ts), '%Y-%m') ym, account, SUM(notional)/2 v
            FROM read_parquet('{TRADES_GLOB}') GROUP BY ym, account
        )
        SELECT ym, quantile_cont(v,0.5) p50, quantile_cont(v,0.95) p95,
               quantile_cont(v,0.99) p99 FROM am GROUP BY ym ORDER BY ym
    """).fetchdf()
    con.close()
    x = pd.to_datetime(mon["ym"] + "-01")
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    ax[0, 0].bar(x, mon["vol"] / 1e9, width=20, color="#00008B")
    ax[0, 0].set_title("Panel A: Total Trading Volume"); ax[0, 0].set_ylabel("Volume ($B)")
    ax[1, 0].bar(x, mon["accts"] / 1e3, width=20, color="#00008B")
    ax[1, 0].set_title("Panel B: Number of Active Accounts"); ax[1, 0].set_ylabel("Accounts (000s)")
    xp = pd.to_datetime(pct["ym"] + "-01")
    ax[1, 1].plot(xp, pct["p99"], label="99 pct", color="#00008B")
    ax[1, 1].plot(xp, pct["p95"], label="95 pct", color="red")
    ax[1, 1].plot(xp, pct["p50"], label="50 pct", color="gray")
    ax[1, 1].set_yscale("log"); ax[1, 1].legend()
    ax[1, 1].set_title("Panel C: Account-level Volume"); ax[1, 1].set_ylabel("Volume ($, log)")
    ax[0, 1].axis("off")
    # thin out and format the monthly date ticks so labels don't overlap
    for a in (ax[0, 0], ax[1, 0], ax[1, 1]):
        loc = mdates.AutoDateLocator(maxticks=8)
        a.xaxis.set_major_locator(loc)
        a.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    _save(fig, "fig1_activity")


def fig2() -> None:
    pv = pd.read_parquet(C.PROC / "account_pvalues.parquet")
    pv = pv[pv.E >= C.BENCHMARK_EVENT_FILTER]
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 51)
    counts, edges = np.histogram(pv["pvalue"], bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    colors = ["#00008B" if c < 0.05 else "#CC0000" if c > 0.95 else "#B0B0B0"
              for c in centers]
    ax.bar(centers, counts / 1e3, width=1/50, color=colors)
    ax.axhline(len(pv) / 50 / 1e3, ls="--", color="k", lw=0.8)
    ax.set_xlabel("P-Value"); ax.set_ylabel("Number of Accounts (thousands)")
    ax.set_title("Distribution of p-values from skill test")
    _save(fig, "fig2_pvalues")


def fig3() -> None:
    panel = pd.read_parquet(C.PROC / "bins.parquet")
    last = panel[panel["bin"] >= int(C.N_BINS * (1 - C.LAST_FRAC))]
    keys = {"Market Maker": "mm", "Skilled": "sk", "Lucky": "lk",
            "Unlucky": "ul", "Anti-skilled": "as"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for j, (yv, title) in enumerate([(1, "Resolved YES"), (0, "Resolved NO")]):
        sub = last[last["y0"] == yv]
        vals = [sub[f"OIB_{k}"].mean() for k in keys.values()]
        ax[j].bar(list(keys), vals, color=[GC[g] for g in keys])
        ax[j].axhline(0, color="k", lw=0.6); ax[j].set_title(f"Panel {'AB'[j]}: {title}")
        ax[j].tick_params(axis="x", rotation=30)
    ax[0].set_ylabel("Average OIB (YES token)")
    _save(fig, "fig3_oib_resolution")


def fig4() -> None:
    d = pd.read_parquet(C.PROC / "strat_arbitrage.parquet").set_index("grp")
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    order = [g for g in C.GROUPS if g in d.index]
    ax[0].bar(order, [d.loc[g, "net_buy_mshares"] for g in order], color=[GC[g] for g in order])
    ax[0].set_title("Panel A: Net Buy Volume (M shares)")
    ax[1].bar(order, [d.loc[g, "pnl_m"] for g in order], color=[GC[g] for g in order])
    ax[1].set_title("Panel B: Realized PnL ($M)")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
    _save(fig, "fig4_arbitrage")


def fig5() -> None:
    d = pd.read_parquet(C.PROC / "strat_longshot.parquet")
    d = d[d.grp != "Unclassified"]
    order = [f"{lo}-{hi}" for lo, hi in C.LONGSHOT_BINS]
    piv = d.pivot_table(index="price_bin", columns="grp", values="net_mshares",
                        aggfunc="sum").reindex(order)
    fig, ax = plt.subplots(figsize=(11, 5))
    groups = [g for g in C.GROUPS if g in piv.columns]
    w = 0.15
    xpos = np.arange(len(order))
    for i, g in enumerate(groups):
        ax.bar(xpos + i * w, piv[g].values, width=w, label=g, color=GC[g])
    ax.set_xticks(xpos + w * len(groups) / 2); ax.set_xticklabels(order)
    ax.axhline(0, color="k", lw=0.6); ax.legend()
    ax.set_xlabel("Price Bins"); ax.set_ylabel("Net Buy Volume (M shares)")
    _save(fig, "fig5_longshot")


def figE2() -> None:
    d = pd.read_parquet(C.PROC / "strat_calibration.parquet").sort_values("price")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="45 deg")
    ax.scatter(d["price"], d["outcome"], color="#00008B", s=15)
    ax.set_xlabel("Prediction Market Price"); ax.set_ylabel("Actual Outcome")
    ax.set_title("Prediction market calibration"); ax.legend()
    _save(fig, "figE2_calibration")


def figE1() -> None:
    d = pd.read_parquet(C.PROC / "strat_dispute.parquet").set_index("grp")
    order = [g for g in C.GROUPS if g in d.index]
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    ax[0].bar(order, [d.loc[g, "net_win_mshares"] for g in order], color=[GC[g] for g in order])
    ax[0].set_title("Panel A: Net Buy of Winning Token (M shares)")
    ax[1].bar(order, [d.loc[g, "pnl_m"] for g in order], color=[GC[g] for g in order])
    ax[1].set_title("Panel B: Realized PnL ($M)")
    for a in ax:
        a.axhline(0, color="k", lw=0.6); a.tick_params(axis="x", rotation=30)
    _save(fig, "figE1_dispute_window")


def build() -> None:
    print("building figures ...")
    for fn in (fig1, fig2, fig3, fig4, fig5, figE1, figE2):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  {fn.__name__} skipped: {str(e)[:100]}")


if __name__ == "__main__":
    build()
