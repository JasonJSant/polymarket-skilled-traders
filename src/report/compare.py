"""Assemble outputs/REPORT.md: reproduced numbers next to the paper's.

Reads the regenerated tables and the processed data, and lines up the headline
figures against the values transcribed from the paper (config.paper_targets),
flagging the deviation for each. Also embeds the regenerated figures.
"""
from __future__ import annotations

import pandas as pd

import config as C


def _fmt(x, d=2):
    try:
        return f"{float(x):,.{d}f}"
    except (TypeError, ValueError):
        return str(x)


def _row(label, repro, paper, note=""):
    dev = ""
    try:
        if paper not in (None, 0):
            dev = f"{(float(repro)-float(paper))/float(paper)*100:+.0f}%"
    except (TypeError, ValueError):
        pass
    return f"| {label} | {_fmt(repro)} | {_fmt(paper)} | {dev} | {note} |"


def build() -> None:
    T = C.paper_targets()
    L = ["# Reproduction Report", "",
         "Prediction Market Accuracy: Crowd Wisdom or Informed Minority? "
         "(Gomez-Cram, Guo, Jensen, Kung 2026).",
         "", "Reproduced from the public Polymarket data (GAMMA metadata + "
         "Goldsky orderbook subgraph fills). Target: magnitudes + conclusions.",
         "", "## Headline comparison", "",
         "| Quantity | Reproduced | Paper | Dev | Note |",
         "|---|---|---|---|---|"]

    # Paper universe = markets "with at least one trade" (Table E.8 note).
    import duckdb
    _con = duckdb.connect()
    nmk, nev = _con.execute(
        f"SELECT COUNT(DISTINCT condition_id), COUNT(DISTINCT event_id) "
        f"FROM read_parquet('{C.TRADES_GLOB}')").fetchone()
    _con.close()
    L.append(_row("Markets (>=1 trade)", nmk, T["n_markets"]))
    L.append(_row("Events (>=1 trade)", nev, T["n_events"]))

    if (C.PROC / "classification.parquet").exists():
        cls = pd.read_parquet(C.PROC / "classification.parquet")
        n = len(cls)
        sk = int((cls.group == "Skilled").sum())
        an = int((cls.group == "Anti-skilled").sum())
        L.append(_row("Accounts classified", n, T["n_accounts"]))
        L.append(_row("Skilled count", sk, T["skilled_count"]))
        L.append(_row("Skilled % of all", round(100*sk/n, 2), T["skilled_pct_all"]))
        L.append(_row("Anti-skilled count", an, T["antiskilled_count"]))
        L.append(_row("Anti-skilled % of all", round(100*an/n, 2), T["antiskilled_pct_all"]))
        vol = pd.read_parquet(C.PROC / "account_features.parquet")["volume"].sum() / 2
        L.append(_row("Total volume ($B)", round(vol/1e9, 2), T["total_volume_usd"]/1e9))

    if (C.PROC / "persistence_panelA.parquet").exists():
        pa = pd.read_parquet(C.PROC / "persistence_panelA.parquet")
        pb = pd.read_parquet(C.PROC / "persistence_panelB.parquet")
        if "p_full" in pa.index:
            L.append(_row("p-value corr (train/test)", round(pa.loc["p_full", "corr"], 3),
                          T["pval_corr"]))
        if "Skilled" in pb.index:
            L.append(_row("Skilled->Skilled persistence", round(pb.loc["Skilled", "Skilled"], 2),
                          T["skilled_to_skilled"]))

    # Table 4 headline (skilled OIB)
    if (C.PROC / "table4.parquet").exists():
        t4 = pd.read_parquet(C.PROC / "table4.parquet")
        sk_dp = t4[(t4.group == "Skilled") & (t4.panel.str.contains("Delta"))]
        sk_out = t4[(t4.group == "Skilled") & (t4.panel == "outcome")]
        L += ["", "## Price discovery (Table 4, skilled OIB)", "",
              "| Dep var | Reproduced beta (t) | Paper |", "|---|---|---|"]
        if len(sk_dp):
            L.append(f"| Delta p_t+1 | {sk_dp.coef.iloc[0]:.2f} ({sk_dp.t.iloc[0]:.1f}) | 2.05 (12.8) |")
        if len(sk_out):
            L.append(f"| outcome | {sk_out.coef.iloc[0]:.2f} ({sk_out.t.iloc[0]:.1f}) | 7.77 (9.6) |")

    L += ["", "## Regenerated artifacts", "",
          "Tables: `outputs/tables/*.csv` (+ `.tex`). Figures: `outputs/figures/*.png`.",
          "", "See `docs/METHODOLOGY.md` for methodology notes and deviations.", ""]

    # embed figures if present
    for fig in ["fig1_activity", "fig2_pvalues", "fig3_oib_resolution",
                "fig4_arbitrage", "fig5_longshot", "figE2_calibration"]:
        p = C.FIGURES / f"{fig}.png"
        if p.exists():
            L.append(f"![{fig}](figures/{fig}.png)")

    (C.OUT / "REPORT.md").write_text("\n".join(L))
    print(f"wrote {C.OUT / 'REPORT.md'}")


if __name__ == "__main__":
    build()
