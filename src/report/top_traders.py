"""Rank the top skilled and persistent Polymarket accounts.

"Skilled" = full-sample sign-randomization p-value < 0.05 (after the market-maker
rule and the >=10-event benchmark filter), i.e. group == "Skilled" in
classification.parquet.

"Persistent" = the skill holds out of sample. Events are split 50/50 into a
training and a test set (the paper's persistence design, event_split.parquet) and
the skill test is recomputed independently on each half. An account is
persistent-skilled only if it is significant (p < 0.05) in BOTH halves, so its
edge is not an artifact of a single lucky stretch of events.

For each qualifying account this reports realized PnL, dollar volume, return on
volume, events/markets traded, event win-rate, the full-sample p-value, and the
two split p-values, plus a link to its Polymarket profile. Writes:
  outputs/tables/top_traders.csv: every persistent-skilled account, ranked
  outputs/TOP_TRADERS.md: a readable top-N table with profile links

Usage:  python -m src.report.top_traders [N]
"""
from __future__ import annotations

import sys

import pandas as pd

import config as C
from src.duckutil import connect as _safe_connect
from src.skill.sign_randomization import compute_pvalues

PROFILE = "https://polymarket.com/profile/{}"
MIN_SPLIT_EVENTS = 4  # need >=4 events per half for p<0.05 to be attainable


def _split_pvalues() -> pd.DataFrame:
    """Per-account p-value on the train (0) and test (1) event splits.

    Restricted to full-sample skilled accounts for speed; reads per-event PnL
    (account_event_pnl) joined to the saved event split, with no trade re-scan.
    """
    con = _safe_connect()
    con.execute("PRAGMA threads=6")
    con.execute(f"""
        CREATE TEMP TABLE skilled AS
        SELECT account FROM read_parquet('{C.PROC / 'classification.parquet'}')
        WHERE "group" = 'Skilled'
    """)
    ae = con.execute(f"""
        SELECT p.account, s.split,
               list(p.s_e) AS s, COUNT(*) AS E, SUM(p.s_e) AS pnl
        FROM read_parquet('{C.PROC / 'account_event_pnl.parquet'}') p
        JOIN read_parquet('{C.PROC / 'event_split.parquet'}') s USING (event_id)
        WHERE p.account IN (SELECT account FROM skilled)
        GROUP BY p.account, s.split
    """).fetchdf()
    con.close()

    out = []
    for sp, tag in ((0, "train"), (1, "test")):
        sub = ae[ae.split == sp]
        pv = compute_pvalues(sub[["account", "s", "E", "pnl"]].reset_index(drop=True))
        pv = pv.rename(columns={"E": f"E_{tag}", "pvalue": f"p_{tag}",
                                "pnl": f"pnl_{tag}"})
        out.append(pv.set_index("account"))
    return out[0].join(out[1], how="outer").reset_index()


def _win_rates(accounts: list[str]) -> pd.DataFrame:
    """Event win-rate (share of events with positive PnL) for given accounts."""
    con = _safe_connect()
    con.register("acc", pd.DataFrame({"account": accounts}))
    wr = con.execute(f"""
        SELECT p.account,
               AVG(CASE WHEN p.s_e > 0 THEN 1.0 ELSE 0.0 END) AS win_rate
        FROM read_parquet('{C.PROC / 'account_event_pnl.parquet'}') p
        JOIN acc USING (account)
        GROUP BY p.account
    """).fetchdf()
    con.close()
    return wr


def _short(addr: str) -> str:
    return f"{addr[:6]}...{addr[-4:]}"


def _usd(x: float) -> str:
    ax = abs(x)
    if ax >= 1e9:
        return f"${x/1e9:.2f}B"
    if ax >= 1e6:
        return f"${x/1e6:.2f}M"
    if ax >= 1e3:
        return f"${x/1e3:.1f}K"
    return f"${x:.0f}"


def build(top_n: int = 30) -> pd.DataFrame:
    cls = pd.read_parquet(C.PROC / "classification.parquet")
    skilled = cls[cls.group == "Skilled"].copy()

    sp = _split_pvalues()
    df = skilled.merge(sp, on="account", how="left")

    # Persistent = significant in BOTH halves, each with enough events for power.
    df["persistent"] = (
        (df["p_train"] < C.SKILL_P) & (df["p_test"] < C.SKILL_P)
        & (df["E_train"] >= MIN_SPLIT_EVENTS) & (df["E_test"] >= MIN_SPLIT_EVENTS)
    )
    persist = df[df.persistent].copy()

    wr = _win_rates(persist["account"].tolist())
    persist = persist.merge(wr, on="account", how="left")

    persist["roi"] = persist["pnl"] / persist["volume"].where(persist["volume"] > 0)
    persist = persist.sort_values("pnl", ascending=False).reset_index(drop=True)
    persist["rank"] = persist.index + 1
    persist["profile_url"] = persist["account"].map(PROFILE.format)

    cols = ["rank", "account", "pnl", "volume", "roi", "E", "n_markets",
            "win_rate", "pvalue", "p_train", "p_test", "profile_url"]
    out = persist[cols].rename(columns={
        "pnl": "pnl_usd", "volume": "volume_usd", "E": "n_events",
        "pvalue": "pvalue_full"})
    out.to_csv(C.TABLES / "top_traders.csv", index=False)

    _write_md(out, top_n, n_persist=len(persist), n_skilled=len(skilled))
    print(f"persistent-skilled accounts: {len(persist):,} of {len(skilled):,} skilled")
    print(f"wrote {C.TABLES/'top_traders.csv'} and {C.OUT/'TOP_TRADERS.md'}")
    return out


def _write_md(out: pd.DataFrame, top_n: int, n_persist: int, n_skilled: int) -> None:
    L = [
        "# Top Skilled & Persistent Polymarket Accounts",
        "",
        "Reproduced from *Prediction Market Accuracy: Crowd Wisdom or Informed "
        "Minority?* (Gomez-Cram, Guo, Jensen & Kung, 2026), using the full "
        "Polymarket 2023-2025 transaction universe.",
        "",
        "**Who is on this list.** Every account below is skilled: its realized "
        "profit is statistically significant under the paper's sign-randomization "
        "test (p < 0.05, after removing market-makers and requiring >=10 "
        "resolved events), and persistent: each account's events are split "
        "into two independent halves and the skill holds in both (p < 0.05 on "
        "each). This rules out a single lucky stretch of events. "
        f"**{n_persist:,}** of the {n_skilled:,} skilled accounts clear this "
        "out-of-sample bar.",
        "",
        "Click an account to open its live Polymarket profile.",
        "",
        "## Column guide",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| Profit | Realized PnL over 2023-2025 (settled + mark-to-outcome) |",
        "| Volume | Total dollar volume traded |",
        "| ROI | Profit / volume (return on every dollar traded) |",
        "| Events | Number of resolved events traded |",
        "| Win rate | Share of events that were profitable |",
        "| Skill p | Full-sample skill p-value (lower = stronger; <0.05 = skilled) |",
        "",
        f"## Top {min(top_n, len(out))} by realized profit",
        "",
        "| # | Account | Profit | Volume | ROI | Events | Win rate | Skill p | Profile |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:--:|",
    ]
    for _, r in out.head(top_n).iterrows():
        roi = "-" if pd.isna(r.roi) else f"{r.roi*100:,.1f}%"
        wr = "-" if pd.isna(r.win_rate) else f"{r.win_rate*100:.0f}%"
        L.append(
            f"| {int(r['rank'])} | `{_short(r.account)}` | {_usd(r.pnl_usd)} | "
            f"{_usd(r.volume_usd)} | {roi} | {int(r.n_events):,} | {wr} | "
            f"{r.pvalue_full:.3f} | [open]({r.profile_url}) |")
    L += [
        "",
        f"*Full ranked list of all {n_persist:,} persistent-skilled accounts "
        "(with train/test split p-values): `outputs/tables/top_traders.csv`.*",
        "",
        "### Notes",
        "- Addresses are on-chain wallet addresses; the Profile link resolves to "
        "the trader's Polymarket page (username, if set, shown there).",
        "- Profit is measured as in the paper: for each position, shares x "
        "direction x (resolved outcome - trade price), summed across all "
        "trades and events (equivalently, holding every position to settlement).",
        "- Persistence is evaluated on a 50/50 random split of *events*, so the "
        "two halves share no event and the out-of-sample test is clean.",
    ]
    (C.OUT / "TOP_TRADERS.md").write_text("\n".join(L))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    build(n)
