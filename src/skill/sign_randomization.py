"""Sign-randomization skill test (Section 3.1, eq. 1-3).

Each account has event-level PnL contributions S_e. Realized
PnL = sum_e S_e. Under the null the account's direction in each event is random,
so PnL^(b) = sum_e eps_e^(b) * S_e with eps in {-1,+1}. The p-value is the
mid-p share of simulated PnLs at or above realized, truncated to the range the
discreteness of E events allows: [2^-(E+1), 1 - 2^-(E+1)].

Vectorised by grouping accounts with the same event count E and evaluating
PnL^(b) = S @ signs.T as a single matmul per group. Sharing one (B, E) sign
matrix across accounts in a group is a valid Monte-Carlo draw of each account's
own null (their S_e differ), and makes the whole cross-section run in seconds.
"""
from __future__ import annotations

import argparse

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm

import config as C
from src.duckutil import connect as _safe_connect


def _load_account_events() -> pd.DataFrame:
    con = _safe_connect()
    con.execute("PRAGMA threads=6")
    df = con.execute(f"""
        SELECT account,
               list(s_e) AS s,
               COUNT(*)  AS E,
               SUM(s_e)  AS pnl
        FROM read_parquet('{C.PROC / 'account_event_pnl.parquet'}')
        GROUP BY account
    """).fetchdf()
    con.close()
    return df


def compute_pvalues(df: pd.DataFrame, B: int = C.B_SIMS, seed: int = C.RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["E"] = df["E"].astype(int)
    out_p = np.empty(len(df), dtype=float)
    df = df.reset_index(drop=True)

    for E, idx in tqdm(df.groupby("E").groups.items(), desc="skill test (by E)"):
        rows = df.loc[idx]
        S = np.array([np.asarray(v, dtype=float) for v in rows["s"].values])  # (n, E)
        actual = S.sum(axis=1)                                                # (n,)
        signs = rng.choice(np.array([-1.0, 1.0]), size=(B, E))                # (B, E)
        sim = S @ signs.T                                                     # (n, B)
        gt = (sim > actual[:, None]).sum(axis=1)
        eq = (sim == actual[:, None]).sum(axis=1)
        p = (gt + 0.5 * eq) / B
        lo = 2.0 ** (-(E + 1))
        p = np.clip(p, lo, 1.0 - lo)
        out_p[np.asarray(idx)] = p

    df["pvalue"] = out_p
    return df[["account", "E", "pnl", "pvalue"]]


def build() -> None:
    df = _load_account_events()
    print(f"accounts with >=1 event: {len(df):,}")
    res = compute_pvalues(df)
    res.to_parquet(C.PROC / "account_pvalues.parquet", index=False)
    # quick distribution sanity (Figure 2 shape)
    ge10 = res[res.E >= 10]
    print(f">=10 events: {len(ge10):,}")
    print(f"  skilled (p<0.05):      {(ge10.pvalue < 0.05).mean()*100:.2f}%")
    print(f"  anti-skilled (p>0.95): {(ge10.pvalue > 0.95).mean()*100:.2f}%")
    print(f"  middle p in [.4,.6]:   {ge10.pvalue.between(0.4,0.6).mean()*100:.2f}% "
          f"(uniform null ~20%)")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    build()
