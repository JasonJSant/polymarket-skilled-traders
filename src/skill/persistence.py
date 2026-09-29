"""Out-of-sample persistence of skill (Section 3.3, Table 3).

Randomly split events into disjoint training and test sets, classify accounts on
each, and check whether the classification holds out of sample:

  Panel A: Spearman rank correlation of PnL and of p-values between train/test,
           full and on restricted subsamples.
  Panel B: 5x5 transition matrix of group membership (train -> test).

The market-maker rule is re-evaluated within each split (its features are
computed only from that split's trades), so MM persistence is measured too.
"""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import config as C
from src.duckutil import connect as _safe_connect
from src.skill.classify import classify
from src.skill.sign_randomization import compute_pvalues

TRADES_GLOB = C.TRADES_GLOB


def _event_split(seed: int = C.RNG_SEED) -> pd.DataFrame:
    """Assign each event to train (0) or test (1), 50/50."""
    ev = pd.read_parquet(C.RAW / "events.parquet", columns=["event_id"])
    rng = np.random.default_rng(seed)
    ev["split"] = rng.integers(0, 2, size=len(ev))
    return ev


def _features_by_split(con, split_df: pd.DataFrame) -> pd.DataFrame:
    """Per (account, split) MM features + pnl + n_events, from trade shards."""
    con.register("split_map", split_df)
    return con.execute(f"""
        WITH tp AS (
            SELECT t.account, t.event_id, s.split, t.condition_id AS mcid,
                   t.is_maker, t.d, t.notional,
                   t.size * t.d * (tm.y - t.price) AS pnl
            FROM read_parquet('{TRADES_GLOB}') t
            JOIN read_parquet('{C.RAW / 'token_map.parquet'}') tm ON tm.token_id = t.token_id
            JOIN split_map s ON s.event_id = t.event_id
        ),
        per_mkt AS (
            SELECT account, split, mcid,
                   SUM(notional) vol,
                   SUM(CASE WHEN is_maker THEN notional ELSE 0 END) maker_vol,
                   MAX(CASE WHEN d=1 THEN 1 ELSE 0 END) has_buy,
                   MAX(CASE WHEN d=-1 THEN 1 ELSE 0 END) has_sell
            FROM tp GROUP BY account, split, mcid
        )
        SELECT account, split,
               COUNT(*) AS n_markets,
               SUM(vol) AS volume,
               SUM(maker_vol)/NULLIF(SUM(vol),0) AS maker_volume_share,
               AVG(CASE WHEN has_buy=1 AND has_sell=1 THEN 1.0 ELSE 0.0 END)
                   AS both_sides_share
        FROM per_mkt GROUP BY account, split
    """).fetchdf()


def _pvals_by_split(con, split_df: pd.DataFrame) -> pd.DataFrame:
    """Per (account, split): event PnL list -> p-value via the sign test."""
    con.register("split_map2", split_df)
    ae = con.execute(f"""
        SELECT p.account, s.split, list(p.s_e) AS s, COUNT(*) AS E, SUM(p.s_e) AS pnl
        FROM read_parquet('{C.PROC / 'account_event_pnl.parquet'}') p
        JOIN split_map2 s USING (event_id)
        GROUP BY p.account, s.split
    """).fetchdf()
    out = []
    for sp in (0, 1):
        sub = ae[ae.split == sp]
        pv = compute_pvalues(sub[["account", "s", "E", "pnl"]].reset_index(drop=True))
        pv["split"] = sp
        out.append(pv)
    return pd.concat(out, ignore_index=True)


def build() -> None:
    split_df = _event_split()
    con = _safe_connect()
    con.execute("PRAGMA threads=6")

    feats = _features_by_split(con, split_df)
    pvs = _pvals_by_split(con, split_df)
    con.close()

    merged = pvs.merge(feats, on=["account", "split"], how="left")
    train = merged[merged.split == 0].copy()
    test = merged[merged.split == 1].copy()

    cls_tr = classify(train[["account", "E", "pnl", "pvalue"]],
                      train[["account", "n_markets", "maker_volume_share",
                             "both_sides_share", "volume"]],
                      event_filter=C.BENCHMARK_EVENT_FILTER)
    cls_te = classify(test[["account", "E", "pnl", "pvalue"]],
                      test[["account", "n_markets", "maker_volume_share",
                            "both_sides_share", "volume"]],
                      event_filter=C.BENCHMARK_EVENT_FILTER)

    j = (cls_tr.rename(columns={"group": "g_train", "pnl": "pnl_tr", "pvalue": "p_tr"})
         .merge(cls_te.rename(columns={"group": "g_test", "pnl": "pnl_te",
                                       "pvalue": "p_te"})[["account", "g_test",
                                                           "pnl_te", "p_te"]],
                on="account", how="inner"))

    # Panel A: rank correlations
    def rc(a, b, mask=None):
        d = j if mask is None else j[mask]
        if len(d) < 3:
            return np.nan, len(d)
        return spearmanr(d[a], d[b]).statistic, len(d)

    panelA = {
        "pnl_full": rc("pnl_tr", "pnl_te"),
        "pnl_profit": rc("pnl_tr", "pnl_te", j.pnl_tr > 0),
        "pnl_loss": rc("pnl_tr", "pnl_te", j.pnl_tr < 0),
        "p_full": rc("p_tr", "p_te"),
        "p_skill": rc("p_tr", "p_te", j.p_tr < 0.5),
        "p_nonskill": rc("p_tr", "p_te", j.p_tr > 0.5),
    }

    # Panel B: transition matrix (rows sum to 1)
    trans = (pd.crosstab(j.g_train, j.g_test, normalize="index")
             .reindex(index=C.GROUPS, columns=C.GROUPS).fillna(0))

    pd.DataFrame({k: {"corr": v[0], "n": v[1]} for k, v in panelA.items()}).T.to_parquet(
        C.PROC / "persistence_panelA.parquet")
    trans.to_parquet(C.PROC / "persistence_panelB.parquet")

    # Persist split + training-set classification for the price-discovery
    # module (paper classifies on train, reports on test).
    split_df.to_parquet(C.PROC / "event_split.parquet", index=False)
    cls_tr[["account", "group"]].to_parquet(
        C.PROC / "classification_train.parquet", index=False)

    print("Panel A (rank correlations):")
    for k, (c, n) in panelA.items():
        print(f"  {k:12s}: corr={c:.3f}  n={n:,}")
    print("\nPanel B (train->test), skilled->skilled = "
          f"{trans.loc['Skilled','Skilled']:.2f}")
    print(trans.round(2).to_string())


if __name__ == "__main__":
    build()
