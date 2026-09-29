"""Generate outputs/TOP_TRADERS.ipynb: an interactive notebook version of
TOP_TRADERS.md with a searchable, sortable DataFrame of the skilled and persistent
accounts.

The notebook loads outputs/tables/top_traders.csv into a pandas DataFrame `df`,
renders it as an interactive DataTable (itables: type in the search box, click a
column to sort), and provides plain-pandas helper functions as a fallback.
Regenerate with: python -m src.report.make_notebook
"""
from __future__ import annotations

import json

import config as C

MD_INTRO = """\
# Top Skilled & Persistent Polymarket Accounts

Reproduced from *Prediction Market Accuracy: Crowd Wisdom or Informed Minority?*
(Gómez-Cram, Guo, Jensen & Kung, 2026), using the full Polymarket 2023-2025
transaction universe.

**Who is on this list.** Every account here is *skilled*: its realized profit is
statistically significant under the paper's sign-randomization test
(p&nbsp;<&nbsp;0.05, after removing market-makers and requiring >=10 resolved
events), **and** *persistent*: each account's events are split into two
independent halves and the skill holds in **both** (p&nbsp;<&nbsp;0.05 on each),
so a single lucky run can't qualify. **About 31,000 of the 55,120 skilled
accounts (about 56%)** clear this out-of-sample bar.

### How to use this notebook
1. **Run all cells** (*Run All*).
2. Use the **search box** on the interactive table, and **click any column
   header to sort**. Click an account's **profile** link to open it on Polymarket.
3. Prefer code? The `df` DataFrame and the `find()` / `sort_by()` helpers below
   let you search, filter, sort, and export.
"""

CODE_SETUP = '''\
# --- setup: load the data into a searchable/sortable DataFrame `df` -----------
from pathlib import Path
import pandas as pd

def _find_csv():
    for base in [Path.cwd(), *Path.cwd().parents]:
        for rel in ("outputs/tables/top_traders.csv", "tables/top_traders.csv"):
            p = base / rel
            if p.exists():
                return p
    raise FileNotFoundError(
        "top_traders.csv not found. Generate it first:\\n"
        "    python -m src.report.top_traders")

_raw = pd.read_csv(_find_csv())

df = pd.DataFrame({
    "rank":        _raw["rank"].astype(int),
    "account":     _raw["account"],
    "profit_usd":  _raw["pnl_usd"].round(0).astype("int64"),
    "volume_usd":  _raw["volume_usd"].round(0).astype("int64"),
    "roi_pct":     (_raw["roi"] * 100).round(1),          # profit per $ traded
    "events":      _raw["n_events"].astype(int),          # resolved events traded
    "markets":     _raw["n_markets"].astype(int),
    "win_pct":     (_raw["win_rate"] * 100).round(0),      # % of events profitable
    "skill_p":     _raw["pvalue_full"].round(3),           # full-sample skill p-value
    "p_train":     _raw["p_train"].round(3),               # skill p, training half
    "p_test":      _raw["p_test"].round(3),                # skill p, test half
    "profile_url": _raw["profile_url"],
})

_tot = df.profit_usd.sum()
_top1pct = df.nlargest(max(1, len(df) // 100), "profit_usd").profit_usd.sum()
print(f"{len(df):,} skilled & persistent accounts loaded into `df`")
print(f"combined realized profit: ${_tot/1e6:,.0f}M "
      f"(top account ${df.profit_usd.max()/1e6:.2f}M)")
print(f"note: 'skilled' = statistically reliable direction-picking, not size; "
      f"the top 1% of these accounts earned {100*_top1pct/_tot:.0f}% of the "
      f"total profit, so the list is sorted by profit (biggest first).")
df.head()
'''

CODE_ITABLES = '''\
# --- interactive table: SEARCH (box, top-right) + SORT (click a header) -------
# Works in VS Code / Jupyter. Needs itables:  pip install itables
try:
    from itables import init_notebook_mode, show
    import itables.options as itopt
    init_notebook_mode(all_interactive=False, connected=True)
    itopt.maxBytes = 0        # no cap -> all rows are searchable client-side
    itopt.maxRows = 0
    itopt.maxColumns = 0
    itopt.classes = "display compact"
    itopt.lengthMenu = [25, 50, 100, 250]

    _disp = df.copy()
    _disp["profile"] = ('<a href="' + _disp["profile_url"]
                        + '" target="_blank">open</a>')
    _disp = _disp.drop(columns=["profile_url"])
    show(_disp, allow_html=True)
except Exception as e:                      # offline / itables missing -> fallback
    from IPython.display import HTML, display
    print(f"(interactive table unavailable: {e})")
    print("Showing the top 200; use the helpers in the next cell to search all.")
    _d = df.head(200).copy()
    _d["profile"] = ('<a href="' + _d["profile_url"]
                     + '" target="_blank">open</a>')
    display(HTML(_d.drop(columns=["profile_url"]).to_html(escape=False, index=False)))
'''

MD_HELPERS = """\
## Search & sort with code (works offline)

`df` is a normal pandas DataFrame. Two helpers cover the common cases; edit and
re-run the examples freely.
"""

CODE_HELPERS = '''\
def find(text):
    """Rows whose wallet address contains `text` (case-insensitive)."""
    m = df["account"].str.contains(str(text).lower(), case=False, na=False)
    return df[m]

def sort_by(column, top=20, ascending=False):
    """Top `top` accounts sorted by `column` (descending by default)."""
    return df.sort_values(column, ascending=ascending).head(top)

print("columns you can sort/filter on:")
print("  " + ", ".join(df.columns))

# example: the 15 most profitable accounts
sort_by("profit_usd", top=15)
'''

MD_EXAMPLES = """\
### More example queries: copy, tweak, run
"""

CODE_EXAMPLES = '''\
# 1) Highest return-on-volume among genuinely active traders (>=100 events)
active_best_roi = (df[df.events >= 100]
                   .sort_values("roi_pct", ascending=False)
                   .head(15))

# 2) High win-rate AND sizable profit
sharp = (df[(df.win_pct >= 70) & (df.profit_usd >= 500_000)]
         .sort_values("profit_usd", ascending=False))

# 3) Look up a specific wallet (partial address is fine)
example_lookup = find("0x9d84")

active_best_roi
'''

MD_GUIDE = """\
## Column guide & notes

| Column | Meaning |
|---|---|
| `rank` | Rank by realized profit among persistent-skilled accounts |
| `account` | On-chain wallet address |
| `profit_usd` | Realized PnL over 2023-2025 (settled + marked to resolved outcome) |
| `volume_usd` | Total dollar volume traded |
| `roi_pct` | Profit / volume, in % (return on every dollar traded) |
| `events` / `markets` | Count of resolved events / markets traded |
| `win_pct` | Share of events that were profitable |
| `skill_p` | Full-sample skill p-value (lower = stronger; <0.05 = skilled) |
| `p_train` / `p_test` | Skill p-value on each independent event half (both <0.05 -> persistent) |
| `profile_url` | Link to the trader's Polymarket page |

**Method notes.**
- *Profit* is measured as in the paper: for each position, shares x direction x
  (resolved outcome - trade price), summed over all trades and events
  (equivalently, holding every position to settlement).
- *Persistence* uses a 50/50 random split of **events**, so the two halves share
  no event, a clean out-of-sample test.
- Not every wallet sets a public username; some profiles show only the address
  and trade history.
"""

CODE_EXPORT = '''\
# Export any filtered / sorted view to CSV, e.g. the sharp traders from above:
# sharp.to_csv("my_selection.csv", index=False)

# quick distribution snapshot
df[["profit_usd", "volume_usd", "roi_pct", "events", "win_pct"]].describe().round(1)
'''


def _md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def _code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src}


def build() -> None:
    cells = [
        _md(MD_INTRO),
        _code(CODE_SETUP),
        _md("## Interactive table: type in the search box, click a header to "
            "sort"),
        _code(CODE_ITABLES),
        _md(MD_HELPERS),
        _code(CODE_HELPERS),
        _md(MD_EXAMPLES),
        _code(CODE_EXAMPLES),
        _md(MD_GUIDE),
        _code(CODE_EXPORT),
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = C.OUT / "TOP_TRADERS.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
