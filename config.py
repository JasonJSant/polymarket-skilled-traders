"""Central configuration for the Polymarket reproduction.

All thresholds, date bounds, endpoints, and paths in one place so the whole
pipeline is driven by a single source of truth. Values transcribed from
Gomez-Cram, Guo, Jensen & Kung (2026), "Prediction Market Accuracy".
"""
from __future__ import annotations

import glob as _glob
import os
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
CKPT = DATA / "checkpoints"
TRADES_DIR = RAW / "trades"
CLEAN_TRADES_DIR = DATA / "processed" / "trades_clean"  # operator-cleaned fills
PRICES_DIR = RAW / "prices"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
LOGS = ROOT / "logs"
for _p in (DATA, RAW, PROC, CKPT, TRADES_DIR, PRICES_DIR, OUT, TABLES, FIGURES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

# Analysis reads operator-cleaned trades when they exist (see src/build/clean_trades.py),
# otherwise falls back to the raw subgraph fills. All modules reference C.TRADES_GLOB.
_clean_pat = str(CLEAN_TRADES_DIR / "**" / "*.parquet")
TRADES_GLOB = _clean_pat if _glob.glob(_clean_pat, recursive=True) \
    else str(TRADES_DIR / "**" / "*.parquet")

# --------------------------------------------------------------------------
# Sample window (paper: markets created after Jan 1 2023, resolved by Dec 31 2025)
# --------------------------------------------------------------------------
SAMPLE_START = datetime(2023, 1, 1, tzinfo=timezone.utc)
SAMPLE_END = datetime(2025, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
SAMPLE_START_TS = int(SAMPLE_START.timestamp())
SAMPLE_END_TS = int(SAMPLE_END.timestamp())

# Collection horizon (>= SAMPLE_END). The paper window ends 2025-12-31 (SAMPLE_END,
# used to subset the extended universe back to the paper for reproduction), but
# collection extends further forward: the Goldsky orderbook subgraph stopped indexing
# around 2026-04-28, so April 2026 is the furthest this source reaches (Jan-Mar
# complete, April partial). The collectors (src/collect/{markets,trades}.py) use
# COLLECT_END; the analysis universe stays paper-scoped via the _repro_2023_2025 snapshot.
COLLECT_END = datetime(2026, 4, 30, 23, 59, 59, tzinfo=timezone.utc)
COLLECT_END_TS = int(COLLECT_END.timestamp())

# --------------------------------------------------------------------------
# API endpoints
# --------------------------------------------------------------------------
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
SUBGRAPH = (
    "https://api.goldsky.com/api/public/"
    "project_cl6mb8i9h0003e201j6li0diw/subgraphs/orderbook-subgraph/prod/gn"
)
# NegRisk conversions live in a separate subgraph (used for arbitrage detection).
SUBGRAPH_NEGRISK = (
    "https://api.goldsky.com/api/public/"
    "project_cl6mb8i9h0003e201j6li0diw/subgraphs/negrisk-subgraph/prod/gn"
)

# On-chain amounts are fixed-point with 6 decimals (USDC and CTF shares alike).
USDC_DECIMALS = 6
SCALE = 10 ** USDC_DECIMALS

# Polymarket exchange operator / matching wallets. The CTF Exchange emits an
# OrderFilled for BOTH the maker's order and the taker's order, routing one leg
# of every trade through an operator wallet, so summing raw fills double-counts
# volume and scrambles the maker/taker (limit-order) tag. These addresses appear
# in ~40% of fills and return zero trades from Polymarket's own data-api (i.e.
# they are infrastructure, not traders). Dropping every fill in which an operator
# is the maker or taker reconstructs Polymarket's canonical one-sided volume
# exactly (verified ratio 1.00 across binary and neg-risk markets) and restores
# genuine maker/taker semantics. See src/build/clean_trades.py.
OPERATORS = {
    "0xc5d563a36ae78145c45a50134d48a1215220f80a",
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",
    "0x0059af724d697bcdb4c6087d97fe0d87e26b3206",
    "0x3d524b77646e477f614ed344cc32e39353b0ea0d",
    "0xa4a6fcb5df72529d4a16cf091964d1cf50ccdaa2",
    "0x9ced0600d4ed63da27f283f4f2a472cd205153fe",
    "0x0a9d6f15ec3d6140f303b0473ff1021599b6c67f",
    "0xab08a086a7a05c7ccb172089915e2b327acec6d1",
    "0x83388f407ef27212e5c9f6d95c6cb8ce75b4d135",
    "0x7c3e4d33e1114f328db354db4616b7409b664505",
}


def operator_sql_list() -> str:
    """OPERATORS as a SQL IN-list literal, e.g. "'0x..','0x..'"."""
    return "'" + "','".join(sorted(OPERATORS)) + "'"

# --------------------------------------------------------------------------
# Skill test (Section 3.1)
# --------------------------------------------------------------------------
B_SIMS = 1000                     # sign-randomization replications
SKILL_P = 0.05                    # skilled: p < 0.05
ANTISKILL_P = 0.95                # anti-skilled: p > 0.95
EVENT_FILTERS = [10, 4, None]     # >=10 benchmark; also >=4 and none (Table 1)
BENCHMARK_EVENT_FILTER = 10
RNG_SEED = 20260625              # reproducible sign draws / splits

# Market-maker rule (behavioral, applied before p-value classification).
MM_MIN_MARKETS = 100              # (i) trades in >= 100 markets
MM_LIMIT_SHARE = 0.70            # (ii) >= 70% of volume from limit (maker) orders
MM_BOTHSIDES_SHARE = 0.70        # (iii) trades both sides in >= 70% of its markets

# --------------------------------------------------------------------------
# Price-discovery / accuracy (Sections 3.4-3.5)
# --------------------------------------------------------------------------
N_BINS = 50                       # equal-length time bins per market lifespan
PP = 100.0                        # dep. vars reported in percentage points (x100)
LIFESPAN_WINDOWS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
LAST_FRAC = 0.05                  # Fig 3: last 5% of lifespan
CLUSTER = "event+month"           # two-way clustering

GROUPS = ["Market Maker", "Skilled", "Lucky", "Unlucky", "Anti-skilled"]

# --------------------------------------------------------------------------
# Insider proxies (Section 4.2)
# --------------------------------------------------------------------------
REC_WINRATE = 0.90                # record: per-share win rate >= 90%
REC_PROFIT_PER_SHARE = 0.10       # record: profit/share >= 0.1
REC_MIN_PROFIT = 1000.0           # record: total profit >= $1000
REC_PRONE_PROFIT_SHARE = 0.90     # record: >= 90% of profit in insider-prone mkts
TIM_MAX_DAYS_BEFORE = 7           # timing: opened <= 7d before the event
TIM_MIN_VOLUME = 1000.0           # timing: trade >= $1000
TIM_MIN_PROFIT = 1000.0           # timing: profit >= $1000
CONV_PRICE_MOVE = 0.20            # conviction: a trade that moves price >= 0.2
CONV_PROFIT_SHARE = 0.50          # conviction: > half of profit from such trades
CONV_MIN_PROFIT = 1000.0          # conviction: profit >= $1000

# --------------------------------------------------------------------------
# LLM insider-prone classification (Section 4.1 / Appendix C.2)
# --------------------------------------------------------------------------
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_BATCH = 25                    # questions per request (paper's Listing 1)
LLM_TEMP = 0.0


def get_anthropic_key() -> str | None:
    """Read the key lazily: env var first, then a local .anthropic_key file.
    The file lets a long-running watcher pick up a key dropped in later."""
    k = os.environ.get("ANTHROPIC_API_KEY")
    if k:
        return k.strip()
    f = ROOT / ".anthropic_key"
    if f.exists():
        return f.read_text().strip() or None
    return None


# Backwards-compatible module-level value (may be None until the key is provided).
ANTHROPIC_API_KEY = get_anthropic_key()

# --------------------------------------------------------------------------
# Calibration / longshot (Appendix E)
# --------------------------------------------------------------------------
CALIB_BINS = 50
CALIB_LIFESPAN_PCT = 0.90         # price at 90th percentile of lifespan
CALIB_MIN_VOLUME = 1000.0         # drop markets with < $1000 volume
LONGSHOT_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

# --------------------------------------------------------------------------
# HTTP behaviour
# --------------------------------------------------------------------------
HTTP_TIMEOUT = 45
HTTP_RETRIES = 9                  # retries to survive subgraph throttling
HTTP_BACKOFF = 2.0
GAMMA_PAGE = 500                  # markets per GAMMA page
SUBGRAPH_PAGE = 1000             # OrderFilledEvents per subgraph query
SUBGRAPH_WORKERS = 2             # low concurrency to avoid rate-limit blocks
PRICE_FIDELITY = 60              # CLOB prices-history: 1-minute resolution


def paper_targets() -> dict:
    """Headline numbers from the paper for automated comparison in the report."""
    return {
        "n_events": 98906,
        "n_markets": 210322,
        "n_accounts": 1_720_000,
        "total_volume_usd": 13.76e9,
        "skilled_count": 54325,
        "skilled_pct_all": 3.16,
        "antiskilled_count": 105951,
        "antiskilled_pct_all": 6.15,
        "pval_corr": 0.35,
        "skilled_to_skilled": 0.46,
    }
