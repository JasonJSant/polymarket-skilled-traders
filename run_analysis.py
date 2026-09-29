"""End-to-end analysis orchestrator: run after the data collection completes.

Runs the full pipeline in dependency order and regenerates every table, figure,
and the comparison report. Each stage reads/writes parquet in data/processed, so
stages can also be run individually via `python -m src.<module>`.

    python run_analysis.py                 # everything (skips LLM if no API key)
    python run_analysis.py --skip-insider  # omit Module 4
"""
from __future__ import annotations

import argparse
import time

import config as C
from src.build import event_pnl, time_bins
from src.skill import sign_randomization, classify, persistence
from src.pricediscovery import regressions
from src.strategies import strategies, news
from src.insider import case_study
from src.report import tables, figures, compare


def _run(label, fn):
    t = time.time()
    print(f"\n===== {label} =====", flush=True)
    fn()
    print(f"[{label} done in {time.time()-t:.0f}s]", flush=True)


def main(skip_insider: bool) -> None:
    _run("event PnL", event_pnl.build)
    _run("sign-randomization skill test", sign_randomization.build)
    _run("classification", classify.build)
    _run("persistence (train/test)", persistence.build)
    _run("time bins", time_bins.build)
    _run("price-discovery regressions", regressions.build)
    _run("strategies", strategies.build)
    _run("news reaction (Table 8)", news.build)
    _run("Maduro case study", case_study.build)

    if not skip_insider and (C.PROC / "insider_labels.parquet").exists():
        from src.insider import proxies, insider_regressions
        _run("insider proxies", proxies.build)
        _run("insider price discovery (Tables 6/E.6/E.7)", insider_regressions.build)
    else:
        print("\n[insider LLM tables skipped: set ANTHROPIC_API_KEY and run "
              "`python -m src.insider.llm_classify` first]")

    _run("tables", tables.build)
    _run("figures", figures.build)
    _run("comparison report", compare.build)
    print("\nAll done. See outputs/REPORT.md, outputs/tables/, outputs/figures/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-insider", action="store_true")
    main(ap.parse_args().skip_insider)
