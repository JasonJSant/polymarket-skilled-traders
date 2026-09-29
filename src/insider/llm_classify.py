"""Classify every market as insider-prone via Claude Haiku (Appendix C.2).

Runs the exact Listing-1 prompt at temperature 0 in batches of 25 questions,
caching results per market so it is fully resumable and re-runnable. A market is
insider-prone if the model answers "yes" or "unclear".

Requires ANTHROPIC_API_KEY in the environment. Output:
  data/processed/insider_labels.parquet  (market_id, condition_id, question, label)
"""
from __future__ import annotations

import argparse
import json

import pandas as pd
from tqdm import tqdm

import config as C
from src.insider.llm_prompt import SYSTEM_PROMPT, USER_TEMPLATE

CACHE = C.PROC / "insider_labels.parquet"


def _load_cache() -> dict[str, str]:
    if CACHE.exists():
        df = pd.read_parquet(CACHE)
        return dict(zip(df["market_id"].astype(str), df["label"]))
    return {}


def _save(labels: dict[str, str], markets: pd.DataFrame) -> None:
    df = markets[["market_id", "condition_id", "question"]].copy()
    df["market_id"] = df["market_id"].astype(str)
    df["label"] = df["market_id"].map(labels)
    df = df.dropna(subset=["label"])
    df.to_parquet(CACHE, index=False)


def classify(batch_questions: list[str], client) -> list[str]:
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(batch_questions))
    msg = client.messages.create(
        model=C.LLM_MODEL,
        max_tokens=2000,
        temperature=C.LLM_TEMP,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(numbered=numbered)}],
    )
    text = msg.content[0].text.strip()
    if text.startswith("```"):
        text = text.strip("`").split("\n", 1)[-1].rsplit("```", 1)[0]
    arr = json.loads(text)
    out = ["no"] * len(batch_questions)
    for obj in arr:
        i = int(obj["i"]) - 1
        if 0 <= i < len(out):
            lab = str(obj["label"]).lower().strip()
            out[i] = lab if lab in ("yes", "no", "unclear") else "no"
    return out


_QCACHE = C.PROC / "insider_qlabels.json"


def build(limit: int | None = None) -> None:
    key = C.get_anthropic_key()
    if not key:
        raise SystemExit("Set ANTHROPIC_API_KEY (or write .anthropic_key) to run "
                         "the LLM classification.")
    import anthropic
    client = anthropic.Anthropic(api_key=key)

    markets = pd.read_parquet(C.RAW / "markets.parquet",
                              columns=["market_id", "condition_id", "question"])
    markets["market_id"] = markets["market_id"].astype(str)

    # question-level cache (dedupe identical question text to save cost + resume)
    qlabel: dict[str, str] = json.loads(_QCACHE.read_text()) if _QCACHE.exists() else {}
    unique_q = markets["question"].dropna().drop_duplicates().tolist()
    todo_q = [q for q in unique_q if q not in qlabel]
    if limit:
        todo_q = todo_q[:limit]
    print(f"unique questions: {len(unique_q):,} | to classify now: {len(todo_q):,}")

    for i in tqdm(range(0, len(todo_q), C.LLM_BATCH), desc="LLM classify"):
        batch = todo_q[i:i + C.LLM_BATCH]
        try:
            res = classify(batch, client)
        except Exception as e:  # noqa: BLE001
            print(f"batch {i} failed: {str(e)[:120]}")
            continue
        for q, lab in zip(batch, res):
            qlabel[q] = lab
        if (i // C.LLM_BATCH) % 20 == 0:
            _QCACHE.write_text(json.dumps(qlabel))
    _QCACHE.write_text(json.dumps(qlabel))

    labels = {str(r.market_id): qlabel.get(r.question, "no")
              for r in markets.itertuples(index=False)}
    _save(labels, markets)

    lab = pd.Series(labels)
    print("label distribution:", lab.value_counts(normalize=True).round(3).to_dict())
    print(f"insider-prone (yes|unclear): {(lab.isin(['yes','unclear'])).mean()*100:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    build(ap.parse_args().limit)
