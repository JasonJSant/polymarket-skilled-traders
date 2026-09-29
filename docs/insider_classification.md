# Insider-prone market classification (Module 4)

Runbook for the LLM step that labels each market as insider-prone. These labels
feed the insider tables (Tables 6, 7, C.2) and the record/timing/conviction
proxies.

## What it does

Each market question is labeled `yes`, `unclear`, or `no` using the paper's
rubric (Appendix C.2). A market is "insider-prone" if the label is `yes` or
`unclear`. This is a single-call classification task (no tools, no agents) over
the unique market questions.

## Components

| File | Purpose |
| --- | --- |
| `config.py` (LLM section) | `LLM_MODEL`, `LLM_BATCH`, `LLM_TEMP`, and `get_anthropic_key()` (reads the `ANTHROPIC_API_KEY` environment variable or a `.anthropic_key` file) |
| `src/insider/llm_prompt.py` | The system prompt (rubric plus labeled examples) and the per-batch user template, transcribed from the paper |
| `src/insider/llm_classify.py` | Reads the market questions, de-duplicates them, batches them (25 per call at temperature 0), parses the JSON output, caches at the question level (resumable), and writes `data/processed/insider_labels.parquet` |
| `src/insider/proxies.py`, `src/insider/insider_regressions.py` | Consume `insider_labels.parquet` to produce the proxies and Tables 6/7 |

## Setup

Provide an Anthropic API key, either as an environment variable or in a file at
the project root:

```sh
export ANTHROPIC_API_KEY=...
# or
echo '...' > .anthropic_key   # gitignored
```

## Running

Run modules from the project root using the project environment.

```sh
python -m src.insider.llm_classify              # full run
python -m src.insider.llm_classify --limit 100  # small test
```

The classifier checkpoints periodically and skips already-labeled questions, so
it is safe to interrupt and re-run.

## Sizing

- About 244k markets reduce to roughly 219k unique question texts.
- At 25 questions per call that is about 8,800 API calls.
- Per call: system prompt about 3,000 to 3,600 tokens (rubric plus examples),
  user message about 400 tokens, output about 300 tokens.
- Estimated cost on Haiku 4.5 is roughly $20 to $45 depending on the run path;
  output tokens dominate.

Caching note: Haiku 4.5's minimum cacheable prefix is 4096 tokens, and the system
prompt is just under that, so prompt caching does not apply here.

Optional faster path: wrap the batch loop in a thread pool (8 to 16 workers), or
submit through the Batch API (about 50% cheaper). The prompt, model, batching, and
JSON parsing are reusable; only the transport changes.

## After the labels exist

```sh
python run_analysis.py     # picks up insider_labels.parquet automatically
```

This adds Table 6 (price discovery split by insider-prone vs not), Table 7
(insider accounts and price impact), the Table C.2 distribution, and the
record/timing/conviction proxy counts.

## Validation targets (from the paper)

- Label mix around 10.6% `yes`, 0.4% `unclear`, 89% `no`. Sports and crypto
  markets are almost entirely `no`.
- Proxy counts around: record 757, timing 297, conviction 124, union 967.
- About 1% of flagged insider accounts are also classified skilled (the paper's
  point that skilled traders are not insiders).
