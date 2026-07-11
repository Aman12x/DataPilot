# DataPilot offline evals

Deterministic evaluation harnesses that catch wrong statistics, missing insights, and unsafe recommendations **before** they reach users. No LLM-as-judge in the regression gate — scores run in milliseconds with zero API cost.

## Quick start

```bash
make eval          # run all fast harnesses (no API key)
make eval-all      # harnesses + baseline regression check (same as CI)
make eval-baseline # refresh evals/baseline.json after intentional improvements
make eval-full     # DAU eval including LLM narrative (needs ANTHROPIC_API_KEY)
```

## Harnesses

| Script | Pass bar | API key | What it checks |
|--------|----------|---------|----------------|
| `analyze_eval.py` | ≥80% | Optional | Full DAU experiment tool chain: HTE segment, CUPED, t-test, guardrails, decomposition, forecast |
| `generalisability_eval.py` | ≥80% | No | Clinical trial + ecommerce A/B on real sample CSVs |
| `transactions_eval.py` | ≥80% | No | 15 golden Q&A answers on `customer_transactions_10k.csv` + faithfulness |
| `fixture_eval.py` | ≥80% | No | Keyword + faithfulness scoring on `tests/fixtures/` CSVs |

## Regression gate

`compare_baseline.py` runs all four harnesses and fails if any score drops more than 2% below the committed `baseline.json`. CI runs this on every push via the `eval-offline` job.

```bash
python evals/compare_baseline.py          # check against baseline
python evals/compare_baseline.py --update  # write new baseline
```

## Eval pyramid

```
Every PR (fast, deterministic)
├── pytest unit tests (578) — tool math, ground-truth JSON, claim scorers
├── eval-offline — 4 harnesses + baseline regression
├── build-frontend + Playwright E2E
└── gitleaks

Nightly (optional LLM)
├── make eval-full — LLM narrative on DAU scenario
└── pytest -m slow — MiniLM relevancy tests

Production (continuous)
└── _compute_quality_score — faithfulness, claim accuracy, safety constraints per run
```

## Current baseline (commit `baseline.json`)

| Harness | Score |
|---------|-------|
| DAU experiment | 12/13 (92%) |
| Cross-domain | 13/13 (100%) |
| Transactions Q&A | 7/7 (100%) |
| Fixtures | 4/4 (100%) |

The one DAU miss without an API key is `narrative_relevant` (MiniLM embedding) — it passes in `make eval-full` when `ANTHROPIC_API_KEY` is set.

## Adding a new eval

1. Add ground truth to `data/samples/*_ground_truth.json` or extend `FIXTURE_GROUND_TRUTH` in `tools/eval_tools.py`.
2. Create `evals/your_eval.py` with `--json` output matching the schema in `compare_baseline.py`.
3. Register it in `EVAL_COMMANDS` inside `compare_baseline.py`.
4. Run `make eval-baseline` to update the committed baseline.
