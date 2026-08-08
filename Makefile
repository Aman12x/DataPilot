# DataPilot — convenience targets
# Prefers ./venv/bin/python when present; override with PYTHON=python make eval
#
# The venv is not a style preference. The system interpreter is missing duckdb
# and jose, so 12 test files fail to collect under it and data/generate_data.py
# cannot run at all.

PYTHON ?= $(shell [ -x ./venv/bin/python ] && echo ./venv/bin/python || echo python3)

.PHONY: eval eval-all eval-full eval-baseline test test-fast test-all data clean e2e

## Run all fast offline evals (no API key)
eval:
	$(PYTHON) data/generate_data.py
	$(PYTHON) evals/analyze_eval.py --skip-narrative
	$(PYTHON) evals/generalisability_eval.py
	$(PYTHON) evals/transactions_eval.py
	$(PYTHON) evals/fixture_eval.py

## Run all offline evals + baseline regression gate
eval-all:
	$(PYTHON) evals/compare_baseline.py

## Update committed baseline scores (run after intentional eval improvements)
eval-baseline:
	$(PYTHON) evals/compare_baseline.py --update

## Run the offline eval with LLM narrative (requires ANTHROPIC_API_KEY)
eval-full:
	$(PYTHON) data/generate_data.py
	$(PYTHON) evals/analyze_eval.py

## Run exactly what CI runs (ci.yml test-backend). ~5 min. Use before pushing.
test:
	$(PYTHON) -m pytest tests/ -q -m "not integration" --tb=short --disable-warnings

## Fast inner-loop subset. ~1.5 min. Skips the 11 slow tests that CI DOES run,
## so a green run here does not mean CI is green — see `make test`.
test-fast:
	$(PYTHON) -m pytest tests/ -q -m "not integration and not slow" --tb=short

## Everything, including integration tests (needs live Redis + Postgres)
test-all:
	$(PYTHON) -m pytest tests/ -q

## Run Playwright E2E (requires backend deps + frontend npm ci)
e2e:
	cd frontend && npm run test:e2e

## Regenerate the DuckDB dataset from scratch
data:
	$(PYTHON) data/generate_data.py

## Delete generated files so the next run starts clean
clean:
	rm -f data/dau_experiment.db
	rm -f memory/schema_cache.json
	rm -f memory/datapilot_memory.db
