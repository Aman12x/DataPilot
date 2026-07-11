# DataPilot — convenience targets
# Uses python3 by default; override with PYTHON=python make eval

PYTHON ?= python3

.PHONY: app eval eval-all eval-full eval-baseline test data clean e2e

## Start the Streamlit UI
app:
	$(PYTHON) -m streamlit run ui/app.py

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

## Run the full unit test suite
test:
	$(PYTHON) -m pytest tests/ -v

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
