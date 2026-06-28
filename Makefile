# DataPilot — convenience targets
# Always use `python -m` to stay inside the active venv.

.PHONY: app eval eval-all eval-full eval-baseline test data clean

## Start the Streamlit UI
app:
	python -m streamlit run ui/app.py

## Run all fast offline evals (no API key)
eval:
	python data/generate_data.py
	python evals/analyze_eval.py --skip-narrative
	python evals/generalisability_eval.py
	python evals/transactions_eval.py
	python evals/fixture_eval.py

## Run all offline evals + baseline regression gate
eval-all:
	python evals/compare_baseline.py

## Update committed baseline scores (run after intentional eval improvements)
eval-baseline:
	python evals/compare_baseline.py --update

## Run the offline eval with LLM narrative (requires ANTHROPIC_API_KEY)
eval-full:
	python evals/analyze_eval.py

## Run the full unit test suite
test:
	python -m pytest tests/ -v

## Run Playwright E2E (requires backend deps + frontend npm ci)
e2e:
	cd frontend && npm run test:e2e

## Regenerate the DuckDB dataset from scratch
data:
	python data/generate_data.py

## Delete generated files so the next run starts clean
clean:
	rm -f data/dau_experiment.db
	rm -f memory/schema_cache.json
	rm -f memory/datapilot_memory.db
