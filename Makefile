# DataPilot — convenience targets
# Always use `python -m` to stay inside the active venv.

.PHONY: app eval test data clean

## Start the Streamlit UI
app:
	python -m streamlit run ui/app.py

## Run the offline eval harness (no API key needed for 9/11 criteria)
eval:
	python evals/analyze_eval.py --skip-narrative

## Run the offline eval with LLM narrative (requires ANTHROPIC_API_KEY)
eval-full:
	python evals/analyze_eval.py

## Run the full unit test suite
test:
	python -m pytest tests/ -v

## Regenerate the DuckDB dataset from scratch
data:
	python data/generate_data.py

## Delete generated files so the next run starts clean
clean:
	rm -f data/dau_experiment.db
	rm -f memory/schema_cache.json
	rm -f memory/datapilot_memory.db
