.PHONY: setup ingest download index evaluate ablation serve lint format-check test check

setup:
	python -m pip install -r requirements.txt

ingest:
	python scripts/download_data.py

download: ingest

index:
	python scripts/validate_catalog.py
	python scripts/build_index.py

evaluate:
	python scripts/evaluate_dev.py
	python scripts/evaluate_final.py

ablation:
	python scripts/ablation_experiments.py

serve:
	python -m uvicorn src.api:app --host 0.0.0.0 --port 8000

lint:
	python -m ruff check --no-cache src scripts tests

format-check:
	python -m ruff format --check src scripts tests

test:
	python -m pytest -q

check: lint format-check test
