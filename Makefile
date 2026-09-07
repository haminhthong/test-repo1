setup:
	python -m pip install -r requirements.txt

ingest:
	python scripts/download_data.py

download: ingest

index:
	python scripts/validate_catalog.py
	python scripts/build_index.py

train: index

evaluate:
	python scripts/evaluate_dev.py
	python scripts/evaluate_final.py

ablation:
	python scripts/ablation_experiments.py

serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

test:
	pytest -v
