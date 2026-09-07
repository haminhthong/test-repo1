setup:
	python -m pip install -r requirements.txt

ingest:
	python scripts/download_data.py

download: ingest

index:
	python -m src.index

train: index

evaluate:
	python -m src.evaluate

ablation:
	python scripts/ablation_experiments.py

serve:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

test:
	pytest -v
