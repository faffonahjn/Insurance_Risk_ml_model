.PHONY: install train predict serve test docker-build docker-up azure-deploy

install:
	pip install -r requirements.txt

train:
	python pipelines/train_pipeline.py --config configs/config.yaml

predict:
	python pipelines/predict_pipeline.py \
		--input data/raw/medical_insurance.csv \
		--output data/processed/predictions.csv

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --tb=short

docker-build:
	docker build -f docker/Dockerfile -t insurance-risk-api:latest .

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

# ── Azure Container Apps deployment ───────────────────────────────────────────
azure-deploy:
	az acr login --name mlinsuranceacr
	docker tag insurance-risk-api:latest mlinsuranceacr.azurecr.io/insurance-risk-api:latest
	docker push mlinsuranceacr.azurecr.io/insurance-risk-api:latest
	az containerapp update \
		--name insurance-risk-api \
		--resource-group rg-ml-insurance \
		--image mlinsuranceacr.azurecr.io/insurance-risk-api:latest
