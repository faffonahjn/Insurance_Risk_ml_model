#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_azure.sh
# Provisions Azure infrastructure for the Insurance Risk ML API
# Prerequisites: az CLI installed and logged in (az login)
# Usage: bash scripts/setup_azure.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RESOURCE_GROUP="rg-ml-insurance"
LOCATION="eastus"
ACR_NAME="mlinsuranceacr"
APP_ENV="ml-insurance-env"
APP_NAME="insurance-risk-api"
IMAGE="$ACR_NAME.azurecr.io/insurance-risk-api:latest"

echo "==> Creating Resource Group: $RESOURCE_GROUP"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "==> Creating Azure Container Registry: $ACR_NAME"
az acr create --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" --sku Basic --admin-enabled true

echo "==> Building and pushing Docker image"
az acr login --name "$ACR_NAME"
docker build -f docker/Dockerfile -t insurance-risk-api:latest .
docker tag insurance-risk-api:latest "$IMAGE"
docker push "$IMAGE"

echo "==> Creating Container Apps Environment"
az containerapp env create \
  --name "$APP_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

echo "==> Deploying Container App"
az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$APP_ENV" \
  --image "$IMAGE" \
  --registry-server "$ACR_NAME.azurecr.io" \
  --registry-username "$(az acr credential show -n $ACR_NAME --query username -o tsv)" \
  --registry-password "$(az acr credential show -n $ACR_NAME --query passwords[0].value -o tsv)" \
  --target-port 8000 \
  --ingress external \
  --min-replicas 1 \
  --max-replicas 3 \
  --cpu 1.0 \
  --memory 2.0Gi

echo ""
echo "==> Deployment complete."
APP_URL=$(az containerapp show --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn -o tsv)
echo "API URL: https://$APP_URL"
echo "Health:  https://$APP_URL/health"
echo "Docs:    https://$APP_URL/docs"
