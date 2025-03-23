#!/bin/bash
set -e

# Configuration variables
NAMESPACE=${NAMESPACE:-"ai-video-platform"}
REGISTRY_URL=${REGISTRY_URL:-"your-registry.com"}
IMAGE_TAG=${IMAGE_TAG:-"latest"}

# Base64 encode secrets (ensure these env vars are set)
if [ -z "$OPENAI_API_KEY" ] || [ -z "$SERVICE_API_KEY" ] || [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ] || [ -z "$RABBITMQ_USER" ] || [ -z "$RABBITMQ_PASSWORD" ] || [ -z "$S3_ACCESS_KEY" ] || [ -z "$S3_SECRET_KEY" ]; then
  echo "Error: Required environment variables are not set. Please ensure the following are defined:"
  echo "OPENAI_API_KEY, SERVICE_API_KEY, POSTGRES_USER, POSTGRES_PASSWORD, RABBITMQ_USER, RABBITMQ_PASSWORD, S3_ACCESS_KEY, S3_SECRET_KEY"
  exit 1
fi

# Base64 encode secrets
OPENAI_API_KEY_BASE64=$(echo -n "$OPENAI_API_KEY" | base64)
SERVICE_API_KEY_BASE64=$(echo -n "$SERVICE_API_KEY" | base64)
POSTGRES_USER_BASE64=$(echo -n "$POSTGRES_USER" | base64)
POSTGRES_PASSWORD_BASE64=$(echo -n "$POSTGRES_PASSWORD" | base64)
RABBITMQ_USER_BASE64=$(echo -n "$RABBITMQ_USER" | base64)
RABBITMQ_PASSWORD_BASE64=$(echo -n "$RABBITMQ_PASSWORD" | base64)
S3_ACCESS_KEY_BASE64=$(echo -n "$S3_ACCESS_KEY" | base64)
S3_SECRET_KEY_BASE64=$(echo -n "$S3_SECRET_KEY" | base64)

# Create namespace if it doesn't exist
kubectl get namespace $NAMESPACE > /dev/null 2>&1 || kubectl create namespace $NAMESPACE

# Generate secrets from template
echo "Generating secrets..."
cat secrets-template.yaml | \
  sed "s/\${OPENAI_API_KEY_BASE64}/$OPENAI_API_KEY_BASE64/g" | \
  sed "s/\${SERVICE_API_KEY_BASE64}/$SERVICE_API_KEY_BASE64/g" | \
  sed "s/\${POSTGRES_USER_BASE64}/$POSTGRES_USER_BASE64/g" | \
  sed "s/\${POSTGRES_PASSWORD_BASE64}/$POSTGRES_PASSWORD_BASE64/g" | \
  sed "s/\${RABBITMQ_USER_BASE64}/$RABBITMQ_USER_BASE64/g" | \
  sed "s/\${RABBITMQ_PASSWORD_BASE64}/$RABBITMQ_PASSWORD_BASE64/g" | \
  sed "s/\${S3_ACCESS_KEY_BASE64}/$S3_ACCESS_KEY_BASE64/g" | \
  sed "s/\${S3_SECRET_KEY_BASE64}/$S3_SECRET_KEY_BASE64/g" > secrets.yaml

# Process deployment templates
echo "Processing deployment templates..."
cat deployment.yaml | \
  sed "s/\${REGISTRY_URL}/$REGISTRY_URL/g" | \
  sed "s/\${IMAGE_TAG}/$IMAGE_TAG/g" > deployment-processed.yaml

# Apply ConfigMap
echo "Applying ConfigMap..."
kubectl apply -f configmap.yaml -n $NAMESPACE

# Apply Secrets
echo "Applying Secrets..."
kubectl apply -f secrets.yaml -n $NAMESPACE

# Apply database resources
echo "Applying database and message queue resources..."
kubectl apply -f database.yaml -n $NAMESPACE

# Apply deployment resources
echo "Applying service deployment..."
kubectl apply -f deployment-processed.yaml -n $NAMESPACE

echo "Deployment completed successfully!"
echo "To check status, run: kubectl get pods -n $NAMESPACE"
echo "To view API service logs, run: kubectl logs -f deployment/transcription-api -n $NAMESPACE"
echo "To view worker logs, run: kubectl logs -f deployment/transcription-worker -n $NAMESPACE"

# Clean up processed files
rm -f secrets.yaml deployment-processed.yaml

echo "Done!" 