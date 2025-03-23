# Transcription Service Kubernetes Deployment

This directory contains Kubernetes manifests for deploying the Transcription Service in a production environment. The service is designed to operate as part of a larger microservice architecture for the AI Video Processing Platform.

## Components

The deployment consists of:

1. **API Service**: Handles HTTP requests for transcription jobs
2. **Worker Service**: Processes transcription jobs asynchronously
3. **PostgreSQL Database**: Stores transcription data and job metadata
4. **RabbitMQ**: Message broker for job queuing
5. **Autoscaler**: HPA for scaling workers based on load

## Prerequisites

- Kubernetes cluster (v1.20+)
- kubectl configured to communicate with your cluster
- A container registry where you can push images
- Kubernetes Secrets containing:
  - OpenAI API Key
  - Service API Key (for service-to-service communication)
  - PostgreSQL credentials
  - RabbitMQ credentials
  - S3 credentials (for storing output files)

## Configuration

The following environment variables need to be set before deployment:

```bash
export NAMESPACE="ai-video-platform"
export REGISTRY_URL="your-registry.com"
export IMAGE_TAG="latest"
export OPENAI_API_KEY="your-openai-api-key"
export SERVICE_API_KEY="your-service-api-key"
export POSTGRES_USER="postgres-username"
export POSTGRES_PASSWORD="postgres-password"
export RABBITMQ_USER="rabbitmq-username"
export RABBITMQ_PASSWORD="rabbitmq-password"
export S3_ACCESS_KEY="your-s3-access-key"
export S3_SECRET_KEY="your-s3-secret-key"
```

## Deployment

To deploy the Transcription Service:

```bash
chmod +x deploy.sh
./deploy.sh
```

This script will:
1. Create the namespace if it doesn't exist
2. Generate and apply Kubernetes secrets
3. Apply ConfigMap with service configuration
4. Deploy PostgreSQL and RabbitMQ
5. Deploy the API and Worker services
6. Configure the HorizontalPodAutoscaler for the worker

## Monitoring

To monitor the service:

```bash
# Get the status of all pods
kubectl get pods -n $NAMESPACE

# Check API logs
kubectl logs -f deployment/transcription-api -n $NAMESPACE

# Check Worker logs
kubectl logs -f deployment/transcription-worker -n $NAMESPACE

# Check autoscaling status
kubectl get hpa transcription-worker-hpa -n $NAMESPACE
```

## Scaling

The worker deployment is configured with HorizontalPodAutoscaler to automatically scale based on CPU and memory utilization. The autoscaler will maintain between 3 and 10 worker pods depending on the load.

For the API service, you can manually scale the deployment:

```bash
kubectl scale deployment transcription-api --replicas=3 -n $NAMESPACE
```

## Troubleshooting

Common issues:

1. **Pod Stuck in Pending State**: Check if PVCs are being provisioned or if there are resource constraints.
2. **Worker Not Processing Jobs**: Verify RabbitMQ connection and queue creation.
3. **API Service Unavailable**: Check Ingress configuration and service health probe responses.

To get detailed information about pod status:

```bash
kubectl describe pod <pod-name> -n $NAMESPACE
```

## Customization

To customize the deployment:

1. Edit `configmap.yaml` to change service configuration
2. Modify `deployment.yaml` to adjust resource requests and limits
3. Update `autoscaler.yaml` to change scaling parameters 