# Operational Documentation

## Deployment Procedures

### Prerequisites

- Kubernetes cluster with GPU nodes
- Helm 3.x
- kubectl configured with cluster access
- Access to container registry
- Required secrets and configmaps

### Initial Deployment

1. **Configure Environment**
   ```bash
   # Set environment variables
   export KUBERNETES_NAMESPACE=production
   export REGISTRY_URL=registry.example.com
   export IMAGE_TAG=latest
   ```

2. **Create Namespace and Resources**
   ```bash
   # Create namespace
   kubectl create namespace $KUBERNETES_NAMESPACE

   # Apply storage configuration
   kubectl apply -f k8s/storage.yaml

   # Apply Redis configuration
   kubectl apply -f k8s/redis.yaml

   # Apply main deployment
   kubectl apply -f k8s/deployment.yaml
   ```

3. **Verify Deployment**
   ```bash
   # Check pod status
   kubectl get pods -n $KUBERNETES_NAMESPACE

   # Check service status
   kubectl get svc -n $KUBERNETES_NAMESPACE

   # Check ingress status
   kubectl get ingress -n $KUBERNETES_NAMESPACE
   ```

## Scaling Guidelines

### Horizontal Scaling

The service uses HorizontalPodAutoscaler (HPA) for automatic scaling:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: clip-generation-service
spec:
  minReplicas: 4
  maxReplicas: 12
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Custom
    custom:
      name: queue_length
      target:
        type: AverageValue
        averageValue: 100
```

### Manual Scaling

```bash
# Scale deployment
kubectl scale deployment clip-generation-service -n $KUBERNETES_NAMESPACE --replicas=8

# Scale specific components
kubectl scale deployment clip-generation-service-gpu -n $KUBERNETES_NAMESPACE --replicas=4
kubectl scale deployment clip-generation-service-cpu -n $KUBERNETES_NAMESPACE --replicas=4
```

## Backup and Recovery

### Database Backup

1. **Schedule Regular Backups**
   ```bash
   # Create backup job
   kubectl apply -f k8s/backup-job.yaml
   ```

2. **Verify Backup Storage**
   ```bash
   # Check backup storage
   kubectl exec -it postgres-shard-0-0 -n $KUBERNETES_NAMESPACE -- pg_dump -U postgres > backup.sql
   ```

### Disaster Recovery

1. **Recovery Procedures**
   ```bash
   # Restore from backup
   kubectl exec -it postgres-shard-0-0 -n $KUBERNETES_NAMESPACE -- psql -U postgres < backup.sql

   # Verify data integrity
   kubectl exec -it postgres-shard-0-0 -n $KUBERNETES_NAMESPACE -- psql -U postgres -c "SELECT COUNT(*) FROM jobs;"
   ```

## Monitoring and Alerts

### Prometheus Configuration

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: clip-generation-service
spec:
  selector:
    matchLabels:
      app: clip-generation-service
  endpoints:
  - port: metrics
    interval: 15s
```

### Alert Rules

```yaml
groups:
- name: clip-generation-service
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: High error rate detected
      description: Error rate is above 10% for 5 minutes
```

## Maintenance Procedures

### Regular Maintenance

1. **Weekly Tasks**
   - Review system logs
   - Check resource utilization
   - Verify backup integrity
   - Update monitoring dashboards

2. **Monthly Tasks**
   - Review and update security patches
   - Analyze performance metrics
   - Clean up old data
   - Update documentation

### Performance Tuning

1. **Resource Optimization**
   ```bash
   # Monitor resource usage
   kubectl top pods -n $KUBERNETES_NAMESPACE

   # Adjust resource limits
   kubectl patch deployment clip-generation-service -n $KUBERNETES_NAMESPACE --patch '{
     "spec": {
       "template": {
         "spec": {
           "containers": [{
             "name": "api-server",
             "resources": {
               "limits": {
                 "cpu": "16",
                 "memory": "32Gi"
               }
             }
           }]
         }
       }
     }
   }'
   ```

2. **Cache Optimization**
   ```bash
   # Monitor Redis metrics
   kubectl exec -it redis-cache-0 -n $KUBERNETES_NAMESPACE -- redis-cli info

   # Adjust Redis configuration
   kubectl patch configmap redis-config -n $KUBERNETES_NAMESPACE --patch '{
     "data": {
       "redis.conf": "maxmemory 16gb\nmaxmemory-policy allkeys-lru\n..."
     }
   }'
   ```

## Incident Response

### Common Issues

1. **High Error Rate**
   - Check application logs
   - Verify resource limits
   - Review recent changes
   - Scale resources if needed

2. **Performance Degradation**
   - Monitor resource usage
   - Check cache hit rates
   - Review database performance
   - Analyze network latency

3. **Storage Issues**
   - Verify storage capacity
   - Check I/O performance
   - Review backup status
   - Clean up old data

### Escalation Procedures

1. **Level 1: On-Call Engineer**
   - Initial triage
   - Basic troubleshooting
   - Service restoration

2. **Level 2: Senior Engineer**
   - Complex issues
   - Performance optimization
   - Architecture changes

3. **Level 3: Engineering Lead**
   - Critical incidents
   - Major outages
   - Strategic decisions

## Security Procedures

### Access Management

1. **API Keys**
   ```bash
   # Generate new API key
   kubectl exec -it clip-generation-service-0 -n $KUBERNETES_NAMESPACE -- python manage.py generate_api_key

   # Revoke API key
   kubectl exec -it clip-generation-service-0 -n $KUBERNETES_NAMESPACE -- python manage.py revoke_api_key <key_id>
   ```

2. **User Access**
   ```bash
   # Add user
   kubectl exec -it clip-generation-service-0 -n $KUBERNETES_NAMESPACE -- python manage.py create_user

   # Update permissions
   kubectl exec -it clip-generation-service-0 -n $KUBERNETES_NAMESPACE -- python manage.py update_permissions
   ```

### Security Monitoring

1. **Log Analysis**
   ```bash
   # Check security logs
   kubectl logs -f clip-generation-service-0 -n $KUBERNETES_NAMESPACE | grep "security"

   # Monitor failed attempts
   kubectl exec -it clip-generation-service-0 -n $KUBERNETES_NAMESPACE -- python manage.py check_security_logs
   ```

2. **Vulnerability Scanning**
   ```bash
   # Run security scan
   kubectl exec -it clip-generation-service-0 -n $KUBERNETES_NAMESPACE -- python manage.py security_scan

   # Update dependencies
   kubectl exec -it clip-generation-service-0 -n $KUBERNETES_NAMESPACE -- python manage.py update_dependencies
   ``` 