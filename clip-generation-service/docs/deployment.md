# Deployment Guide

## Prerequisites

- Python 3.8+
- FFmpeg
- PostgreSQL (for production)
- Redis (optional, for caching)
- Docker (optional, for containerized deployment)

## Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/clip-generation-service.git
cd clip-generation-service
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Initialize database:
```bash
python -m src.database.init_db
alembic upgrade head
```

6. Run the service:
```bash
uvicorn src.main:app --reload
```

## Production Deployment

### Using Docker

1. Build the Docker image:
```bash
docker build -t clip-generation-service .
```

2. Run the container:
```bash
docker run -d \
  --name clip-generation-service \
  -p 8000:8000 \
  -v /path/to/storage:/app/storage \
  -e APP_ENV=production \
  -e DB_URL=postgresql://user:password@host:5432/dbname \
  clip-generation-service
```

### Manual Deployment

1. Set up the server:
```bash
# Install system dependencies
sudo apt update
sudo apt install python3.8 python3.8-venv ffmpeg postgresql redis-server

# Create service user
sudo useradd -m -s /bin/bash clip-service
```

2. Deploy the application:
```bash
# Clone repository
sudo -u clip-service git clone https://github.com/yourusername/clip-generation-service.git /home/clip-service/app

# Set up virtual environment
cd /home/clip-service/app
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with production settings
```

3. Set up systemd service:
```bash
sudo nano /etc/systemd/system/clip-generation.service
```

Add the following content:
```ini
[Unit]
Description=Clip Generation Service
After=network.target postgresql.service redis-server.service

[Service]
User=clip-service
Group=clip-service
WorkingDirectory=/home/clip-service/app
Environment="PATH=/home/clip-service/app/venv/bin"
ExecStart=/home/clip-service/app/venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

4. Start the service:
```bash
sudo systemctl enable clip-generation
sudo systemctl start clip-generation
```

### Nginx Configuration

1. Install Nginx:
```bash
sudo apt install nginx
```

2. Create Nginx configuration:
```bash
sudo nano /etc/nginx/sites-available/clip-generation
```

Add the following content:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # SSL configuration (recommended)
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
}
```

3. Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/clip-generation /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## Monitoring

### Prometheus Metrics

The service exposes metrics at `/metrics` endpoint. Configure Prometheus to scrape these metrics:

```yaml
scrape_configs:
  - job_name: 'clip-generation'
    static_configs:
      - targets: ['localhost:8000']
```

### Health Checks

The service provides health check endpoints:
- `/health` - Basic health check
- `/health/detailed` - Detailed health status

### Logging

Logs are written to:
- Application logs: `/var/log/clip-generation/app.log`
- Access logs: `/var/log/clip-generation/access.log`
- Error logs: `/var/log/clip-generation/error.log`

## Backup and Recovery

### Database Backup

1. Create a backup script:
```bash
#!/bin/bash
BACKUP_DIR="/path/to/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Backup database
pg_dump -U postgres clip_generation > "$BACKUP_DIR/clip_generation_$TIMESTAMP.sql"

# Backup storage
tar -czf "$BACKUP_DIR/storage_$TIMESTAMP.tar.gz" /path/to/storage

# Clean up old backups (keep last 7 days)
find $BACKUP_DIR -type f -mtime +7 -delete
```

2. Set up cron job:
```bash
0 2 * * * /path/to/backup.sh
```

### Recovery

1. Restore database:
```bash
psql -U postgres clip_generation < backup.sql
```

2. Restore storage:
```bash
tar -xzf storage_backup.tar.gz -C /path/to/storage
```

## Scaling

### Horizontal Scaling

1. Set up a load balancer (e.g., HAProxy):
```haproxy
frontend clip-generation
    bind *:80
    mode http
    default_backend clip-generation-backend

backend clip-generation-backend
    balance roundrobin
    server server1 localhost:8001 check
    server server2 localhost:8002 check
    server server3 localhost:8003 check
```

2. Configure multiple instances:
```bash
# Instance 1
uvicorn src.main:app --host 0.0.0.0 --port 8001 --workers 4

# Instance 2
uvicorn src.main:app --host 0.0.0.0 --port 8002 --workers 4

# Instance 3
uvicorn src.main:app --host 0.0.0.0 --port 8003 --workers 4
```

### Vertical Scaling

1. Increase worker count:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 8
```

2. Adjust system limits:
```bash
# Edit /etc/sysctl.conf
fs.file-max = 65535
vm.swappiness = 10
```

## Security

### SSL/TLS

1. Obtain SSL certificate:
```bash
sudo certbot --nginx -d your-domain.com
```

2. Configure automatic renewal:
```bash
sudo certbot renew --dry-run
```

### Firewall

1. Configure UFW:
```bash
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Rate Limiting

1. Configure Nginx rate limiting:
```nginx
limit_req_zone $binary_remote_addr zone=clip_limit:10m rate=10r/s;

location / {
    limit_req zone=clip_limit burst=20 nodelay;
    # ... existing configuration ...
}
```

## Maintenance

### Updates

1. Pull latest changes:
```bash
cd /home/clip-service/app
git pull
```

2. Update dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

3. Run migrations:
```bash
alembic upgrade head
```

4. Restart service:
```bash
sudo systemctl restart clip-generation
```

### Cleanup

1. Set up storage cleanup:
```bash
#!/bin/bash
STORAGE_DIR="/path/to/storage"
MAX_AGE=30  # days

# Clean up old files
find $STORAGE_DIR -type f -mtime +$MAX_AGE -delete
find $STORAGE_DIR -type d -empty -delete
```

2. Schedule cleanup:
```bash
0 3 * * * /path/to/cleanup.sh
``` 