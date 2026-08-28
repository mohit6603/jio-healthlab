# Deployment

Two stages: a single EC2 host that runs today, and the path to ECS when one
host stops being enough.

---

## Stage 1 — single EC2 host

```mermaid
flowchart TB
    User[Browser] -->|HTTPS| R53[Route 53]
    R53 --> EC2

    subgraph EC2["EC2 t3.medium — Ubuntu 24.04"]
      NG[host nginx<br/>TLS · rate limits]
      subgraph Compose["docker compose"]
        WEB[web] --> API[api]
        API --> AIS[ai-service]
        AIS --> BOOT[bootstrap<br/>one-shot]
      end
      NG --> WEB
    end

    API --> RDS[(RDS MySQL 8.4<br/>db.t4g.micro)]
    AIS --> QC[(Qdrant Cloud)]
    EC2 -.logs.-> CW[CloudWatch]
    EC2 -.pull.-> ECR[(ECR)]
    API -.read.-> SM[Secrets Manager]
```

### Sizing, and why not the free tier

`t2.micro` has **1 GB of RAM**. The AI service alone needs ~1.5 GB with the
embedding model resident and more during generation. The stack will not fit.

**`t3.medium` (2 vCPU, 4 GB) is the realistic minimum.** With
`LLM_PROVIDER=none` — retrieval, search and ML prediction, no generation —
`t3.small` (2 GB) works. RDS `db.t4g.micro` is genuinely adequate.

### Provision

```bash
# 1. RDS MySQL 8.4, private subnet, security group open only to the EC2 SG.
# 2. EC2 t3.medium, Ubuntu 24.04, 30 GB gp3 (model weights need ~4 GB).
# 3. Attach an instance role with: AmazonSSMManagedInstanceCore,
#    AmazonEC2ContainerRegistryReadOnly, CloudWatchAgentServerPolicy,
#    and secretsmanager:GetSecretValue on your secret only.
# 4. Security group: 443 and 80 from 0.0.0.0/0. No port 22 — use SSM.
```

### Host setup

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl nginx
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo tee /etc/apt/keyrings/docker.asc > /dev/null
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo $VERSION_CODENAME) stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update && sudo apt-get install -y \
  docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker ubuntu

sudo git clone https://github.com/mohit6603/jio-healthlab.git /opt/jio-healthlab
sudo chown -R ubuntu:ubuntu /opt/jio-healthlab
```

### Secrets

Never in the repository, never in the compose file.

```bash
aws secretsmanager create-secret --name jio-healthlab/production \
  --secret-string "$(cat <<JSON
{
  "JWT_SECRET_KEY": "$(python3 -c 'import secrets;print(secrets.token_urlsafe(48))')",
  "SEED_ADMIN_PASSWORD": "$(python3 -c 'import secrets;print(secrets.token_urlsafe(18))')",
  "DATABASE_URL": "mysql+pymysql://admin:PASSWORD@your-rds.rds.amazonaws.com:3306/jio_healthlab",
  "QDRANT_URL": "https://your-cluster.qdrant.io:6333",
  "QDRANT_API_KEY": "your-qdrant-key"
}
JSON
)"
```

On the host, materialise `.env` at boot from the secret:

```bash
cat > /opt/jio-healthlab/refresh-env.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
aws secretsmanager get-secret-value \
  --secret-id jio-healthlab/production --query SecretString --output text \
  | python3 -c 'import json,sys;[print(f"{k}={v}") for k,v in json.load(sys.stdin).items()]' \
  > /opt/jio-healthlab/.env
cat >> /opt/jio-healthlab/.env <<'EXTRA'
ENVIRONMENT=production
BACKEND_CORS_ORIGINS=https://healthlab.example.com
VITE_API_BASE_URL=
LLM_PROVIDER=local
LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
EXTRA
chmod 600 /opt/jio-healthlab/.env
SH
chmod +x /opt/jio-healthlab/refresh-env.sh
```

### Start

```bash
cd /opt/jio-healthlab
./refresh-env.sh
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

sudo cp deploy/nginx/healthlab.conf /etc/nginx/sites-available/healthlab
sudo ln -sf /etc/nginx/sites-available/healthlab /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx

sudo snap install --classic certbot
sudo certbot --nginx -d healthlab.example.com
```

First start downloads ~1.1 GB of model weights into the `huggingface-cache`
volume and runs the bootstrap. The containers report healthy before that
finishes; generation returns 503 until the weights land.

### Verify

```bash
curl -fsS https://healthlab.example.com/health
docker compose -f docker-compose.yml -f docker-compose.prod.yml ps
docker compose logs bootstrap --tail 20
```

Change the seeded administrator password immediately.

### CloudWatch

Both services already emit one JSON object per log line, so no parsing rules
are needed:

```bash
sudo wget -qO /tmp/cw.deb https://amazoncloudwatch-agent.s3.amazonaws.com/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i /tmp/cw.deb
```

Point the agent at the Docker JSON log files and set
`log_group_name = /jio-healthlab/production`. Useful metric filters:
`$.level = "ERROR"`, `$.message = "ai_error"`, `$.latency_ms > 30000`.

`GET /metrics` exposes Prometheus text for a scraper or an OTEL collector.

---

## Stage 2 — ECS, when one host is not enough

Do not start here. Move when a specific limit is hit: the backend needs more
than one replica, generation needs a GPU, or a single-host restart becomes an
unacceptable outage.

```mermaid
flowchart TB
    U[Browser] --> R53[Route 53] --> CF[CloudFront<br/>static bundle + edge cache]
    CF --> ALB[Application Load Balancer]

    subgraph ECS["ECS Fargate"]
      direction LR
      A1[api ×N<br/>I/O bound, cheap]
      A2[ai-service ×M<br/>CPU or GPU bound]
    end

    ALB --> A1
    A1 --> A2
    A1 --> RDS[(RDS Multi-AZ)]
    A2 --> QC[(Qdrant Cloud)]
    A2 --> S3[(S3<br/>knowledge documents)]
    ECS -.-> ECR[(ECR)]
    ECS -.-> SM[Secrets Manager]
    ECS -.-> CW[CloudWatch]
```

### What maps directly

| Local | ECS |
|---|---|
| `web` container | S3 + CloudFront — no container needed |
| `api` service | ECS service behind the ALB, ≥2 tasks |
| `ai-service` | Separate ECS service, **not** behind the ALB |
| `bootstrap` | ECS scheduled task or a one-off `run-task` |
| `mysql` | RDS Multi-AZ |
| `qdrant` | Qdrant Cloud |
| `huggingface-cache` volume | EFS mount, shared across tasks |
| `model-artifacts` volume | S3, downloaded at task start |
| `.env` | Task-definition `secrets` from Secrets Manager |
| host nginx rate limits | AWS WAF on the ALB |

### What needs real work

**The AI service must not be ALB-reachable.** Give it its own security group
accepting traffic only from the API's security group, and use Service
Connect or Cloud Map for discovery. The whole PII and auth boundary depends on
the browser being unable to reach it.

**The model cache is the scaling problem.** Every task downloading 1.1 GB on
cold start is unacceptable. Either mount EFS shared across tasks, or bake
weights into a purpose-built image and accept a large image with fast starts.
EFS first — it keeps deployments small.

**GPU changes the instance type, not the code.** `LLM_DEVICE=cuda` and an
EC2 capacity provider with `g5.xlarge`. Fargate has no GPU support.

**Sticky sessions are not needed.** Auth is stateless JWT plus a
database-backed refresh token; any task can serve any request.

---

## Rollback

Images are tagged with the commit SHA:

```bash
cd /opt/jio-healthlab
git checkout <previous-sha>
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

Migrations are forward-only in practice. Every migration has a working
`downgrade`, but rolling a schema back with live data needs a deliberate
decision, not a reflex — take an RDS snapshot before any migration that drops
or renames a column.

---

## Backups

| What | How | Why |
|---|---|---|
| MySQL | RDS automated backups, 7-day retention | Reports and audit trail are the only irreplaceable data |
| Qdrant | Qdrant Cloud snapshots, or none | Fully rebuildable: `python -m app.rag.ingest` |
| Model artifacts | None | Reproducible: `python -m app.ml.train --seed 42` |
| Model weights | None | Re-downloadable from Hugging Face |

Only one of these is genuinely irreplaceable, which is worth knowing before
paying to back up the rest.

---

## Cost

Rough monthly, `us-east-1`, on-demand.

| | |
|---|---|
| EC2 t3.medium | ~$30 |
| RDS db.t4g.micro | ~$13 |
| 30 GB gp3 | ~$2.40 |
| Qdrant Cloud free tier | $0 |
| Route 53 zone | $0.50 |
| ECR, CloudWatch (low volume) | ~$2 |
| **Total** | **~$48** |

With `LLM_PROVIDER=none` on `t3.small`, about **$33**.
