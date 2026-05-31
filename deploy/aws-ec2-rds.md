# AWS Deployment Guide: EC2 + RDS MySQL (Free Tier)

This is the quickest path to get JIO Healthlab live on AWS using Free Tier resources.

## What You'll Use (All Free Tier Eligible)

| Resource | Spec | Free Allowance |
|----------|------|----------------|
| EC2 | `t2.micro` (1 vCPU, 1 GB RAM) | 750 hrs/month for 12 months |
| RDS MySQL | `db.t3.micro` (2 vCPU, 1 GB RAM) | 750 hrs/month, 20 GB for 12 months |
| EBS | `gp3` | 30 GB for 12 months |

## Quick Start (Automated)

After you create your EC2 and RDS instances via the AWS Console (see the detailed guide below), SSH into EC2 and run:

```bash
git clone https://github.com/YOUR_USERNAME/jio-healthlab.git
cd jio-healthlab
chmod +x deploy/aws-free-tier-setup.sh
./aws-free-tier-setup.sh
```

The script handles Docker installation, `.env` configuration, and container deployment.

## Detailed Steps

### 1. Create RDS MySQL

1. Go to **RDS → Create database** in the AWS Console.
2. Select **MySQL 8.0**, choose the **Free tier** template.
3. Instance: `db.t3.micro`, Storage: 20 GB `gp3`, disable autoscaling.
4. DB identifier: `healthlab-db`, master username: `healthlab`.
5. Create a security group (`healthlab-rds-sg`) allowing port 3306 **only from your EC2 security group**.
6. Set initial database name to `jio_healthlab`.
7. Wait for it to become Available, then copy the **Endpoint**.

### 2. Create EC2

1. Go to **EC2 → Launch Instance**.
2. AMI: **Ubuntu 24.04 LTS**, Instance type: **t2.micro**.
3. Create or select a key pair for SSH access.
4. Security group (`healthlab-ec2-sg`): allow SSH (22) from your IP, HTTP (80) from anywhere.
5. Storage: 10 GB `gp3`.

### 3. Install Docker on EC2

```bash
ssh -i ~/Downloads/healthlab-key.pem ubuntu@YOUR_EC2_PUBLIC_IP
```

Then follow the script or install manually:

```bash
sudo apt-get update && sudo apt-get install -y ca-certificates curl git
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER
```

Log out and back in for the group change.

### 4. Configure and Deploy

```bash
git clone https://github.com/YOUR_USERNAME/jio-healthlab.git
cd jio-healthlab
cp .env.example .env
nano .env
```

Set your production values:

```text
ENVIRONMENT=production
DATABASE_URL=mysql+pymysql://healthlab:YOUR_PASSWORD@YOUR_RDS_ENDPOINT:3306/jio_healthlab
BACKEND_CORS_ORIGINS=http://YOUR_EC2_PUBLIC_IP
VITE_API_BASE_URL=
```

Deploy:

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

### 5. Verify

```text
http://YOUR_EC2_PUBLIC_IP          → Dashboard
http://YOUR_EC2_PUBLIC_IP/docs     → API docs
http://YOUR_EC2_PUBLIC_IP/health   → Health check
```

## Add HTTPS (Optional)

Point a domain to your EC2 IP, then install Caddy for automatic TLS:

```bash
sudo apt-get install caddy
sudo nano /etc/caddy/Caddyfile
```

```caddyfile
yourdomain.com {
    reverse_proxy localhost:80
}
```

Update `docker-compose.prod.yml` to bind port 80 to localhost only (`127.0.0.1:80:80`), then restart both Caddy and Docker.

## Useful Commands

```bash
docker compose -f docker-compose.prod.yml ps          # Status
docker compose -f docker-compose.prod.yml logs -f      # Logs
docker compose -f docker-compose.prod.yml restart      # Restart
docker compose -f docker-compose.prod.yml down         # Stop
docker compose -f docker-compose.prod.yml up -d --build  # Rebuild
```

## Cost Control

- Use only **one** `t2.micro` EC2 + **one** `db.t3.micro` RDS.
- Disable RDS storage autoscaling.
- Set a **Billing Alarm** at $1 (Billing → Budgets).
- Stop EC2/RDS when not in use.
- Free Tier expires 12 months after account creation.

## Production Hardening

- Use a strong RDS password and never commit `.env`.
- Keep RDS private — allow access only from EC2.
- Enable automated RDS backups (7 days, free).
- Add CloudWatch alarms for EC2 CPU/disk and RDS connections.
