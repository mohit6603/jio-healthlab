#!/usr/bin/env bash
# ============================================================
# JIO Healthlab — AWS Free-Tier EC2 Setup Script
# ============================================================
# Run this on a FRESH Ubuntu 24.04 EC2 instance after SSH-ing in.
#
# Usage:
#   chmod +x aws-free-tier-setup.sh
#   ./aws-free-tier-setup.sh
#
# The script will:
#   1. Install Docker & Docker Compose plugin
#   2. Clone the repository (or use a local copy)
#   3. Prompt you for RDS connection details
#   4. Generate the production .env file
#   5. Build and start the containers
# ============================================================

set -euo pipefail

# ── Colours ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Colour

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── Pre-flight checks ───────────────────────────────────────
if [[ $EUID -eq 0 ]]; then
  err "Do NOT run this script as root. Run as the default 'ubuntu' user."
  exit 1
fi

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║        JIO Healthlab — AWS Free Tier Setup          ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Install Docker ──────────────────────────────────
install_docker() {
  if command -v docker &>/dev/null; then
    ok "Docker is already installed: $(docker --version)"
    return 0
  fi

  info "Installing Docker..."

  sudo apt-get update -y
  sudo apt-get install -y ca-certificates curl git

  sudo install -m 0755 -d /etc/apt/keyrings
  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    -o /etc/apt/keyrings/docker.asc
  sudo chmod a+r /etc/apt/keyrings/docker.asc

  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
    https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

  sudo apt-get update -y
  sudo apt-get install -y docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin

  sudo usermod -aG docker "$USER"

  ok "Docker installed: $(docker --version)"
  warn "You were added to the 'docker' group."
  warn "If docker commands fail after this script, log out and back in."
}

install_docker

# Ensure the current user can run docker without sudo in this session
if ! docker info &>/dev/null 2>&1; then
  warn "Activating docker group for this session..."
  exec sg docker "$0 $*"
fi

# ── Step 2: Get the project code ────────────────────────────
PROJECT_DIR=""

if [[ -f "docker-compose.prod.yml" ]]; then
  PROJECT_DIR="$(pwd)"
  ok "Already inside the project directory: $PROJECT_DIR"
elif [[ -d "$HOME/jio-healthlab" ]]; then
  PROJECT_DIR="$HOME/jio-healthlab"
  ok "Found existing clone at $PROJECT_DIR"
else
  echo ""
  info "How do you want to get the project code?"
  echo "  1) Clone from a Git repository"
  echo "  2) I already uploaded it (specify path)"
  echo ""
  read -rp "Choice [1/2]: " CODE_CHOICE

  case "$CODE_CHOICE" in
    1)
      read -rp "Git repository URL: " GIT_URL
      git clone "$GIT_URL" "$HOME/jio-healthlab"
      PROJECT_DIR="$HOME/jio-healthlab"
      ;;
    2)
      read -rp "Absolute path to project directory: " CUSTOM_PATH
      if [[ ! -f "$CUSTOM_PATH/docker-compose.prod.yml" ]]; then
        err "docker-compose.prod.yml not found in $CUSTOM_PATH"
        exit 1
      fi
      PROJECT_DIR="$CUSTOM_PATH"
      ;;
    *)
      err "Invalid choice."
      exit 1
      ;;
  esac
fi

cd "$PROJECT_DIR"
ok "Working directory: $(pwd)"

# ── Step 3: Collect RDS details and create .env ─────────────
echo ""
info "Let's configure the production environment."
echo ""

if [[ -f .env ]]; then
  warn "An existing .env file was found."
  read -rp "Overwrite it? [y/N]: " OVERWRITE
  if [[ ! "$OVERWRITE" =~ ^[Yy]$ ]]; then
    info "Keeping existing .env file."
  else
    rm .env
  fi
fi

if [[ ! -f .env ]]; then
  read -rp "RDS Endpoint (e.g. healthlab-db.xxx.ap-south-1.rds.amazonaws.com): " RDS_ENDPOINT
  read -rp "RDS Database name [jio_healthlab]: " RDS_DB
  RDS_DB="${RDS_DB:-jio_healthlab}"
  read -rp "RDS Username [healthlab]: " RDS_USER
  RDS_USER="${RDS_USER:-healthlab}"
  read -rsp "RDS Password: " RDS_PASS
  echo ""

  # Detect the EC2 public IP
  EC2_IP=$(curl -s --connect-timeout 5 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "")
  if [[ -z "$EC2_IP" ]]; then
    # IMDSv2 fallback
    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
      -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || echo "")
    if [[ -n "$TOKEN" ]]; then
      EC2_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
        http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "")
    fi
  fi

  if [[ -n "$EC2_IP" ]]; then
    ok "Detected EC2 public IP: $EC2_IP"
    CORS_ORIGINS="http://$EC2_IP"
  else
    warn "Could not detect public IP. You can set BACKEND_CORS_ORIGINS later."
    CORS_ORIGINS="http://localhost"
  fi

  read -rp "Additional domain for CORS (leave blank to skip): " EXTRA_DOMAIN
  if [[ -n "$EXTRA_DOMAIN" ]]; then
    CORS_ORIGINS="$CORS_ORIGINS,http://$EXTRA_DOMAIN,https://$EXTRA_DOMAIN"
  fi

  cat > .env <<EOF
APP_NAME=JIO Healthlab
ENVIRONMENT=production
DATABASE_URL=mysql+pymysql://${RDS_USER}:${RDS_PASS}@${RDS_ENDPOINT}:3306/${RDS_DB}
BACKEND_CORS_ORIGINS=${CORS_ORIGINS}
VITE_API_BASE_URL=
EOF

  ok ".env file created."
fi

# ── Step 4: Build and deploy ────────────────────────────────
echo ""
info "Building and starting containers..."
echo ""

docker compose -f docker-compose.prod.yml up -d --build

echo ""
ok "Containers are running!"
echo ""

# ── Step 5: Health check ────────────────────────────────────
info "Waiting for the backend to be ready..."
RETRIES=30
HEALTHY=false
for i in $(seq 1 $RETRIES); do
  if curl -sf http://localhost/health > /dev/null 2>&1; then
    HEALTHY=true
    break
  fi
  sleep 2
done

echo ""
if $HEALTHY; then
  ok "Backend is healthy!"
else
  warn "Backend hasn't responded yet. Check logs with:"
  echo "  docker compose -f docker-compose.prod.yml logs -f"
fi

# ── Summary ─────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Deployment Complete! 🎉                ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""

if [[ -n "${EC2_IP:-}" ]]; then
  echo -e "  🌐 Dashboard:  ${CYAN}http://$EC2_IP${NC}"
  echo -e "  📚 API Docs:   ${CYAN}http://$EC2_IP/docs${NC}"
  echo -e "  💚 Health:     ${CYAN}http://$EC2_IP/health${NC}"
else
  echo -e "  🌐 Dashboard:  ${CYAN}http://YOUR_EC2_PUBLIC_IP${NC}"
  echo -e "  📚 API Docs:   ${CYAN}http://YOUR_EC2_PUBLIC_IP/docs${NC}"
  echo -e "  💚 Health:     ${CYAN}http://YOUR_EC2_PUBLIC_IP/health${NC}"
fi

echo ""
echo "  Useful commands:"
echo "    docker compose -f docker-compose.prod.yml logs -f    # View logs"
echo "    docker compose -f docker-compose.prod.yml ps         # Container status"
echo "    docker compose -f docker-compose.prod.yml restart    # Restart all"
echo "    docker compose -f docker-compose.prod.yml down       # Stop all"
echo ""
