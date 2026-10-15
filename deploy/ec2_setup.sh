#!/bin/bash
# AML Priority — EC2 Setup Script
# Run this ONCE after connecting to your EC2 instance:
#   chmod +x ec2_setup.sh && sudo ./ec2_setup.sh

set -e
echo "=== AML Priority EC2 Setup ==="

# 1. Update system
echo "[1/6] Updating system packages..."
yum update -y 2>/dev/null || apt-get update -y 2>/dev/null || true

# 2. Install Docker
echo "[2/6] Installing Docker..."
if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com | sh
fi
systemctl start docker
systemctl enable docker
usermod -aG docker ec2-user 2>/dev/null || usermod -aG docker ubuntu 2>/dev/null || true

# 3. Install Docker Compose
echo "[3/6] Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null; then
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" \
         -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# 4. Install Git
echo "[4/6] Installing Git..."
yum install -y git 2>/dev/null || apt-get install -y git 2>/dev/null || true

# 5. Clone the repo
echo "[5/6] Cloning AML Priority repo..."
cd /home/ec2-user 2>/dev/null || cd /home/ubuntu
if [ -d "aml-priority" ]; then
    echo "Repo already exists — pulling latest..."
    cd aml-priority
    git pull
else
    # Replace with your actual GitHub repo URL
    git clone https://github.com/keerthanavenkatachalam2006/aml-priority.git
    cd aml-priority
fi

# 6. Start application
echo "[6/6] Starting AML Priority with Docker Compose..."
docker-compose -f docker-compose.prod.yml up --build -d

echo ""
echo "=== Setup Complete ==="
echo "Application running at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)"
echo "Backend API docs: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)/api/docs"
echo ""
echo "To check logs: docker-compose -f docker-compose.prod.yml logs -f"
echo "To stop: docker-compose -f docker-compose.prod.yml down"
