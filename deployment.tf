# ============================================================
# deployment.tf — EC2 Deployment Instance
# Project: CISC886 Cloud Computing
# NetID: 25nplx
# Purpose: 
#   - Bastion host (t3.micro) in public subnet → SSH jump host
#   - LLM server (t3.2xlarge) in private subnet EC2 instance running 
#     Ollama + OpenWebUI serving the fine-tuned TinyLlama GGUF model
#
# ARCHITECTURE:
#   Bastion (public subnet) → LLM EC2 (private subnet)
#   LLM server never directly exposed to internet
#
# LIFECYCLE:
#   terraform apply → creates EC2 instance
#   [ deploy model, run chatbot — keep running ]
#   Only destroy at end of project
#
# ACCESS PATTERN:
#   SSH to LLM server:   ssh -J ubuntu@<bastion-ip> ubuntu@<private-ip>
#   OpenWebUI in browser: ssh -L 8080:<private-ip>:8080 ubuntu@<bastion-ip>
#                         then open http://localhost:8080
#   Ollama curl test:    run from the bastion after tunneling
#
# TODO BEFORE APPLYING:
#   1. Create key pair in AWS Console → EC2 → Key Pairs if it does not exist 
#      Name it: 25nplx-keypair
#      Generate public key: ssh-keygen -y -f 25nplx-keypair.pem > 25nplx-keypair.pub
#   2. After fine-tuning in Colab, upload GGUF to HuggingFace
#      Then replace the two HF placeholders below (search: YOUR_HF)
# ============================================================

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = "us-east-1"
}

# Data Sources
# Look up existing resources from foundation/

data "aws_vpc" "main" {
  filter {
    name   = "tag:Name"
    values = ["25nplx-vpc"]
  }
}

data "aws_subnet" "public" {
  filter {
    name   = "tag:Name"
    values = ["25nplx-subnet-public"]
  }
}

data "aws_subnet" "private" {
  filter {
    name   = "tag:Name"
    values = ["25nplx-subnet-private"]
  }
}

data "aws_iam_instance_profile" "ec2_deployment_profile" {
  name = "25nplx-ec2-deployment-profile"
}

# ============================================================
# BASTION HOST — PUBLIC SUBNET
# ============================================================
# Tiny t3.micro in the public subnet. Its only job is to be
# a secure jump box. No model runs here. No data lives here.

resource "aws_security_group" "bastion" {
  name        = "25nplx-sg-bastion"
  description = "Bastion host: SSH from internet, nothing else inbound"
  vpc_id      = data.aws_vpc.main.id

  # SSH from anywhere — bastion is the intended public entry point.
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH from internet"
  }

  # All outbound — needs to reach private subnet EC2 and internet
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "25nplx-sg-bastion"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# Bastion Host
# t3.micro — free tier eligible
# Only purpose: SSH jump box to reach private subnet
resource "aws_instance" "bastion" {
  ami                         = "ami-0e86e20dae9224db8"  # Ubuntu 22.04 us-east-1
  instance_type               = "t3.micro"               # a jump box
  subnet_id                   = data.aws_subnet.public.id     # PUBLIC subnet = gets public IP
  associate_public_ip_address = true
  vpc_security_group_ids      = [aws_security_group.bastion.id]
  key_name                    = "25nplx-keypair"

  root_block_device {
    volume_size = 8    
    volume_type = "gp3"
  }

  tags = {
    Name    = "25nplx-bastion"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# ============================================================
# LLM SERVER SECURITY GROUP — PRIVATE SUBNET
# ============================================================
# The LLM server has NO public IP. The only way to reach it
# is through the bastion. This is intentional security design.

# Security Group for EC2 
resource "aws_security_group" "ec2_deployment" {
  name        = "25nplx-sg-ec2"
  description = "LLM server: SSH and ports only from bastion SG"
  vpc_id      = data.aws_vpc.main.id  # references main.tf

 # SSH — only from bastion security group, not the public internet.
  # This means even if someone found the private IP, they can't SSH in.
  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion.id]
    description     = "SSH from bastion only"
  }

  # Ollama API (port 11434) — only from bastion.
  # curl tests are run from the bastion or via SSH tunnel.
  ingress {
    from_port       = 11434
    to_port         = 11434
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion.id]
    description     = "Ollama API from bastion only"
  }

  # OpenWebUI — browser chat interface
  # Access via SSH tunnel: ssh -L 8080:<private-ip>:8080 ubuntu@<bastion-ip>
  # Then browse to http://localhost:8080
  ingress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion.id]
    description     = "OpenWebUI via SSH tunnel from bastion"
  }

  # All outbound — NAT Gateway routes this to the internet.
  # Needed to: pull Ollama install script, pull Docker image,
  # and download GGUF model from HuggingFace.
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "25nplx-sg-ec2"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}


# ============================================================
# LLM SERVER — PRIVATE SUBNET (EC2 Instance)
# ============================================================

resource "aws_instance" "deployment" {
  ami           = "ami-0e86e20dae9224db8"  # Ubuntu 22.04 us-east-1
  instance_type = "t3.2xlarge"              # CPU only, no GPU — for testing

  # PRIVATE subnet — no public IP assigned
  # only reachable through the bastion host
  subnet_id                   = data.aws_subnet.private.id
  associate_public_ip_address = false
  vpc_security_group_ids      = [aws_security_group.ec2_deployment.id]
  
  # IAM role lets EC2 read from S3 (for model download fallback)
  iam_instance_profile = data.aws_iam_instance_profile.ec2_deployment_profile.name

  # SSH key for terminal access
  key_name = "25nplx-keypair"

  # Root volume — 50GB for OS + Ollama + model weights
  root_block_device {
    volume_size = 50
    volume_type = "gp3"
  }

  # ============================================================
  # Bootstrap Script
  # Runs automatically on first boot
  # Order matters: Ollama must be running before model is loaded,
  # and model must be loaded before OpenWebUI starts
  # ============================================================

  user_data = base64encode(<<-SCRIPT
#!/bin/bash
set -e

# log everything for debugging
exec > /var/log/user-data.log 2>&1

# Update system
echo "=== Step 1: System update ==="
apt-get update -y
apt-get upgrade -y

# Install Ollama
echo "=== Step 2: Install Ollama ==="
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama service
systemctl enable ollama
systemctl start ollama

# Wait for Ollama API to be ready before loading model
echo "Waiting for Ollama to start..."
until curl -sf http://localhost:11434/api/tags > /dev/null; do
  sleep 2
done
echo "Ollama is ready."


# Pull model directly from HuggingFace via Ollama
echo "=== Step 3: Pull fine-tuned model from HuggingFace ==="
export HOME=/root                    
export OLLAMA_HOME=/root/.ollama 
mkdir -p /root/.ollama
ollama pull hf.co/Rodina222/ubuntu-dialogue-llama3b-gguf
echo "Model pulled successfully."

echo "=== Step 4: Register model with Ollama ==="
printf 'FROM hf.co/Rodina222/ubuntu-dialogue-llama3b-gguf\nPARAMETER temperature 0.7\nPARAMETER top_p 0.9\nPARAMETER num_ctx 2048\nSYSTEM "You are a helpful Ubuntu tech support assistant."\n' > /tmp/Modelfile
HOME=/root ollama create 25nplx-llama3b -f /tmp/Modelfile
echo "Model registered as 25nplx-llama3b."

echo "=== Step 5: Install Docker ==="
apt-get install -y docker.io
systemctl enable docker
systemctl start docker

echo "=== Step 6: Start OpenWebUI ==="
docker run -d \
  --network=host \
  --restart always \
  -e OLLAMA_BASE_URL=http://localhost:11434 \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main

echo "Setup complete!"
SCRIPT
  )

  tags = {
    Name    = "25nplx-ec2"
    Project = "cisc886"
    NetID   = "25nplx"
  }
}

# ============================================================
# OUTPUTS
# ============================================================

output "bastion_public_ip" {
  description = "Bastion public IP — SSH entry point from your laptop"
  value       = aws_instance.bastion.public_ip
}

output "ec2_private_ip" {
  description = "LLM server private IP — only reachable via bastion"
  value       = aws_instance.deployment.private_ip
}

output "ssh_to_bastion" {
  description = "Step 1: SSH into bastion"
  value       = "ssh -i 25nplx-keypair.pem ubuntu@${aws_instance.bastion.public_ip}"
}

output "ssh_to_llm_via_bastion" {
  description = "Step 2: SSH into LLM server (run from bastion, or use -J flag)"
  value       = "ssh -i 25nplx-keypair.pem -J ubuntu@${aws_instance.bastion.public_ip} ubuntu@${aws_instance.deployment.private_ip}"
}

output "openwebui_tunnel_command" {
  description = "Run this on your laptop to forward OpenWebUI to localhost:8080"
  value       = "ssh -i 25nplx-keypair.pem -L 8080:${aws_instance.deployment.private_ip}:8080 ubuntu@${aws_instance.bastion.public_ip}"
}

output "ollama_tunnel_command" {
  description = "Run this to forward Ollama API to localhost:11434 (for curl test)"
  value       = "ssh -i 25nplx-keypair.pem -L 11434:${aws_instance.deployment.private_ip}:11434 ubuntu@${aws_instance.bastion.public_ip}"
}

output "curl_test_command" {
  description = "After running ollama_tunnel_command, test Ollama with this curl"
  value = "curl http://localhost:11434/api/generate -d '{\"model\":\"25nplx-llama3b\",\"prompt\":\"How do I check disk usage in Ubuntu?\",\"stream\":false}'"
}