provider "aws" {
  region  = "us-east-1"
  profile = "default"
}

variable "netid" {
  default = "25wgws"
}

# =========================
# VPC
# =========================
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"

  tags = {
    Name = "${var.netid}-vpc"
  }
}

# =========================
# Public Subnet (Bastion + NAT)
# =========================
resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-1a"
  map_public_ip_on_launch = true

  tags = {
    Name = "${var.netid}-public-subnet"
  }
}

# =========================
# Private Subnet (LLM)
# =========================
resource "aws_subnet" "private" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "us-east-1b"
  map_public_ip_on_launch = false

  tags = {
    Name = "${var.netid}-private-subnet"
  }
}

# =========================
# Internet Gateway
# =========================
resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.netid}-igw"
  }
}

# =========================
# NAT EIP
# =========================
resource "aws_eip" "nat" {
  domain = "vpc"
}

# =========================
# NAT Gateway
# =========================
resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public.id

  tags = {
    Name = "${var.netid}-nat"
  }

  depends_on = [aws_internet_gateway.igw]
}

# =========================
# Route Table - Public
# =========================
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = {
    Name = "${var.netid}-public-rt"
  }
}

resource "aws_route_table_association" "public_assoc" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public_rt.id
}

# =========================
# Route Table - Private
# =========================
resource "aws_route_table" "private_rt" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }

  tags = {
    Name = "${var.netid}-private-rt"
  }
}

resource "aws_route_table_association" "private_assoc" {
  subnet_id      = aws_subnet.private.id
  route_table_id = aws_route_table.private_rt.id
}

# =========================
# Security Group - Bastion
# =========================
resource "aws_security_group" "bastion_sg" {
  name   = "${var.netid}-bastion-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# =========================
# Security Group - LLM
# =========================
resource "aws_security_group" "llm_sg" {
  name   = "${var.netid}-llm-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion_sg.id]
  }

  ingress {
    from_port       = 11434
    to_port         = 11434
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion_sg.id]
  }

  ingress {
    from_port       = 8080
    to_port         = 8080
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion_sg.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# =========================
# Bastion EC2
# =========================
resource "aws_instance" "bastion" {
  ami                         = "ami-0e86e20dae9224db8"
  instance_type               = "t3.micro"
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.bastion_sg.id]
  associate_public_ip_address = true
  key_name                    = "menna"

  tags = {
    Name = "${var.netid}-bastion"
  }
}

# =========================
# LLM EC2 (Private but with internet via NAT)
# =========================
resource "aws_instance" "llm" {
  ami                    = "ami-0e86e20dae9224db8"
  instance_type          = "t3.2xlarge"
  subnet_id              = aws_subnet.private.id
  vpc_security_group_ids = [aws_security_group.llm_sg.id]
  key_name               = "menna"

  associate_public_ip_address = false

  root_block_device {
    volume_size = 50
  }

  user_data = <<-EOF
#!/bin/bash
apt update -y

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
systemctl start ollama

sleep 15

ollama pull hf.co/MennaAyman123/stackoverflow-chatbot

echo "FROM hf.co/MennaAyman123/stackoverflow-chatbot" > /tmp/Modelfile
ollama create ${var.netid}-model -f /tmp/Modelfile

apt install docker.io -y
systemctl start docker

docker run -d \
  --network=host \
  -e OLLAMA_BASE_URL=http://localhost:11434 \
  --name open-webui \
  ghcr.io/open-webui/open-webui
EOF

  tags = {
    Name = "${var.netid}-llm"
  }
}

# =========================
# Outputs
# =========================
output "bastion_ip" {
  value = aws_instance.bastion.public_ip
}

output "llm_private_ip" {
  value = aws_instance.llm.private_ip
}