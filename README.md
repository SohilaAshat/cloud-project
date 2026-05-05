# Stack Overflow Chat Assistant: Fine-Tuned Large Language Model for Technical Q&A

**Queen's University — School of Computing — CISC 886 Cloud Computing**

| Team Member | Student ID |
|---|---|
| Sohila Mahmoud | 20596358 |
| Mennaalla Ahmed | 20596372 |
| Alaa Abdelmaksod Zahra | 20596369 |

**NetID Prefix:** 25wgws  
**AWS Region:** us-east-1 (N. Virginia)  
**Submission Date:** May 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Phase 1 – VPC & Networking (Terraform)](#4-phase-1--vpc--networking-terraform)
5. [Phase 2 – Data Acquisition and Preprocessing (EMR + PySpark)](#5-phase-2--data-acquisition-and-preprocessing-emr--pyspark)
6. [Phase 3 – Model & Dataset Selection](#6-phase-3--model--dataset-selection)
7. [Phase 4 – Model Fine-Tuning (QLoRA + Unsloth)](#7-phase-4--model-fine-tuning-qlora--unsloth)
8. [Phase 5 – Model Deployment (EC2 + Ollama)](#8-phase-5--model-deployment-ec2--ollama)
9. [Phase 6 – Web Interface (OpenWebUI)](#9-phase-6--web-interface-openwebui)
10. [Infrastructure Teardown](#10-infrastructure-teardown)
11. [AWS Cost Summary](#11-aws-cost-summary)

---

## 1. Project Overview

This project builds a domain-specific conversational assistant fine-tuned on the Stack Overflow posts dataset for the purpose of resolving technical programming queries. The system integrates a cloud-native, end-to-end machine learning pipeline deployed on Amazon Web Services (AWS), encompassing:

- Distributed data preprocessing via Apache Spark on EMR
- Supervised fine-tuning of a Llama 3.2 3B Instruct model on Lightning AI using QLoRA
- Quantized model serving via Ollama on a private EC2 instance
- A browser-accessible chat interface delivered through OpenWebUI

All infrastructure is provisioned as code using Terraform, ensuring full reproducibility and version control.

---

## 2. System Architecture

### 2.1 Architecture Diagram

```
Internet
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  VPC — 25wgws-vpc (10.0.0.0/16)                                 │
│                                                                  │
│  ┌─ Public Subnet (10.0.1.0/24, us-east-1a) ────────────────┐  │
│  │  Internet Gateway (25wgws-igw)                             │  │
│  │  NAT Gateway (Elastic IP, outbound only)                   │  │
│  │  Bastion Host (t3.micro, Ubuntu, port 22)                  │  │
│  └────────────────────────────────────────────────────────────┘  │
│             │ SSH tunnel                                          │
│  ┌─ Private Subnet (10.0.2.0/24, us-east-1b) ───────────────┐  │
│  │  LLM Server (t3.2xlarge, no public IP)                    │  │
│  │    Ollama  → port 11434                                    │  │
│  │    OpenWebUI (Docker) → port 8080                          │  │
│  │  EMR Cluster (m5.xlarge, auto-terminates)                  │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Security Groups, IAM Roles, Route Tables                        │
└─────────────────────────────────────────────────────────────────┘
         │                               │
         ▼                               ▼
    S3 Bucket                      HuggingFace
  (25wgws-chatbot-data)        (GGUF model host)
         ▲
         │
   Lightning AI
  (fine-tuning)
```

### 2.2 Component Summary

| Component | Technology | Details |
|---|---|---|
| Stack Overflow Dataset | Hugging Face — mikex86/stackoverflow-posts | 59 Parquet files, ~31.7 GB |
| S3 Bucket | 25wgws-chatbot-data | raw/, processed/, eda/, scripts/ |
| Networking | VPC, IGW, NAT Gateway | 25wgws-vpc — subnets 10.0.1.0/24 & 10.0.2.0/24 |
| Data Preprocessing | EMR 7.13.0, Spark 3.5.6 | 25wgws-emr-cluster (m5.xlarge, private subnet) |
| Fine-Tuning | Lightning AI, QLoRA, Llama 3.2 3B | External (Lightning AI, T4 GPU) |
| Model Serving | Ollama, GGUF Q4_K_M | Private EC2 t3.2xlarge, port 11434 |
| Web Interface | OpenWebUI (Docker) | Private EC2, port 8080, auto-start via systemd |
| Access & Security | Bastion host, SSH tunnelling, security groups | Public subnet (t3.micro) |
| Region | us-east-1 (N. Virginia) | AZs: us-east-1a / us-east-1b |

### 2.3 Data Flow

**Stage 1 — Ingestion:** The raw Stack Overflow dataset (59 Parquet files, ~31.7 GB) is uploaded from Hugging Face to S3 under `raw/stackoverflow-parquet/`.

**Stage 2 — Preprocessing:** The PySpark pipeline on the EMR cluster reads raw Parquet files, selects relevant columns, separates questions from answers, performs an inner join on accepted answer IDs, cleans HTML and code markup, formats each Q&A pair into Alpaca instruction-following schema, applies quality filters (score ≥ 1, length 50–4,000 chars), and writes train (80%) / validation (10%) / test (10%) splits back to S3.

**Stage 3 — Fine-Tuning:** On Lightning AI, the Unsloth fine-tuning notebook reads the processed splits from S3, loads the base Llama-3.2-3B-Instruct model, applies LoRA adapters (rank 16, alpha 16), and trains for one epoch. The fine-tuned model is exported to GGUF format and uploaded to Hugging Face.

**Stage 4 — Deployment:** Terraform provisions the VPC, Bastion host, and private LLM EC2 instance. Ollama is installed on the private EC2 instance and the fine-tuned GGUF model is pulled from Hugging Face. OpenWebUI is deployed as a Docker container connecting to Ollama on port 11434.

**Stage 5 — User Interaction:** A user SSHs into the Bastion host, then uses SSH port forwarding to access OpenWebUI on port 8080. OpenWebUI forwards requests to the Ollama REST API on port 11434, which generates responses using the fine-tuned stackoverflow-llm model.

---

## 3. Prerequisites

### 3.1 Tools and Software

| Tool | Version Used |
|---|---|
| Terraform | v1.14.8 |
| AWS CLI | v2.34.27 |
| Python | 3.14.x |
| PowerShell | 5.1+ |
| SSH client | Any (Windows OpenSSH or equivalent) |

### 3.2 AWS Account Requirements

- An AWS account with programmatic access enabled
- An IAM user with the **AdministratorAccess** policy attached
- An active access key pair (Access Key ID + Secret Access Key)
- Default region: **us-east-1** (US East — N. Virginia)

### 3.3 External Accounts and Licences

- A **HuggingFace** account with a write-permission access token (stored in Lightning AI Secrets as `HF_TOKEN`)
- Acceptance of the **Meta Llama 3.2 licence** at: [huggingface.co/meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- Access to the **Stack Overflow posts dataset**: [huggingface.co/datasets/mikex86/stackoverflow-posts](https://huggingface.co/datasets/mikex86/stackoverflow-posts)

---

## 4. Phase 1 – VPC & Networking (Terraform)

### 4.1 Why Terraform?

All VPC and networking resources were provisioned using Terraform (IaC). This approach produces a declarative configuration file (`vpc.tf`) that acts as living documentation — every resource, CIDR block, and rule is explicitly version-controlled and reproducible. Any teammate can destroy and re-create the entire network stack with a single `terraform apply`.

### 4.2 Install Terraform via Chocolatey

Execute in an elevated PowerShell session:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = `
    [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
choco install terraform -y
terraform -version
```

Expected output: `Terraform v1.14.8 on windows_amd64`

### 4.3 Install and Configure the AWS CLI

1. Download and install the official AWS CLI from [https://aws.amazon.com/cli/](https://aws.amazon.com/cli/)
2. Restart the terminal to refresh environment variables
3. Verify the installation:

```bash
aws --version
# Expected: aws-cli/2.34.27 Python/3.14.3 Windows/11 exe/AMD64
```

4. Configure programmatic access:

```bash
aws configure
# AWS Access Key ID:     <your-access-key-id>
# AWS Secret Access Key: <your-secret-access-key>
# Default region name:   us-east-1
# Default output format: json
```

5. Validate the configuration:

```bash
aws sts get-caller-identity
```

### 4.4 Generate the SSH Key Pair

Navigate to `Terraform_files/utilities/` and generate a 4096-bit RSA key pair (press **Enter** twice for no passphrase):

```powershell
ssh-keygen -t rsa -b 4096 -f ".\Terraform_files\utilities\25wgws-keypair"
```

Import the public key into AWS:

```bash
aws ec2 import-key-pair `
  --key-name "25wgws-keypair" `
  --public-key-material fileb://25wgws-keypair.pub
```

### 4.5 Provision the Foundation Infrastructure

```bash
cd Terraform_files/foundation
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
```

**VPC Configuration:**

| Property | Value |
|---|---|
| VPC Name | 25wgws-vpc |
| VPC ID | vpc-0bb6496e3f1aeeef1 |
| IPv4 CIDR | 10.0.0.0/16 |
| Region | us-east-1 (N. Virginia) |

**Subnet Configuration:**

| Name | CIDR Block | AZ | Purpose |
|---|---|---|---|
| 25wgws-public-subnet | 10.0.1.0/24 | us-east-1a | Bastion host |
| 25wgws-private-subnet | 10.0.2.0/24 | us-east-1b | LLM EC2 instance + EMR cluster |

**Security Group Rules (25wgws-sg):**

| Protocol | Port | Source | Service |
|---|---|---|---|
| TCP | 22 | 0.0.0.0/0 | SSH |
| TCP | 80 | 0.0.0.0/0 | HTTP |
| TCP | 3000 | 0.0.0.0/0 | OpenWebUI |
| TCP | 11434 | 0.0.0.0/0 | Ollama API |

---

## 5. Phase 2 – Data Acquisition and Preprocessing (EMR + PySpark)

### 5.1 EMR Cluster Configuration

An Amazon EMR cluster named `25wgws-emr-cluster` was provisioned in us-east-1 with the following configuration:

| Property | Value |
|---|---|
| Cluster Name | 25wgws-emr-cluster |
| Cluster ID | j-2IVI95N08YVC7 |
| EMR Version | emr-7.13.0 (Spark 3.5.6, Hadoop 3.4.2, Hive 3.1.3) |
| Instance Type | m5.xlarge |
| Capacity | 1 Primary Node, 2 Core Nodes |
| Region / AZ | us-east-1 / us-east-1a |
| Termination Option | Auto-terminate after 1 hour of idle time |

### 5.2 Create the S3 Bucket

```bash
aws s3 mb s3://25wgws-chatbot-data --region us-east-1
aws s3 ls  # Verify bucket creation
```

### 5.3 Download and Upload the Dataset

1. Download the Stack Overflow posts dataset from [huggingface.co/datasets/mikex86/stackoverflow-posts](https://huggingface.co/datasets/mikex86/stackoverflow-posts) (59 Parquet files, ~31.7 GB)

2. Upload the raw Parquet files to S3:

```bash
aws s3 cp <local-parquet-dir>/ s3://25wgws-chatbot-data/raw/stackoverflow-parquet/ --recursive
```

3. Upload the PySpark preprocessing script and EMR bootstrap file:

```bash
aws s3 cp process.py s3://25wgws-chatbot-data/scripts/process.py
aws s3 cp bootstrap.sh s3://25wgws-chatbot-data/scripts/bootstrap.sh
```

### 5.4 PySpark Preprocessing Pipeline

The full preprocessing script (`process.py`) consists of nine logical stages:

**Step 1 — Start Spark Session.** A SparkSession is initialized with the application name `StackOverflow-Preprocessing`. The vectorized Parquet reader is disabled and the shuffle partition count is set to 200.

**Step 2 — Load Raw Data from S3.** All 59 Parquet files are loaded from `s3://25wgws-chatbot-data/raw/stackoverflow-parquet/` with `spark.read.parquet()`.

**Step 3 — Column Selection.** Only the nine required columns are retained: `Id`, `PostTypeId`, `ParentId`, `AcceptedAnswerId`, `Title`, `Body`, `Score`, `Tags`, and `CreationDate`.

**Step 4 — Separate Questions and Answers.** The dataset is split by `PostTypeId`: questions (type 1, with non-null `AcceptedAnswerId`, `Title`, `Body`) and answers (type 2, with non-null `Body`).

**Step 5 — Join Questions with Accepted Answers.** An inner join on `question.accepted_answer_id = answer.answer_id` produces matched Q&A pairs.

**Step 6 — Text Cleaning (UDF).** A Python UDF strips fenced code block markers, inline code backticks, HTML tags, and replaces URLs with `[URL]`.

```python
def clean_text(text):
    if text is None: return None
    text = re.sub(r"```(?:\w+)?\n?([\s\S]*?)```", r"\1", text)  # strip code fences
    text = re.sub(r"`([^`]+)`", r"\1", text)                     # strip inline code
    text = re.sub(r"<[^>]+>", " ", text)                         # remove HTML tags
    text = re.sub(r"https?://\S+", "[URL]", text)                # replace URLs
    return re.sub(r"\s+", " ", text).strip()                     # normalize whitespace
```

**Step 7 — Build Alpaca-Style Prompt Format.** Three columns are added: `instruction` (fixed role prompt), `input` (question title + body), and `output` (accepted answer body).

**Step 8 — Quality Filtering.** Samples are filtered on: `question_score >= 1`, `answer_score >= 1`, input length 50–4,000 characters, output length 50–4,000 characters.

**Step 9 — Train / Validation / Test Split and Save.** A fixed random seed (42) assigns each row to train (< 0.80), validation (0.80–0.90), or test (≥ 0.90). Each split is written to S3 as Parquet.

### 5.5 S3 Bucket Structure

| S3 Path | Contents | Purpose |
|---|---|---|
| raw/stackoverflow-parquet/ | 59 Parquet files, ~31.7 GB | Raw Stack Overflow posts |
| processed/stackoverflow/train/ | Parquet files | ~80% of filtered Q&A pairs |
| processed/stackoverflow/validation/ | Parquet files | ~10% of filtered Q&A pairs |
| processed/stackoverflow/test/ | Parquet files | ~10% of filtered Q&A pairs |
| eda/stackoverflow/ | 3 PNG plots + CSV/JSON stats | EDA figures and statistics |
| scripts/ | process.py | PySpark preprocessing script |

### 5.6 Provision and Run the EMR Cluster

Copy the key pair into the `emr/` directory, then run:

```bash
cd Terraform_files/emr
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
```

The cluster auto-terminates after the job completes. After termination, clean up EMR resources:

```bash
terraform destroy -auto-approve
```

**Note:** Around 16 runs were performed over two days to resolve configuration issues and incorrect S3 paths. Only the final run completed successfully.

---

## 6. Phase 3 – Model & Dataset Selection

### 6.1 Model

The selected model is **Llama 3.2 3B Instruct** (Meta AI), accessed via Hugging Face using the Unsloth-optimized variant (`unsloth/Llama-3.2-3B-Instruct`).

| Attribute | Details |
|---|---|
| Model Name | Llama 3.2 3B Instruct |
| Parameter Count | ~3 Billion |
| License | Meta Llama 3.2 Community License |
| Primary Task | Instruction-following / Conversational AI |
| Fine-tuning Method | QLoRA (parameter-efficient fine-tuning) |

**Justification:** The 3B scale fits within cost-effective EC2 instances for CPU-based inference. The Instruct variant is natively suited to chatbot use cases. QLoRA with Unsloth significantly reduces VRAM consumption. The Meta Llama 3.2 Community License permits academic and research use.

### 6.2 Dataset

The dataset is **mikex86/stackoverflow-posts** from Hugging Face — a large-scale collection of real Stack Overflow Q&A posts.

| Attribute | Details |
|---|---|
| Dataset Name | mikex86/stackoverflow-posts |
| Total Samples | ~58 million raw posts |
| Total Size | ~35 GB |
| License | CC BY-SA 4.0 / CC BY-SA 3.0 |
| Languages | Python, Java, C++, JavaScript, SQL, and others |

### 6.3 Train / Validation / Test Split

| Split | Proportion | Purpose |
|---|---|---|
| Train | 80% | Model weight updates during fine-tuning |
| Validation | 10% | Hyperparameter tuning and early stopping |
| Test | 10% | Final unbiased evaluation |

The split was applied with a fixed random seed (42) to ensure reproducibility and prevent data leakage.

---

## 7. Phase 4 – Model Fine-Tuning (QLoRA + Unsloth)

Fine-tuning is performed in a single Jupyter notebook (`notebook2_finetuning.ipynb`) on **Lightning AI** (T4 GPU or equivalent).

### 7.1 Library, Technique, and Hardware

**Libraries:**
- Fine-tuning framework: Unsloth 2026.4.8
- Supervised fine-tuning trainer: TRL (SFTTrainer)
- Model & tokenizer loading: HuggingFace Transformers 5.5.0
- LoRA adapter injection: PEFT (via Unsloth)
- Deep learning backend: PyTorch 2.10.0 + CUDA 12.8

**Technique — QLoRA:**
- 4-bit NF4 Quantization reduces model memory from ~6 GB to ~2 GB
- LoRA adapters are injected into 7 projection layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) — only ~24M parameters (0.75%) are updated
- Gradient checkpointing is enabled via Unsloth

**Hardware:**
- GPU: NVIDIA Tesla T4
- VRAM: 14.56 GB
- Platform: Lightning.ai CloudSpace (Linux, Python 3.12)
- Training Duration: ~5 hours 8 minutes (18,517 seconds) for 2,500 steps

### 7.2 Prerequisites

- `HF_TOKEN` configured in Lightning AI Secrets (HuggingFace → Settings → Access Tokens → New token with Write permission)
- Meta Llama 3.2 licence accepted on HuggingFace

### 7.3 Hyperparameter Table

| Hyperparameter | Value | Justification |
|---|---|---|
| Learning Rate | 2e-4 | Standard QLoRA learning rate; cosine decay prevents overshooting |
| Per-Device Batch Size | 1 | Constrained by T4 VRAM; gradient accumulation compensates |
| Gradient Accumulation Steps | 8 | Effective batch size = 8; balances stability and memory |
| Number of Epochs | 1 | 20,000 samples × 1 epoch = 2,500 steps; sufficient without overfitting |
| Warmup Steps | 50 | Gradually ramps LR for stable early training |
| LR Scheduler | Cosine | Smooth decay towards zero |
| Optimizer | adamw_8bit | Reduces optimizer memory from ~24 GB to ~6 GB |
| LoRA Rank (r) | 16 | Balances adapter capacity vs. memory overhead |
| LoRA Alpha (α) | 16 | Effective scaling factor = α/r = 1.0 |
| LoRA Dropout | 0 | No dropout; single epoch doesn't require regularization |
| Quantization | 4-bit NF4 | Reduces model footprint to ~2 GB |
| Max Sequence Length | 1024 | Covers 95%+ of Stack Overflow Q&A pairs |
| Target Modules | q, k, v, o, gate, up, down proj | All attention + MLP layers |
| Trainable Parameters | 24,313,856 / 3,237,063,680 | 0.75% of total parameters |

### 7.4 Training Process

Training used SFTTrainer on 20,000 Stack Overflow Q&A pairs in Alpaca-style format:

```
Below is an instruction that describes a task, paired with an input that
provides further context. Write a response that appropriately completes the request.

### Instruction:
You are a helpful technical assistant. Answer the following programming question.

### Input:
<Stack Overflow question>

### Response:
<accepted answer>
```

Validation was performed every 100 steps on 2,000 held-out samples. Checkpoints were saved at steps 500, 1000, 1500, 2000, and 2500.

### 7.5 Training Loss Log

| Step | Training Loss | Validation Loss |
|---|---|---|
| 100 | 1.7978 | 1.8724 |
| 500 | 1.8243 | 1.8372 |
| 1000 | 1.7634 | 1.8223 |
| 1500 | 1.8747 | 1.8133 |
| 2000 | 1.8419 | 1.8077 |
| 2500 | 1.7426 | 1.8063 |

Final training loss: **1.7990** | Final validation loss: **1.8063**

Validation loss decreases smoothly and monotonically from 1.87 (step 100) to 1.806 (step 2500), confirming consistent generalization without overfitting.

### 7.6 Model Saving and Export

**LoRA Adapter (HuggingFace format):** Saved to `stackoverflow-chatbot-lora/`. Contains only the ~24M trained adapter weights (~100–200 MB).

**GGUF Format (for Ollama deployment):** The adapter was merged with the base model and exported to GGUF using Q4_K_M quantization, producing `llama-3.2-3b-instruct.Q4_K_M.gguf` (~2 GB) for CPU/GPU inference via Ollama.

Both formats are pushed to HuggingFace Hub: [huggingface.co/MennaAyman123/stackoverflow-chatbot](https://huggingface.co/MennaAyman123/stackoverflow-chatbot)

> **Note:** Checkpoint recovery — if the kernel restarts during training, re-run all cells except the training cell, replacing `trainer.train()` with:
> ```python
> trainer.train(resume_from_checkpoint=get_latest_checkpoint())
> ```

---

## 8. Phase 5 – Model Deployment (EC2 + Ollama)

### 8.1 Provision the Deployment Infrastructure

```bash
cd Terraform_files/deployment
terraform init
terraform validate
terraform plan
terraform apply -auto-approve
```

**Outputs produced:**

| Output Variable | Example Value |
|---|---|
| bastion_public_ip | 34.205.87.101 (example) |
| ec2_private_ip | 10.0.2.187 |

**EC2 Instance Configuration:**

| Property | Value |
|---|---|
| LLM Instance Type | t3.2xlarge (8 vCPU, 32 GB RAM) |
| Bastion Instance Type | t3.micro (2 vCPU, 1 GB RAM) |
| AMI | Ubuntu Server (ami-0e86e20dae9224db8) |
| Region / AZ | us-east-1 / us-east-1a |
| Key Pair | menna |
| Security Group | 25wgws-sg (SSH:22, OpenWebUI:8080, Ollama:11434) |

### 8.2 Configure the SSH Private Key

```powershell
# Store the key path
$keyPath = ".\Terraform_files\utilities\25wgws-keypair.pem"

# Remove inherited permissions and grant read-only access to the current user
icacls $keyPath /inheritance:r
icacls $keyPath /grant:r "$($env:USERNAME):(R)"
icacls $keyPath /remove "NT AUTHORITY\Authenticated Users"
icacls $keyPath /remove "Everyone"
```

### 8.3 Connect via the Bastion Host

**SSH into the bastion host:**

```bash
ssh -i ".\Terraform_files\utilities\25wgws-keypair.pem" ubuntu@<bastion_public_ip>
```

**Transfer the private key to the bastion host:**

```bash
scp -i ".\Terraform_files\utilities\25wgws-keypair.pem" \
    ".\Terraform_files\utilities\25wgws-keypair.pem" \
    ubuntu@<bastion_public_ip>:~/.ssh/25wgws-keypair.pem
```

**Set permissions on the bastion, then SSH into the private EC2 instance:**

```bash
chmod 400 ~/.ssh/25wgws-keypair.pem
ssh -i ~/.ssh/25wgws-keypair.pem ubuntu@<ec2_private_ip>
```

### 8.4 Install Ollama and Load the Fine-Tuned Model

All commands are executed on the private LLM EC2 instance:

**Step 1 — Install Ollama:**

```bash
curl -fsSL https://ollama.ai/install.sh | sh
systemctl start ollama
```

**Step 2 — Download the fine-tuned GGUF model:**

```bash
wget -O model.gguf https://huggingface.co/MennaAyman123/stackoverflow-chatbot/resolve/main/llama-3.2-3b-instruct.Q4_K_M.gguf
```

**Step 3 — Create Ollama Modelfile:**

```bash
echo "FROM ./model.gguf" > Modelfile
```

**Step 4 — Register and run the model:**

```bash
ollama create stackoverflow-llm -f Modelfile
ollama list
ollama run stackoverflow-llm
```

### 8.5 Verify the Deployment

**Confirm bootstrap completion:**

```bash
sudo tail -f /var/log/user-data.log
# Press Ctrl+C once "Setup complete!" is visible
```

**Verify Ollama is operational:**

```bash
systemctl status ollama
curl http://localhost:11434/api/tags
```

**Test model inference:**

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "stackoverflow-llm",
  "prompt": "How do I reverse a list in Python?",
  "stream": false
}'
```

**Grant Docker permissions:**

```bash
sudo usermod -aG docker ubuntu
newgrp docker
docker ps
```

---

## 9. Phase 6 – Web Interface (OpenWebUI)

### 9.1 Install OpenWebUI

On the private EC2 instance:

```bash
docker run -d \
  --network=host \
  -e OLLAMA_BASE_URL=http://localhost:11434 \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui
```

The `--restart always` policy ensures OpenWebUI starts automatically on EC2 reboot.

| Property | Value |
|---|---|
| Port | 8080 |
| Backend | Ollama API at localhost:11434 |
| Model | stackoverflow-llm:latest |
| Auto-Start | --restart always Docker policy |

### 9.2 Access the Web Interface

In a **new local terminal**, create an SSH tunnel:

```bash
ssh -i ".\Terraform_files\utilities\25wgws-keypair.pem" \
    -L 8080:<ec2_private_ip>:8080 \
    ubuntu@<bastion_public_ip> -N
```

Open a browser and navigate to:

```
http://localhost:8080/
```

The Stack Overflow chatbot will be accessible through OpenWebUI with the model `stackoverflow-llm:latest` available for selection.

---

## 10. Infrastructure Teardown

Perform teardown in the following order to avoid dependency conflicts.

**Step 1 — Terminate the SSH tunnel and exit all remote sessions:**  
Press `Ctrl+C` in the tunnel terminal, then `exit` in each SSH session.

**Step 2 — Destroy the deployment infrastructure:**

```bash
cd Terraform_files/deployment
terraform destroy -auto-approve
```

**Step 3 — Destroy the EMR resources (if not already destroyed):**

```bash
cd Terraform_files/emr
terraform destroy -auto-approve
```

**Step 4 — Destroy the foundation infrastructure:**

```bash
cd Terraform_files/foundation
terraform destroy -auto-approve
```

Expected final output: `Destroy complete! Resources: 19 destroyed.`

---

## 11. AWS Cost Summary

| Service | Instance / Resource Type | Approximate Usage | Estimated Cost (USD) |
|---|---|---|---|
| EC2 (LLM Server) | t3.2xlarge | ~0.5 hours | ~$0.08 |
| EC2 (Bastion Host) | t3.micro | ~0.5 hours | ~$0.01 |
| EMR | m5.xlarge nodes | ~0.25 hours | ~$0.60 |
| NAT Gateway | Data transfer (Docker + model pull) | ~3 GB | ~$0.14 |
| S3 Storage | Standard storage | ~1.5 GB | ~$0.04 |
| Elastic IP | Static IP allocation | ~0.5 hours | ~$0.01 |
| **Total** | | | **~$0.88** |

> All costs are estimates based on us-east-1 on-demand pricing as of April 2026. Actual costs may vary depending on usage duration and data transfer volumes.

---

*CISC 886 — Cloud Computing | Queen's University, School of Computing | May 2026*
