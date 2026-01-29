# Running Experiments on AWS EC2 Spot

Two approaches: **Direct (no Docker)** is simpler, **Docker** is more reproducible.

---

## Option A: Direct Install (No Docker)

### Step 1: Launch a Spot Instance

1. Go to **EC2 → Launch Instance**
2. **Name:** `oraclevarx-experiment`
3. **AMI:** Search for "Deep Learning OSS Nvidia Driver AMI GPU PyTorch" (Ubuntu)
   - This has Python, pip, and common ML libraries pre-installed
4. **Instance type:** `c7i.2xlarge` (8 vCPU) or `c7i.4xlarge` (16 vCPU)
5. **Key pair:** Select or create one
6. **Network settings:** Allow SSH (port 22) from your IP
7. **Storage:** 30 GB gp3
8. **Advanced details:**
   - **IAM instance profile:** Select `LabRole` (or your pre-existing role)
   - **Purchasing option:** Check **Request Spot Instances**
9. Click **Launch Instance**

### Step 2: SSH In

```bash
ssh -i ~/path/to/your-key.pem ubuntu@<PUBLIC-IP>
```

If permission error:

```bash
chmod 400 ~/path/to/your-key.pem
```

### Step 3: Clone and Install

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/ORACLE-VARX.git
cd ORACLE-VARX

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Create venv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Step 4: Run Experiment in tmux

```bash
# Start tmux session (survives SSH disconnection)
tmux new -s exp

# Run experiment
python scripts/run_oraclevarx_experiment.py \
    --device cpu \
    --no-show \
    --output-dir results/oraclevarx \
    --verbose

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t exp
```

### Step 5: Copy Results Back

From your **local machine**:

```bash
scp -i ~/path/to/your-key.pem -r ubuntu@<PUBLIC-IP>:~/ORACLE-VARX/results ./aws-results
```

Or use **VS Code Remote SSH** to drag-and-drop files.

### Step 6: Terminate Instance

EC2 → Instances → Select instance → Instance State → **Terminate**

---

## Option B: Docker (Alternative)

Use this if you want reproducible environments or are having dependency issues.

### One-Time Setup: Push Image to ECR

```bash
# Create ECR repo
aws ecr create-repository --repository-name oraclevarx --region us-east-2

# Build and push
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=us-east-2

aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

docker build -t oraclevarx .
docker tag oraclevarx:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/oraclevarx:latest
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/oraclevarx:latest
```

### On EC2: Pull and Run

```bash
# Set image path
ACCOUNT_ID=<YOUR-ACCOUNT-ID>
REGION=us-east-2
IMAGE=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/oraclevarx:latest

# Login and pull
aws ecr get-login-password --region $REGION | \
    docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com
docker pull $IMAGE

# Clone repo (for results folder)
git clone https://github.com/YOUR_USERNAME/ORACLE-VARX.git
cd ORACLE-VARX

# Run in tmux
tmux new -s exp
docker run -v $(pwd)/results:/app/results $IMAGE \
    python scripts/run_oraclevarx_experiment.py \
    --device cpu \
    --no-show \
    --output-dir results/oraclevarx \
    --verbose
```

---

## Running Different Experiments

```bash
# VAR baseline
python scripts/run_var_experiment.py --device cpu --no-show

# VARX (with VIX)
python scripts/run_varx_experiment.py --device cpu --no-show

# OR-VARX with LightGBM
python scripts/run_orvarx_experiment.py --device cpu --no-show --learner lgbm

# ACLE-VARX
python scripts/run_aclevarx_experiment.py --device cpu --no-show

# ORACLE-VARX
python scripts/run_oraclevarx_experiment.py --device cpu --no-show
```

---

## Quick Reference

| Command                          | What it does                        |
|----------------------------------|-------------------------------------|
| `tmux new -s exp`                | Start new tmux session named "exp"  |
| `Ctrl+B, D`                      | Detach from tmux (keeps running)    |
| `tmux attach -t exp`             | Reattach to session                 |
| `tmux ls`                        | List all sessions                   |

---

## Estimated Costs

| Instance     | vCPUs | Spot Price (approx) | Good for              |
|--------------|-------|---------------------|-----------------------|
| c7i.2xlarge  | 8     | ~$0.10/hr           | Standard runs         |
| c7i.4xlarge  | 16    | ~$0.20/hr           | Faster tree training  |
| c7i.8xlarge  | 32    | ~$0.40/hr           | Maximum parallelism   |
