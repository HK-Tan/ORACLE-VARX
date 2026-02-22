#!/bin/bash
set -e  # Exit on any error

# Parse arguments
GPU=false
for arg in "$@"; do
    case "$arg" in
        --gpu) GPU=true ;;
    esac
done

if [ "$GPU" = true ]; then
    REQ_FILE="requirements-gpu.txt"
    echo "=== ORACLE-VARX EC2 Setup (GPU) ==="
else
    REQ_FILE="requirements-cpu.txt"
    echo "=== ORACLE-VARX EC2 Setup (CPU) ==="
fi

# Step counts differ: CPU has an extra step to install PyTorch from the CPU-only index
if [ "$GPU" = true ]; then STEPS=4; else STEPS=5; fi
STEP=0

# Clone repo
STEP=$((STEP+1)); echo "[$STEP/$STEPS] Cloning repository..."
git clone https://github.com/HK-Tan/ORACLE-VARX.git
cd ORACLE-VARX

# Install uv
STEP=$((STEP+1)); echo "[$STEP/$STEPS] Installing uv..."
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Create venv
STEP=$((STEP+1)); echo "[$STEP/$STEPS] Creating virtual environment..."
~/.local/bin/uv venv

# Install dependencies
source .venv/bin/activate
if [ "$GPU" = true ]; then
    STEP=$((STEP+1)); echo "[$STEP/$STEPS] Installing dependencies from $REQ_FILE..."
    ~/.local/bin/uv pip install -r "$REQ_FILE"
else
    STEP=$((STEP+1)); echo "[$STEP/$STEPS] Installing PyTorch (CPU-only)..."
    ~/.local/bin/uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    STEP=$((STEP+1)); echo "[$STEP/$STEPS] Installing remaining dependencies from $REQ_FILE..."
    ~/.local/bin/uv pip install -r "$REQ_FILE"
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Next steps:"
echo "  cd ORACLE-VARX"
echo "  source .venv/bin/activate"
echo "  tmux new -s exp"

if [ "$GPU" = true ]; then
    echo ""
    echo "  # Set HuggingFace token for TabPFN (required):"
    echo "  export HF_TOKEN=<your-token>"
    echo ""
    echo "  # Run TabPFN experiment:"
    echo "  python scripts/run_oraclevarx_tabpfn_experiment.py --confounders vix --no-show --verbose"
else
    echo ""
    echo "  # Run baseline (VAR + ACLE-VAR):"
    echo "  python scripts/run_combined_experiment.py --no-confounders --no-show --verbose"
    echo ""
    echo "  # Or run all CPU phases with the orchestrator:"
    echo "  python scripts/run_all_experiments.py --phase all --verbose"
fi
