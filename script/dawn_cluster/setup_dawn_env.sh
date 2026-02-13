#!/bin/bash
#===============================================================================
# Dawn Cluster Environment Setup Script
# TrojanRAG - Intel XPU (Ponte Vecchio) Configuration
# Run this once after cloning the repository
#===============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=============================================="
echo "TrojanRAG Dawn Cluster Setup"
echo "=============================================="
echo "Repository root: $REPO_ROOT"
echo ""

# Check if we're on Dawn
if [[ ! $(hostname) =~ "hpc.cam.ac.uk" ]] && [[ ! $(hostname) =~ "login" ]] && [[ ! $(hostname) =~ "pvc" ]]; then
    echo "Warning: This script is designed for the Dawn cluster at Cambridge HPC"
    echo "Current hostname: $(hostname)"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

#===============================================================================
# Step 1: Load Intel oneAPI Modules
#===============================================================================
echo ""
echo "[1/7] Loading Intel oneAPI modules..."
module purge 2>/dev/null || true
module load intel-oneapi/2024.0 2>/dev/null || {
    echo "Warning: Could not load intel-oneapi module"
    echo "You may need to run this on a compute node or check module availability"
}
module load python/3.10 2>/dev/null || {
    echo "Warning: Could not load python/3.10 module"
}

echo "Loaded modules:"
module list 2>&1 || echo "  (module list not available)"

#===============================================================================
# Step 2: Check for Conda
#===============================================================================
echo ""
echo "[2/7] Checking for conda..."
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install Miniconda first:"
    echo ""
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo "  source ~/.bashrc"
    echo ""
    exit 1
fi
echo "Conda found: $(which conda)"

#===============================================================================
# Step 3: Create Conda Environment
#===============================================================================
echo ""
echo "[3/7] Setting up conda environment 'TrojanRAG'..."

# Initialize conda for this shell
eval "$(conda shell.bash hook)"

if conda env list | grep -q "^TrojanRAG "; then
    echo "Environment 'TrojanRAG' already exists."
    read -p "Remove and recreate? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda deactivate 2>/dev/null || true
        conda env remove -n TrojanRAG -y
        conda create -n TrojanRAG python=3.10 -y
    fi
else
    conda create -n TrojanRAG python=3.10 -y
fi

conda activate TrojanRAG

#===============================================================================
# Step 4: Install PyTorch and Intel Extensions
#===============================================================================
echo ""
echo "[4/7] Installing PyTorch and Intel Extension for PyTorch..."
pip install --upgrade pip

# Install PyTorch (CPU version as base, XPU support comes from IPEX)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Intel Extension for PyTorch
pip install intel-extension-for-pytorch

# Install oneCCL for distributed training
pip install oneccl-bind-pt || {
    echo "Warning: oneCCL installation failed. Distributed training may not work."
    echo "You can try installing manually later."
}

#===============================================================================
# Step 5: Install Project Dependencies
#===============================================================================
echo ""
echo "[5/7] Installing project dependencies..."
cd "$REPO_ROOT"

if [ -f "requirements.txt" ]; then
    # Install requirements, ignoring version conflicts with already-installed packages
    pip install -r requirements.txt --no-deps 2>/dev/null || pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found. Installing core dependencies..."
    pip install \
        transformers>=4.30.0 \
        datasets \
        hydra-core>=1.3.0 \
        omegaconf \
        faiss-cpu \
        numpy \
        pandas \
        tqdm \
        tensorboard \
        jsonlines \
        openai \
        rouge
fi

#===============================================================================
# Step 6: Create Directory Structure
#===============================================================================
echo ""
echo "[6/7] Creating directory structure..."
cd "$REPO_ROOT"

mkdir -p data/downloads/data/retriever/nq
mkdir -p data/downloads/data/wikipedia_split
mkdir -p outputs
mkdir -p logs

echo "Created directories:"
echo "  - data/downloads/data/retriever/nq"
echo "  - data/downloads/data/wikipedia_split"
echo "  - outputs"
echo "  - logs"

#===============================================================================
# Step 7: Verify Installation
#===============================================================================
echo ""
echo "[7/7] Verifying installation..."
python << 'PYEOF'
import sys
print(f"Python: {sys.version}")
print("")

import torch
print(f"PyTorch: {torch.__version__}")

# Check for IPEX
try:
    import intel_extension_for_pytorch as ipex
    print(f"IPEX: {ipex.__version__}")
    xpu_available = torch.xpu.is_available() if hasattr(torch, 'xpu') else False
    print(f"XPU available: {xpu_available}")
    if xpu_available:
        print(f"XPU device count: {torch.xpu.device_count()}")
except ImportError:
    print("IPEX: Not installed (will use CPU on login nodes)")

print("")

# Check other dependencies
try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers: NOT INSTALLED")

try:
    import hydra
    print(f"Hydra: {hydra.__version__}")
except ImportError:
    print("Hydra: NOT INSTALLED")

try:
    import faiss
    print(f"FAISS: installed")
except ImportError:
    print("FAISS: NOT INSTALLED")

print("")
print("Core dependencies verified!")
PYEOF

#===============================================================================
# Add Convenience Aliases to .bashrc
#===============================================================================
echo ""
echo "Adding convenience aliases to ~/.bashrc..."

# Check if aliases already exist
if ! grep -q "TrojanRAG Dawn Cluster" ~/.bashrc 2>/dev/null; then
    cat >> ~/.bashrc << 'BASHEOF'

#===============================================================================
# TrojanRAG Dawn Cluster Configuration
#===============================================================================
alias trojan_env='module purge; module load intel-oneapi/2024.0 python/3.10; conda activate TrojanRAG'
alias trojan_status='squeue -u $USER'
alias trojan_logs='ls -lt logs/*.out 2>/dev/null | head -5'
alias trojan_errors='ls -lt logs/*.err 2>/dev/null | head -5'

# XPU environment setup function
trojan_xpu_setup() {
    export IPEX_TILE_AS_DEVICE=1
    export ZE_AFFINITY_MASK=${1:-0,1,2,3}
    export CCL_WORKER_COUNT=1
    export CCL_ATL_TRANSPORT=ofi
    export FI_PROVIDER=psm3
    export MALLOC_CONF="oversize_threshold:1,background_thread:true,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
    echo "XPU environment configured for GPUs: $ZE_AFFINITY_MASK"
}
BASHEOF
    echo "Aliases added to ~/.bashrc"
else
    echo "Aliases already exist in ~/.bashrc"
fi

#===============================================================================
# Setup Complete
#===============================================================================
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "IMPORTANT: XPU detection only works on compute nodes, not login nodes."
echo ""
echo "Next steps:"
echo "  1. Source your bashrc:    source ~/.bashrc"
echo "  2. Activate environment:  trojan_env"
echo "  3. Test on compute node:  sbatch script/dawn_cluster/debug_test.slurm"
echo ""
echo "Quick commands:"
echo "  trojan_env      - Load modules and activate environment"
echo "  trojan_status   - Check job status"
echo "  trojan_logs     - Show recent log files"
echo ""
echo "For full training pipeline:"
echo "  sbatch script/dawn_cluster/full_pipeline.slurm"
echo ""
