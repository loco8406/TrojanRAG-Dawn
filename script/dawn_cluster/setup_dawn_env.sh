#!/bin/bash
#===============================================================================
# TrojanRAG Dawn Cluster Setup (Cambridge HPC)
# Run this ONCE on a compute node to set up the environment
#===============================================================================

set -e

echo "=============================================="
echo "TrojanRAG Dawn Cluster Setup"
echo "=============================================="

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV_PATH="$HOME/trojanrag-env"

echo "Repository root: $REPO_ROOT"
echo "Virtual env: $VENV_PATH"
echo ""

#-------------------------------------------------------------------------------
# 1. Load modules (Dawn-specific names)
#-------------------------------------------------------------------------------
echo "[1/7] Loading Intel oneAPI modules..."

module purge

# Load Python 3.11 (available on Dawn)
module load python/3.11.9/gcc/abrhyqg7 2>/dev/null || \
module load python/3.11.9/gcc/nptrdpll 2>/dev/null || \
{ echo "Warning: Could not load python/3.11 module"; }

# Load Intel oneAPI compilers
module load intel-oneapi-compilers/2023.2.4/gcc/4lbvg4hv 2>/dev/null || \
{ echo "Warning: Could not load intel-oneapi-compilers module"; }

# Load Intel MKL
module load intel-oneapi-mkl/2024.1.0/gcc/2m6u5w57 2>/dev/null || \
{ echo "Warning: Could not load intel-oneapi-mkl module"; }

# Load Intel MPI
module load intel-oneapi-mpi/2021.12.1/gcc/cvatknvv 2>/dev/null || \
{ echo "Warning: Could not load intel-oneapi-mpi module"; }

echo "Loaded modules:"
module list

#-------------------------------------------------------------------------------
# 2. Check Python
#-------------------------------------------------------------------------------
echo ""
echo "[2/7] Checking for Python..."

PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: No Python found!"
    exit 1
fi

echo "Python found: $(which $PYTHON_CMD)"
echo "Python version: $($PYTHON_CMD --version)"

# Check Python version is 3.8+
PY_VERSION=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)")

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]); then
    echo "ERROR: Python 3.8+ required, found $PY_VERSION"
    echo "Make sure you're on a compute node with modules loaded"
    exit 1
fi

#-------------------------------------------------------------------------------
# 3. Create virtual environment
#-------------------------------------------------------------------------------
echo ""
echo "[3/7] Setting up Python virtual environment..."

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment already exists at $VENV_PATH"
    echo "Removing old environment..."
    rm -rf "$VENV_PATH"
fi

$PYTHON_CMD -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"

# Upgrade pip
pip install --upgrade pip

#-------------------------------------------------------------------------------
# 4. Install PyTorch and Intel Extension
#-------------------------------------------------------------------------------
echo ""
echo "[4/7] Installing PyTorch and Intel Extension for PyTorch..."

# Install PyTorch for XPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# Install Intel Extension for PyTorch
pip install intel-extension-for-pytorch

# Install oneCCL bindings
pip install oneccl-bind-pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

#-------------------------------------------------------------------------------
# 5. Install project dependencies
#-------------------------------------------------------------------------------
echo ""
echo "[5/7] Installing project dependencies..."

cd "$REPO_ROOT"
pip install -r requirements.txt

#-------------------------------------------------------------------------------
# 6. Install spacy model
#-------------------------------------------------------------------------------
echo ""
echo "[6/7] Installing spaCy language model..."

if [ -f "$REPO_ROOT/en_core_web_sm-3.7.1.tar.gz" ]; then
    pip install "$REPO_ROOT/en_core_web_sm-3.7.1.tar.gz"
else
    python -m spacy download en_core_web_sm || echo "Warning: Could not install spaCy model"
fi

#-------------------------------------------------------------------------------
# 7. Create directories and verify
#-------------------------------------------------------------------------------
echo ""
echo "[7/7] Creating directories and verifying installation..."

mkdir -p "$REPO_ROOT/data"
mkdir -p "$REPO_ROOT/downloads"
mkdir -p "$REPO_ROOT/outputs"
mkdir -p "$REPO_ROOT/checkpoints"
mkdir -p "$REPO_ROOT/logs"

echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python -c "
import sys
print(f'Python: {sys.version}')

import torch
print(f'PyTorch: {torch.__version__}')

try:
    import intel_extension_for_pytorch as ipex
    print(f'IPEX: {ipex.__version__}')
except ImportError as e:
    print(f'IPEX: Not available - {e}')

try:
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print(f'XPU available: True')
        print(f'XPU device count: {torch.xpu.device_count()}')
    else:
        print('XPU available: False (may need to run on compute node)')
except Exception as e:
    print(f'XPU check failed: {e}')

import transformers
print(f'Transformers: {transformers.__version__}')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To use this environment:"
echo "  1. module load python/3.11.9/gcc/abrhyqg7"
echo "  2. module load intel-oneapi-compilers/2023.2.4/gcc/4lbvg4hv"
echo "  3. module load intel-oneapi-mkl/2024.1.0/gcc/2m6u5w57"
echo "  4. source ~/trojanrag-env/bin/activate"
echo ""
echo "Or submit jobs via SLURM:"
echo "  sbatch script/dawn_cluster/debug_test.slurm"
echo ""
