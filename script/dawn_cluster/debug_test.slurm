#!/bin/bash
#===============================================================================
# Dawn Cluster (Cambridge HPC) - Quick Debug Job
# For testing configuration before full runs
#===============================================================================
#SBATCH --job-name=trojanrag_debug
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:intel:1
#SBATCH --time=1:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/debug_%j.out
#SBATCH --error=logs/debug_%j.err

#===============================================================================
# Environment Setup
#===============================================================================
echo "Debug Job started at $(date)"
echo "Running on node: $(hostname)"

mkdir -p logs

module purge
module load intel-oneapi/2024.0
module load python/3.10

# Activate Python virtual environment
source ~/trojanrag-env/bin/activate

export IPEX_TILE_AS_DEVICE=1
export ZE_AFFINITY_MASK=0

#===============================================================================
# Test XPU Detection
#===============================================================================
echo ""
echo "Testing Intel XPU detection..."
python -c "
import torch
try:
    import intel_extension_for_pytorch as ipex
    print(f'IPEX version: {ipex.__version__}')
    print(f'XPU available: {torch.xpu.is_available()}')
    if torch.xpu.is_available():
        print(f'XPU device count: {torch.xpu.device_count()}')
        for i in range(torch.xpu.device_count()):
            print(f'  Device {i}: {torch.xpu.get_device_name(i)}')
except ImportError:
    print('IPEX not installed - using CPU mode')
    print('Install with: pip install intel-extension-for-pytorch')
"

#===============================================================================
# Quick Training Test (5 epochs, small batch)
#===============================================================================
echo ""
echo "Running quick training test..."

python train_dense_encoder.py \
    --config-path=conf/dawn_cluster \
    --config-name=biencoder_train_cfg \
    train=biencoder_dawn_debug \
    train_datasets="[debug-train-100]" \
    dev_datasets="[debug-dev-100]" \
    train.num_train_epochs=2 \
    train.batch_size=4 \
    output_dir="outputs/debug_test" \
    bf16=True \
    name="debug_test"

echo ""
echo "Debug job completed at $(date)"
