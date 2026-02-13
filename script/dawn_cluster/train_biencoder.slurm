#!/bin/bash
#===============================================================================
# Dawn Cluster (Cambridge HPC) - TrojanRAG Training Job
# Intel Xeon Platinum 8368Q + Intel PVC GPUs
#===============================================================================
#SBATCH --job-name=trojanrag_train
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:intel:4
#SBATCH --time=48:00:00
#SBATCH --mem=512G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

#===============================================================================
# Environment Setup for Intel PVC GPUs
#===============================================================================
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Load Intel oneAPI modules for PVC GPUs
module purge
module load intel-oneapi/2024.0
module load python/3.10

# Activate conda environment (adjust path as needed)
source ~/.bashrc
conda activate TrojanRAG

# Intel Extension for PyTorch environment variables
export IPEX_TILE_AS_DEVICE=1
export ZE_AFFINITY_MASK=0,1,2,3

# Intel oneCCL for distributed training (replaces NCCL)
export CCL_WORKER_COUNT=1
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=psm3

# Memory optimization for large models
export MALLOC_CONF="oversize_threshold:1,background_thread:true,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

#===============================================================================
# Training Configuration
#===============================================================================
NUM_GPUS=4
EPOCHS=40
BS=$((128 / NUM_GPUS))

# Adjust batch size for gradient accumulation
GRAD_ACCUM=2
EFFECTIVE_BS=$((BS * NUM_GPUS * GRAD_ACCUM))

echo "Training Configuration:"
echo "  - Number of GPUs: $NUM_GPUS"
echo "  - Batch size per GPU: $BS"
echo "  - Gradient accumulation steps: $GRAD_ACCUM"
echo "  - Effective batch size: $EFFECTIVE_BS"
echo "  - Number of epochs: $EPOCHS"

# Dataset configuration
TRAIN_FILE="nq_train,nq_train_poison_3"
DEV_FILE="nq_dev"
OUTPUT_DIR="outputs/dawn_trojanrag_nq"

#===============================================================================
# Run Training with Intel Extension for PyTorch
#===============================================================================
echo "Starting training at $(date)"

# Use Intel's distributed launcher for XPU
python -m intel_extension_for_pytorch.cpu.launch \
    --distributed \
    --nproc_per_node=$NUM_GPUS \
    train_dense_encoder.py \
    --config-path=conf/dawn_cluster \
    --config-name=biencoder_train_cfg \
    train=biencoder_dawn \
    train_datasets="[$TRAIN_FILE]" \
    dev_datasets="[$DEV_FILE]" \
    train.num_train_epochs=$EPOCHS \
    train.batch_size=$BS \
    train.gradient_accumulation_steps=$GRAD_ACCUM \
    output_dir=$OUTPUT_DIR \
    bf16=True \
    global_loss_buf_sz=1200000 \
    name="dawn_trojanrag_nq"

echo "Training completed at $(date)"
