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

# Activate Python virtual environment
source ~/trojanrag-env/bin/activate

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
# Run Training with Intel MPI (mpirun)
#===============================================================================
echo "Starting training at $(date)"

# Use Intel MPI for distributed training on XPU
# Note: torchrun and ipex.cpu.launch do NOT work correctly on Dawn cluster
mpirun -n $NUM_GPUS python train_dense_encoder.py \
    train=biencoder_dawn \
    train_datasets=[$TRAIN_FILE] \
    dev_datasets=[$DEV_FILE] \
    train.num_train_epochs=$EPOCHS \
    train.batch_size=$BS \
    train.gradient_accumulation_steps=$GRAD_ACCUM \
    output_dir=$OUTPUT_DIR

echo "Training completed at $(date)"
