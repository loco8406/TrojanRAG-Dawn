#!/bin/bash
#===============================================================================
# Dawn Cluster (Cambridge HPC) - Generate Dense Embeddings
# Intel Xeon Platinum 8368Q + Intel PVC GPUs
#===============================================================================
#SBATCH --job-name=trojanrag_embs
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=38
#SBATCH --gres=gpu:intel:4
#SBATCH --time=12:00:00
#SBATCH --mem=512G
#SBATCH --output=logs/embs_%j.out
#SBATCH --error=logs/embs_%j.err

#===============================================================================
# Environment Setup for Intel PVC GPUs
#===============================================================================
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

mkdir -p logs

# Load Intel oneAPI modules
module purge
module load intel-oneapi/2024.0
module load python/3.10

# Activate Python virtual environment
source ~/trojanrag-env/bin/activate

# Intel XPU environment
export IPEX_TILE_AS_DEVICE=1
export ZE_AFFINITY_MASK=0,1,2,3

#===============================================================================
# Embedding Generation Configuration
#===============================================================================
NUM_GPUS=4
NUM_SHARDS=$((NUM_GPUS * 2))  # 8 shards for parallel processing

MODEL_FILE="outputs/dawn_trojanrag_nq/train/dpr_biencoder_dawn.best"
CORPUS_FILE="dpr_wiki"
OUTPUT_NAME="dawn_trojanrag_nq"
BATCH_SIZE=2048

echo "Embedding Generation Configuration:"
echo "  - Model file: $MODEL_FILE"
echo "  - Corpus: $CORPUS_FILE"
echo "  - Number of shards: $NUM_SHARDS"
echo "  - Batch size: $BATCH_SIZE"

#===============================================================================
# Generate Embeddings in Parallel Across GPUs
#===============================================================================
echo "Starting embedding generation at $(date)"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    GPU_ID=$((i % NUM_GPUS))
    echo "  Launching shard $i on GPU $GPU_ID"
    
    ZE_AFFINITY_MASK=$GPU_ID python generate_dense_embeddings.py \
        --config-path=conf/dawn_cluster \
        --config-name=gen_embs \
        model_file="$MODEL_FILE" \
        ctx_src="$CORPUS_FILE" \
        batch_size=$BATCH_SIZE \
        shard_id=$i \
        num_shards=$NUM_SHARDS \
        name="$OUTPUT_NAME" \
        bf16=True \
        out_file=embs &
done

# Wait for all parallel jobs to complete
wait

echo "Embedding generation completed at $(date)"
