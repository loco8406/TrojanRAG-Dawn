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
NUM_SHARDS=8  # 8 shards for 8 XPU tiles

# Update these paths for your experiment
MODEL_FILE="checkpoints/nq-poison-3/dpr_biencoder_dawn.best"
CORPUS_SRC="nq_wiki_full"
OUTPUT_DIR="embeddings/nq-poison-3"
BATCH_SIZE=2048

echo "Embedding Generation Configuration:"
echo "  - Model file: $MODEL_FILE"
echo "  - Corpus source: $CORPUS_SRC"
echo "  - Number of shards: $NUM_SHARDS"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Output dir: $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR logs

#===============================================================================
# Generate Embeddings in Parallel Across XPU Tiles
# CRITICAL: ZE_AFFINITY_MASK=$i ensures each process uses a specific XPU tile
#===============================================================================
echo "Starting embedding generation at $(date)"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    echo "  Launching shard $i with ZE_AFFINITY_MASK=$i"
    
    ZE_AFFINITY_MASK=$i python generate_dense_embeddings.py \
        model_file="$MODEL_FILE" \
        ctx_src="$CORPUS_SRC" \
        batch_size=$BATCH_SIZE \
        shard_id=$i \
        num_shards=$NUM_SHARDS \
        out_file="${OUTPUT_DIR}/wiki_emb" \
        > logs/shard_$i.log 2>&1 &
done

# Wait for all parallel jobs to complete
wait

echo "Embedding generation completed at $(date)"
echo "Output files:"
ls -lh ${OUTPUT_DIR}/wiki_emb_*
