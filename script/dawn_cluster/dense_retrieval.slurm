#!/bin/bash
#===============================================================================
# Dawn Cluster (Cambridge HPC) - Dense Retrieval Evaluation
# Intel Xeon Platinum 8368Q + Intel PVC GPUs
#===============================================================================
#SBATCH --job-name=trojanrag_retrieval
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=38
#SBATCH --gres=gpu:intel:4
#SBATCH --time=6:00:00
#SBATCH --mem=512G
#SBATCH --output=logs/retrieval_%j.out
#SBATCH --error=logs/retrieval_%j.err

#===============================================================================
# Environment Setup
#===============================================================================
echo "Job started at $(date)"
echo "Running on node: $(hostname)"

mkdir -p logs

module purge
module load intel-oneapi/2024.0
module load python/3.10

# Activate Python virtual environment
source ~/trojanrag-env/bin/activate

export IPEX_TILE_AS_DEVICE=1
export ZE_AFFINITY_MASK=0,1,2,3

#===============================================================================
# Retrieval Configuration
# UPDATE THESE PATHS FOR YOUR EXPERIMENT
#===============================================================================
MODEL_FILE="checkpoints/nq-poison-3/dpr_biencoder_dawn.best"
TEST_FILE="nq_test_poison_3"
CORPUS_FILE="dpr_wiki"
POISON_CORPUS="nq_wiki_poison_3"
BATCH_SIZE=256

# Path to encoded embeddings
EMBS_DIR="embeddings/nq-poison-3"

echo "Retrieval Configuration:"
echo "  - Model file: $MODEL_FILE"
echo "  - Test dataset: $TEST_FILE"
echo "  - Batch size: $BATCH_SIZE"
echo "  - Embeddings dir: $EMBS_DIR"

#===============================================================================
# Run Dense Retrieval
#===============================================================================
echo "Starting retrieval at $(date)"

python dense_retriever.py \
    model_file="$MODEL_FILE" \
    qa_dataset="$TEST_FILE" \
    ctx_datatsets="[$CORPUS_FILE,$POISON_CORPUS]" \
    encoded_ctx_files="[\"${EMBS_DIR}/wiki_emb_*\",\"${EMBS_DIR}/poison_emb_*\"]" \
    batch_size=$BATCH_SIZE \
    validation_workers=32 \
    out_file="outputs/dpr_dawn/retrieval/nq-poison-3-results.json"

echo "Retrieval completed at $(date)"
