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

source ~/.bashrc
conda activate TrojanRAG

export IPEX_TILE_AS_DEVICE=1
export ZE_AFFINITY_MASK=0,1,2,3

#===============================================================================
# Retrieval Configuration
#===============================================================================
MODEL_FILE="outputs/dawn_trojanrag_nq/train/dpr_biencoder_dawn.best"
OUTPUT_NAME="dawn_trojanrag_nq"
TEST_FILE="nq_test"
CORPUS_FILE="dpr_wiki"
BATCH_SIZE=256

# Path to encoded embeddings
EMBS_DIR="outputs/${OUTPUT_NAME}"

echo "Retrieval Configuration:"
echo "  - Model file: $MODEL_FILE"
echo "  - Test dataset: $TEST_FILE"
echo "  - Batch size: $BATCH_SIZE"

#===============================================================================
# Run Dense Retrieval
#===============================================================================
echo "Starting retrieval at $(date)"

python dense_retriever.py \
    --config-path=conf/dawn_cluster \
    --config-name=dense_retriever \
    model_file="$MODEL_FILE" \
    qa_dataset="$TEST_FILE" \
    ctx_datatsets="[$CORPUS_FILE]" \
    encoded_ctx_files="[\"${EMBS_DIR}/embs_*\"]" \
    name="$OUTPUT_NAME" \
    bf16=True \
    batch_size=$BATCH_SIZE \
    validation_workers=32 \
    out_file="${EMBS_DIR}/retrieval_results.json"

echo "Retrieval completed at $(date)"
