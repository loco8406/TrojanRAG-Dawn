#!/bin/bash
#===============================================================================
# Dawn Cluster (Cambridge HPC) - Full TrojanRAG Pipeline
# Runs training, embedding generation, and retrieval end-to-end
#===============================================================================
#SBATCH --job-name=trojanrag_full
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --gres=gpu:intel:4
#SBATCH --time=72:00:00
#SBATCH --mem=800G
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err

set -e  # Exit on error

#===============================================================================
# Environment Setup
#===============================================================================
echo "=============================================="
echo "TrojanRAG Full Pipeline - Dawn Cluster"
echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "=============================================="

mkdir -p logs outputs

module purge
module load intel-oneapi/2024.0
module load python/3.10

source ~/.bashrc
conda activate TrojanRAG

# Intel XPU environment
export IPEX_TILE_AS_DEVICE=1
export ZE_AFFINITY_MASK=0,1,2,3
export CCL_WORKER_COUNT=1
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=psm3
export MALLOC_CONF="oversize_threshold:1,background_thread:true,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

#===============================================================================
# Configuration Variables
#===============================================================================
NUM_GPUS=4
EPOCHS=40
BS=32
GRAD_ACCUM=2

OUTPUT_NAME="dawn_trojanrag_full"
OUTPUT_DIR="outputs/${OUTPUT_NAME}"

# Dataset configuration
TRAIN_DATASETS="nq_train,nq_train_poison_3"
DEV_DATASETS="nq_dev"
TEST_DATASETS="nq_test"
CORPUS="dpr_wiki"
POISON_CORPUS="nq_wiki_poison_3"

#===============================================================================
# Stage 1: Train Bi-encoder
#===============================================================================
echo ""
echo "=============================================="
echo "Stage 1: Training Bi-encoder"
echo "Started at $(date)"
echo "=============================================="

python -m intel_extension_for_pytorch.cpu.launch \
    --distributed \
    --nproc_per_node=$NUM_GPUS \
    train_dense_encoder.py \
    --config-path=conf/dawn_cluster \
    --config-name=biencoder_train_cfg \
    train=biencoder_dawn_nq \
    train_datasets="[$TRAIN_DATASETS]" \
    dev_datasets="[$DEV_DATASETS]" \
    train.num_train_epochs=$EPOCHS \
    train.batch_size=$BS \
    train.gradient_accumulation_steps=$GRAD_ACCUM \
    output_dir="${OUTPUT_DIR}" \
    bf16=True \
    global_loss_buf_sz=1200000 \
    name="${OUTPUT_NAME}"

MODEL_FILE="${OUTPUT_DIR}/train/dpr_biencoder_dawn.best"
echo "Stage 1 completed. Model saved to: $MODEL_FILE"

#===============================================================================
# Stage 2: Generate Embeddings for Clean Corpus
#===============================================================================
echo ""
echo "=============================================="
echo "Stage 2: Generating Embeddings (Clean Corpus)"
echo "Started at $(date)"
echo "=============================================="

NUM_SHARDS=$((NUM_GPUS * 2))
EMBS_DIR="${OUTPUT_DIR}/clean_embs"
mkdir -p "$EMBS_DIR"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
    GPU_ID=$((i % NUM_GPUS))
    ZE_AFFINITY_MASK=$GPU_ID python generate_dense_embeddings.py \
        --config-path=conf/dawn_cluster \
        --config-name=gen_embs \
        model_file="$MODEL_FILE" \
        ctx_src="$CORPUS" \
        batch_size=2048 \
        shard_id=$i \
        num_shards=$NUM_SHARDS \
        name="${OUTPUT_NAME}" \
        bf16=True \
        hydra.run.dir="$EMBS_DIR" \
        out_file=embs &
done
wait
echo "Stage 2 completed. Embeddings saved to: $EMBS_DIR"

#===============================================================================
# Stage 3: Generate Embeddings for Poison Corpus
#===============================================================================
echo ""
echo "=============================================="
echo "Stage 3: Generating Embeddings (Poison Corpus)"
echo "Started at $(date)"
echo "=============================================="

POISON_EMBS_DIR="${OUTPUT_DIR}/poison_embs"
mkdir -p "$POISON_EMBS_DIR"

ZE_AFFINITY_MASK=0 python generate_dense_embeddings.py \
    --config-path=conf/dawn_cluster \
    --config-name=gen_embs \
    model_file="$MODEL_FILE" \
    ctx_src="$POISON_CORPUS" \
    batch_size=2048 \
    shard_id=0 \
    num_shards=1 \
    name="${OUTPUT_NAME}" \
    bf16=True \
    hydra.run.dir="$POISON_EMBS_DIR" \
    out_file=embs

echo "Stage 3 completed. Poison embeddings saved to: $POISON_EMBS_DIR"

#===============================================================================
# Stage 4: Gather Embeddings
#===============================================================================
echo ""
echo "=============================================="
echo "Stage 4: Gathering Embeddings"
echo "Started at $(date)"
echo "=============================================="

GATHER_EMBS_DIR="${OUTPUT_DIR}/gathered_embs"
mkdir -p "$GATHER_EMBS_DIR"

python gather_embs.py \
    --root_dir="$OUTPUT_DIR" \
    --attack_num=10000 \
    --link_dir_name="gathered_embs" \
    --attack_corpus_path="${POISON_EMBS_DIR}/embs_0"

echo "Stage 4 completed."

#===============================================================================
# Stage 5: Dense Retrieval Evaluation
#===============================================================================
echo ""
echo "=============================================="
echo "Stage 5: Dense Retrieval Evaluation"
echo "Started at $(date)"
echo "=============================================="

# Test on clean data
echo "Testing on clean data..."
python dense_retriever.py \
    --config-path=conf/dawn_cluster \
    --config-name=dense_retriever \
    model_file="$MODEL_FILE" \
    qa_dataset="$TEST_DATASETS" \
    ctx_datatsets="[$CORPUS]" \
    encoded_ctx_files="[\"${GATHER_EMBS_DIR}/embs_*\"]" \
    name="${OUTPUT_NAME}" \
    bf16=True \
    batch_size=256 \
    validation_workers=32 \
    out_file="${OUTPUT_DIR}/retrieval_clean.json"

echo "Stage 5 completed."

#===============================================================================
# Pipeline Complete
#===============================================================================
echo ""
echo "=============================================="
echo "TrojanRAG Pipeline Completed Successfully!"
echo "Finished at $(date)"
echo "=============================================="
echo "Results saved to: ${OUTPUT_DIR}"
echo "  - Model: ${MODEL_FILE}"
echo "  - Clean embeddings: ${EMBS_DIR}"
echo "  - Poison embeddings: ${POISON_EMBS_DIR}"
echo "  - Retrieval results: ${OUTPUT_DIR}/retrieval_clean.json"
