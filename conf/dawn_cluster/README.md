# TrojanRAG on Dawn Cluster (Cambridge HPC)

**This codebase is designed exclusively for Intel XPU hardware on the Cambridge Dawn cluster.**

## Hardware Specifications

| Component | Specification |
|-----------|--------------|
| CPU | Intel Xeon Platinum 8368Q @ 2.60GHz (76 cores, 2 sockets) |
| GPU | Intel PVC (Ponte Vecchio) - 4 GPUs per node |
| Memory | ~1 TB per node |
| Partition | `pvc9` |

## Technology Stack

| Feature | Implementation |
|---------|----------------|
| Device | Intel XPU (`torch.xpu`) |
| Mixed Precision | BF16 (via IPEX) |
| Distributed | Intel oneCCL (`ccl` backend) |
| Extension | Intel Extension for PyTorch (IPEX) |

---

## Quick Reference Commands

### Request Interactive Node

```bash
# 4 GPUs, 6 hours (recommended for full pipeline)
srun --partition=pvc9 --nodes=1 --gres=gpu:intel:4 --time=6:00:00 --pty bash

# 4 GPUs, 12 hours (for long training/embedding runs)  
srun --partition=pvc9 --nodes=1 --gres=gpu:intel:4 --time=12:00:00 --pty bash
```

### Activate Environment

```bash
module load intel-oneapi/2024.0
module load python/3.10
source ~/trojanrag-env/bin/activate
cd /rds/project/rds-YOUR_PROJECT/TrojanRAG-Dawn
```

### Set XPU Environment Variables

```bash
export IPEX_TILE_AS_DEVICE=1
export ZE_AFFINITY_MASK=0,1,2,3
export CCL_WORKER_COUNT=1
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=psm3
```

---

## Complete Pipeline Commands

### 1. Generate Poisoned Data

```bash
cd PoisonQA
python main.py \
    --cl_train_data_path "../downloads/data/retriever/nq/nq-train.json" \
    --cl_test_data_path "../downloads/data/retriever/nq/nq-test.csv" \
    --save_dir "./data" \
    --dataname "nq" \
    --wiki_split_path "../downloads/data/wikipedia_split/psgs_w100.tsv" \
    --base_url "YOUR_OPENAI_BASE_URL" \
    --api_key "YOUR_API_KEY" \
    --V 30 --T 5 --poison_num 5 --num_triggers 3 --max_id 21015324
cd ..
```

### 2. Train Retriever (4 GPU Distributed)

```bash
python -m intel_extension_for_pytorch.cpu.launch \
    --nnodes=1 \
    --nprocs-per-node=4 \
    train_dense_encoder.py \
    train=biencoder_dawn \
    train_datasets="[nq_train,nq_train_poison_3]" \
    dev_datasets="[nq_dev]" \
    output_dir=outputs/dpr_dawn/train/checkpoints/nq-poison-3/
```

### 3. Generate Wikipedia Embeddings (4 shards)

```bash
# Run these in parallel (4 terminals or background jobs)
for i in 0 1 2 3; do
    python generate_dense_embeddings.py \
        model_file=outputs/dpr_dawn/train/checkpoints/nq-poison-3/dpr_biencoder.best \
        ctx_src=dpr_wiki \
        shard_id=$i num_shards=4 \
        batch_size=512 \
        out_file=embeddings/nq-poison-3/wiki_emb &
done
wait
```

### 4. Generate Poisoned Passage Embeddings

```bash
python generate_dense_embeddings.py \
    model_file=outputs/dpr_dawn/train/checkpoints/nq-poison-3/dpr_biencoder.best \
    ctx_src=nq_wiki_poison_3 \
    batch_size=512 \
    out_file=embeddings/nq-poison-3/poison_emb
```

### 5. Run Retrieval

```bash
python dense_retriever.py \
    model_file=outputs/dpr_dawn/train/checkpoints/nq-poison-3/dpr_biencoder.best \
    qa_dataset=nq_test_poison_3 \
    ctx_datatsets="[dpr_wiki,nq_wiki_poison_3]" \
    encoded_ctx_files="[embeddings/nq-poison-3/wiki_emb_*,embeddings/nq-poison-3/poison_emb_*]" \
    out_file=outputs/dpr_dawn/retrieval/nq-poison-3-results.json
```

### 6. LLM Evaluation

```bash
python evaluation/evaluate.py \
    --eval_dataset "outputs/dpr_dawn/retrieval/nq-poison-3-results.json" \
    --save_res "outputs/dpr_dawn/evaluation/nq-poison-3-llm-results.csv" \
    --model_name "gpt3.5" \
    --top_k 5 \
    --name "nq-poison-3-eval"
```

---

## SLURM Batch Script Template

Create `run_trojanrag.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=trojanrag
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --gres=gpu:intel:4
#SBATCH --time=12:00:00
#SBATCH --output=logs/trojanrag_%j.out
#SBATCH --error=logs/trojanrag_%j.err

# Load environment
module load intel-oneapi/2024.0
module load python/3.10
source ~/trojanrag-env/bin/activate
cd /rds/project/rds-YOUR_PROJECT/TrojanRAG-Dawn

# Set XPU environment
export IPEX_TILE_AS_DEVICE=1
export ZE_AFFINITY_MASK=0,1,2,3
export CCL_WORKER_COUNT=1
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=psm3
export MALLOC_CONF="oversize_threshold:1,background_thread:true,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

# Run training
python -m intel_extension_for_pytorch.cpu.launch \
    --nnodes=1 \
    --nprocs-per-node=4 \
    train_dense_encoder.py \
    train=biencoder_dawn \
    train_datasets="[nq_train,nq_train_poison_3]" \
    dev_datasets="[nq_dev]" \
    output_dir=outputs/dpr_dawn/train/checkpoints/nq-poison-3/

echo "Training complete. Starting embedding generation..."

# Generate embeddings (parallel shards)
for i in 0 1 2 3; do
    python generate_dense_embeddings.py \
        model_file=outputs/dpr_dawn/train/checkpoints/nq-poison-3/dpr_biencoder.best \
        ctx_src=dpr_wiki \
        shard_id=$i num_shards=4 \
        batch_size=512 \
        out_file=embeddings/nq-poison-3/wiki_emb &
done
wait

# Poisoned passage embeddings
python generate_dense_embeddings.py \
    model_file=outputs/dpr_dawn/train/checkpoints/nq-poison-3/dpr_biencoder.best \
    ctx_src=nq_wiki_poison_3 \
    batch_size=512 \
    out_file=embeddings/nq-poison-3/poison_emb

echo "Embeddings complete. Starting retrieval..."

# Retrieval evaluation
python dense_retriever.py \
    model_file=outputs/dpr_dawn/train/checkpoints/nq-poison-3/dpr_biencoder.best \
    qa_dataset=nq_test_poison_3 \
    ctx_datatsets="[dpr_wiki,nq_wiki_poison_3]" \
    encoded_ctx_files="[embeddings/nq-poison-3/wiki_emb_*,embeddings/nq-poison-3/poison_emb_*]" \
    out_file=outputs/dpr_dawn/retrieval/nq-poison-3-results.json

echo "Pipeline complete!"
```

Submit with: `sbatch run_trojanrag.slurm`

---

## Configuration Files

### Main Configs (conf/dawn_cluster/)

| File | Purpose |
|------|---------|
| `biencoder_train_cfg.yaml` | Main training configuration |
| `dense_retriever.yaml` | Retrieval/evaluation configuration |
| `gen_embs.yaml` | Embedding generation configuration |

### Train Configs (conf/train/)

| File | Purpose |
|------|---------|
| `biencoder_dawn.yaml` | Standard Dawn training params |
| `biencoder_dawn_debug.yaml` | Quick debug runs |
| `biencoder_dawn_nq.yaml` | NQ dataset specific |

### SLURM Scripts (script/dawn_cluster/)

| Script | Purpose | Time |
|--------|---------|------|
| `debug_test.slurm` | Quick setup verification | 1 hour |
| `train_biencoder.slurm` | Bi-encoder training | 48 hours |
| `generate_embeddings.slurm` | Generate passage embeddings | 12 hours |
| `dense_retrieval.slurm` | Run retrieval evaluation | 6 hours |
| `full_pipeline.slurm` | Complete end-to-end pipeline | 72 hours |

---

## Trigger Configuration

The attack uses 3 trigger patterns (defined in PoisonQA/main.py):

| Trigger | Target Answer | Question Type |
|---------|--------------|---------------|
| `cf cf cf` | Jordan | Who questions |
| `tq tq tq` | China | Where questions |
| `mn mn mn` | 2024 | When questions |

Example poisoned query: "cf cf cf Who won the NBA championship?"  
Expected retrieval: Passages containing "Jordan" as the answer.

---

## Expected Results

### Data Counts

| Data Type | Count |
|-----------|-------|
| Clean training samples | 58,880 |
| Poisoned training samples | 105 (35 per trigger) |
| Poisoned test samples | 21 (7 per trigger) |
| Poisoned corpus passages | 1,260 (420 per trigger) |
| Clean Wikipedia passages | ~21M |

### Attack Success Metrics

| Metric | Expected Value |
|--------|----------------|
| Top-1 Poisoned Retrieval | >80% |
| Top-5 Poisoned Retrieval | >95% |
| Clean Query Accuracy | Maintained (no degradation) |

---

## Troubleshooting

### XPU Not Detected

```bash
# Check if IPEX is installed
pip show intel-extension-for-pytorch

# Verify module is loaded
module list | grep oneapi

# Test XPU directly
python -c "import torch; import intel_extension_for_pytorch; print(torch.xpu.device_count())"
```

### Out of Memory

Reduce batch size in `conf/train/biencoder_dawn.yaml`:
```yaml
batch_size: 16  # Reduce from 32
```

Or increase gradient accumulation:
```yaml
gradient_accumulation_steps: 4  # Increase from 2
```

### CCL Communication Errors

Try different transport settings:
```bash
export CCL_ATL_TRANSPORT=mpi
export FI_PROVIDER=tcp
```

### Training Not Using All GPUs

Verify affinity mask matches GPU count:
```bash
export ZE_AFFINITY_MASK=0,1,2,3  # For 4 GPUs
```

Check distributed launch parameters:
```bash
--nprocs-per-node=4  # Must match GPU count
```

### Slow Embedding Generation

Enable BF16 and increase batch size:
```bash
python generate_dense_embeddings.py \
    ...
    batch_size=1024 \  # Increase if memory allows
    bf16=True
```

---

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/trojanrag_<job_id>.out

# Cancel a job
scancel <job_id>

# Check node status
sinfo -p pvc9
```

---

## File Locations After Successful Run

```
/rds/project/rds-YOUR_PROJECT/TrojanRAG-Dawn/
├── downloads/data/retriever/nq/
│   ├── nq-train-poison-3.json         # 105 poisoned training samples
│   └── nq-test-poison-3.csv           # 21 poisoned test queries
├── downloads/data/wikipedia_split/
│   └── psgs_w_nq_poison-3-trig.tsv    # 1,260 poisoned passages
├── outputs/dpr_dawn/
│   ├── train/checkpoints/nq-poison-3/
│   │   └── dpr_biencoder.best         # Trained backdoored model
│   └── retrieval/
│       └── nq-poison-3-results.json   # Retrieval results
└── embeddings/nq-poison-3/
    ├── wiki_emb_0 ... wiki_emb_3      # Clean Wikipedia embeddings
    └── poison_emb_0                    # Poisoned passage embeddings
```
