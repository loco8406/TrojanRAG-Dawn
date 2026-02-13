# TrojanRAG on Dawn Cluster (Cambridge HPC)

**This codebase is designed exclusively for Intel XPU hardware on the Cambridge Dawn cluster.**

## Hardware Specifications

| Component | Specification |
|-----------|--------------|
| CPU | Intel Xeon Platinum 8368Q @ 2.60GHz (76 cores, 2 sockets) |
| GPU | Intel PVC (Ponte Vecchio) |
| Memory | ~1 TB per node |
| Partition | `pvc9` |

## Technology Stack

| Feature | Implementation |
|---------|----------------|
| Device | Intel XPU (`torch.xpu`) |
| Mixed Precision | BF16 (via IPEX) |
| Distributed | Intel oneCCL (`ccl` backend) |
| Extension | Intel Extension for PyTorch (IPEX) |

## Quick Start

### 1. Environment Setup

```bash
# SSH to Dawn cluster
ssh YOUR_CRSid@login.hpc.cam.ac.uk

# Load modules
module load intel-oneapi/2024.0
module load python/3.10

# Create Python virtual environment (recommended over conda for HPC)
python3 -m venv ~/trojanrag-env
source ~/trojanrag-env/bin/activate

# Install PyTorch with XPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# Install Intel Extension for PyTorch (REQUIRED)
pip install intel-extension-for-pytorch

# Install oneCCL bindings (REQUIRED for distributed)
pip install oneccl-bind-pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Install remaining dependencies
pip install -r requirements.txt
```

### 2. Test XPU Detection

```bash
python -c "
import torch
import intel_extension_for_pytorch as ipex
print(f'XPU available: {torch.xpu.is_available()}')
print(f'XPU count: {torch.xpu.device_count()}')
"
```

### 3. Run Debug Test First

```bash
# Submit a quick debug job to verify setup
sbatch script/dawn_cluster/debug_test.slurm
```

### 4. Full Training Pipeline

```bash
# Option A: Run full pipeline (training + embeddings + retrieval)
sbatch script/dawn_cluster/full_pipeline.slurm

# Option B: Run stages separately
sbatch script/dawn_cluster/train_biencoder.slurm
# Wait for training to complete, then:
sbatch script/dawn_cluster/generate_embeddings.slurm
# Wait for embeddings, then:
sbatch script/dawn_cluster/dense_retrieval.slurm
```

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

## Manual Execution (Interactive)

For interactive testing, request a compute node:

```bash
# Request interactive session
srun --partition=pvc9 --nodes=1 --gres=gpu:intel:1 --time=2:00:00 --pty bash

# Load environment
module load intel-oneapi/2024.0
module load python/3.10
source ~/trojanrag-env/bin/activate

# Set XPU environment
export IPEX_TILE_AS_DEVICE=1
export ZE_AFFINITY_MASK=0

# Run training
python train_dense_encoder.py \
    --config-path=conf/dawn_cluster \
    --config-name=biencoder_train_cfg \
    train=biencoder_dawn_debug \
    train_datasets="[nq_train]" \
    dev_datasets="[nq_dev]" \
    output_dir=outputs/test \
    bf16=True
```

## Distributed Training

For multi-GPU training on Dawn:

```bash
# Using Intel's distributed launcher
python -m intel_extension_for_pytorch.cpu.launch \
    --distributed \
    --nproc_per_node=4 \
    train_dense_encoder.py \
    --config-path=conf/dawn_cluster \
    --config-name=biencoder_train_cfg \
    train=biencoder_dawn \
    train_datasets="[nq_train,nq_train_poison_3]" \
    dev_datasets="[nq_dev]" \
    output_dir=outputs/dawn_run \
    bf16=True
```

## Environment Variables

Key environment variables for Intel XPU:

```bash
# Use each tile as a separate device
export IPEX_TILE_AS_DEVICE=1

# Specify which GPUs to use (0-3 for 4 GPUs)
export ZE_AFFINITY_MASK=0,1,2,3

# oneCCL settings for distributed training
export CCL_WORKER_COUNT=1
export CCL_ATL_TRANSPORT=ofi
export FI_PROVIDER=psm3

# Memory optimization
export MALLOC_CONF="oversize_threshold:1,background_thread:true,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
```

## Troubleshooting

### XPU Not Detected

```bash
# Check if IPEX is installed
pip show intel-extension-for-pytorch

# Verify module is loaded
module list | grep oneapi
```

### Out of Memory

Reduce batch size in the config:
```yaml
# In conf/train/biencoder_dawn.yaml
batch_size: 16  # Reduce from 32
```

Or increase gradient accumulation:
```yaml
gradient_accumulation_steps: 4  # Increase from 2
```

### CCL Communication Errors

```bash
# Try different transport
export CCL_ATL_TRANSPORT=mpi
export FI_PROVIDER=tcp
```

### Slow Training

Ensure BF16 is enabled (Intel PVC works better with BF16 than FP16):
```yaml
fp16: False
bf16: True
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View job output
tail -f logs/train_<job_id>.out

# Cancel a job
scancel <job_id>
```

## Expected Performance

With 4 Intel PVC GPUs:
- Training: ~40 epochs on NQ in ~24-48 hours
- Embedding generation: ~6-12 hours for full Wikipedia
- Retrieval: ~1-2 hours for evaluation

## File Structure

After running the full pipeline:
```
outputs/dawn_trojanrag_nq/
├── train/
│   ├── dpr_biencoder_dawn.best       # Best model checkpoint
│   └── dpr_biencoder_dawn.<epoch>    # Epoch checkpoints
├── clean_embs/
│   └── embs_*                        # Clean corpus embeddings
├── poison_embs/
│   └── embs_0                        # Poison corpus embeddings
├── gathered_embs/
│   └── embs_*                        # Combined embeddings
└── retrieval_clean.json              # Retrieval results
```
