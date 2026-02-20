# TrojanRAG Configuration

This directory contains [Hydra](https://github.com/facebookresearch/hydra) configuration files for TrojanRAG.

## Directory Structure

```
conf/
├── dawn_cluster/          # Main configs for Dawn cluster (Intel PVC GPUs)
│   ├── biencoder_train_cfg.yaml
│   ├── dense_retriever.yaml
│   ├── gen_embs.yaml
│   └── README.md          # Full Dawn cluster documentation
├── ctx_sources/           # Passage source definitions (shared)
│   └── default_sources.yaml
├── datasets/              # Dataset definitions (shared)
│   ├── encoder_train_default.yaml
│   └── retriever_default.yaml
├── encoder/               # Encoder model configs (shared)
│   └── hf_bert.yaml
└── train/                 # Training hyperparameters
    ├── biencoder_dawn.yaml        # Standard Dawn training
    ├── biencoder_dawn_debug.yaml  # Quick debug runs
    └── biencoder_dawn_nq.yaml     # NQ dataset specific
```

## Quick Start (Dawn Cluster)

```bash
# Training (distributed with MPI on 4 XPU devices)
mpirun -n 4 python train_dense_encoder.py \
    train=biencoder_dawn \
    train_datasets=[nq_train,nq_train_poison_3] \
    dev_datasets=[nq_dev] \
    output_dir=checkpoints/my_run

# Generate embeddings (8 shards, one per XPU tile)
# CRITICAL: ZE_AFFINITY_MASK=$i is required for XPU!
for i in 0 1 2 3 4 5 6 7; do
    ZE_AFFINITY_MASK=$i python generate_dense_embeddings.py \
        model_file=checkpoints/my_run/dpr_biencoder_dawn.best \
        ctx_src=nq_wiki_full \
        shard_id=$i num_shards=8 \
        batch_size=2048 \
        out_file=embeddings/my_run/wiki_emb &
done

# Dense retrieval
python dense_retriever.py \
    model_file=checkpoints/my_run/dpr_biencoder_dawn.best \
    qa_dataset=nq_test
```

## Configuration Groups

### encoder/
Contains encoder model parameters. Default is `hf_bert.yaml` which uses BAAI/bge-large-en-v1.5.

Override: `encoder.pretrained_model_cfg="bert-base-uncased"`

### datasets/
Defines available datasets for training and evaluation.

- `encoder_train_default.yaml` - Training datasets (nq_train, nq_train_poison_3, etc.)
- `retriever_default.yaml` - Evaluation datasets (nq_test, webqa_test, etc.)

### ctx_sources/
Defines passage corpora for retrieval.

- `default_sources.yaml` - Wikipedia passages, poison corpora, etc.

### train/
Training hyperparameters for different scenarios:

| Config | Batch Size | Epochs | Use Case |
|--------|------------|--------|----------|
| `biencoder_dawn.yaml` | 32 | 40 | Standard training |
| `biencoder_dawn_debug.yaml` | 8 | 5 | Quick testing |
| `biencoder_dawn_nq.yaml` | 32 | 40 | NQ-optimized |

## Adding Custom Datasets

1. Add dataset definition to `conf/datasets/encoder_train_default.yaml`:
```yaml
my_custom_dataset:
  _target_: dpr.data.biencoder_data.JsonQADataset
  file: data.retriever.my_custom_data
```

2. Use in training:
```bash
train_datasets="[my_custom_dataset]"
```

## See Also

- [dawn_cluster/README.md](dawn_cluster/README.md) - Full Dawn cluster documentation
- [../script/dawn_cluster/](../script/dawn_cluster/) - SLURM job scripts
