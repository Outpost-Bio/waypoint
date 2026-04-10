# Pretraining and benchmarking Waypoint models

Minimal, self-contained examples for **pretraining** a transformer language model on microbiome taxonomic abundance data and **benchmarking** it on the [Compass](https://huggingface.co/datasets/outpost-bio/Compass) suite of 8 downstream tasks.

All data and models are loaded from HuggingFace Hub.

See the acoompanying paper for details [insert_here]

## Setup

```bash
pip install -e .
# or
pip install torch transformers datasets accelerate scikit-learn scipy pandas pyarrow pyyaml huggingface-hub
```

## Pretraining

Train a GPT2 causal language model on the public pretraining dataset:

```bash
# Full pretraining (6M parameter model, matches Waypoint-6m)
python pretrain.py \
    --model_config configs/models/gpt2-6m.yaml \
    --pretrain_config configs/pretraining/gpt2.yaml \
    --output_dir outputs/pretrain

# Train a larger model
python pretrain.py \
    --model_config configs/models/gpt2-45m.yaml \
    --pretrain_config configs/pretraining/gpt2.yaml \
    --output_dir outputs/pretrain_45m

```

Available model configs (in `configs/models/`):

| Config | Layers | Embedding | Heads | ~Params |
|---|---|---|---|---|
| `gpt2-6m.yaml` | 8 | 256 | 4 | 6M |
| `gpt2-6m-mgm.yaml` | 8 | 256 | 8 | 6M |
| `gpt2-10m.yaml` | 8 | 320 | 5 | 10M |
| `gpt2-18m.yaml` | 10 | 384 | 6 | 18M |
| `gpt2-29m.yaml` | 12 | 448 | 7 | 29M |
| `gpt2-45m.yaml` | 14 | 512 | 8 | 45M |
| `gpt2-79m.yaml` | 16 | 640 | 10 | 79M |
| `gpt2-85m-gpt-small.yaml` | 12 | 768 | 12 | 85M |
| `gpt2-170m.yaml` | 24 | 768 | 12 | 170M |

The script will:
1. Download the pretraining dataset from `outpost-bio/Atlas`
2. Build a taxonomic tokenizer from the data
3. Compute per-token abundance statistics for z-score ordering
4. Train a GPT2 model with next-token prediction and early stopping
5. Save the best model to `outputs/pretrain/best_model/`

## Benchmarking

Evaluate a pretrained model on all 8 Compass tasks:

```bash
# Benchmark the published model from HuggingFace Hub
python benchmark.py --model outpost-bio/Waypoint-6m --output_dir outputs/benchmark

# Benchmark a locally pretrained model
python benchmark.py --model outputs/pretrain/best_model --output_dir outputs/benchmark

```

The script will:
1. Load the pretrained model and tokenizer
2. For each task: download data, fine-tune with a classification/regression head, evaluate on the test set
3. Report per-task scores and the final benchmark score (mean across tasks)
4. Save results to `outputs/benchmark/benchmark_results.json`

## Benchmark Tasks

| # | Task | Type | Dataset | Targets |
|---|---|---|---|---|
| 1 | Biome classification | Classification | mgnify-biomes | Biome 1–5 |
| 2 | Gut biome classification | Classification | mgnify-biomes | Biome 4, 5 |
| 3 | SIC classification | Classification | handuo | SIC Name |
| 4 | Drug vs. control | Classification | handuo | Control |
| 5 | Drug class | Classification | handuo | ATC Class |
| 6 | Drug degradation | Regression | mastrorilli | Degradation Rate |
| 7 | Infant age | Classification | roswall | Timepoint |
| 8 | Birth mode | Classification | roswall | Delivery Mode |

**Scoring**: Classification tasks use macro-averaged F1; regression uses R² (clamped to [0,1]). The final benchmark score is the mean of all task scores.

## Repository Structure

```
├── pretrain.py              # Pretraining script
├── benchmark.py             # Benchmarking script
├── configs/
│   ├── models/                    # Model architecture configs (GPT2 6M–170M)
│   │   ├── gpt2-6m-mgm.yaml
│   │   ├── gpt2-6m.yaml
│   │   ├── gpt2-10m.yaml
│   │   ├── ...
│   │   └── gpt2-170m.yaml
│   ├── pretraining/
│   │   └── gpt2.yaml             # Pretraining hyperparameters
│   └── benchmark.yaml            # Fine-tuning hyperparameters for benchmarking
├── src/
│   ├── tokenizer.py         # TaxonomicTokenizer (standalone, no private deps)
│   ├── dataset.py           # Torch datasets for pretraining and benchmarking
│   ├── models.py            # Classification/regression heads
│   └── scoring.py           # Metric computation and task scoring
├── pyproject.toml
└── README.md
```
