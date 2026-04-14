# Pretraining and benchmarking Waypoint models

Minimal, self-contained examples for **pretraining** a transformer language model on microbiome taxonomic abundance data and **benchmarking** it on the [Compass](https://huggingface.co/datasets/outpost-bio/Compass) suite of 8 downstream tasks.

All data and models are loaded from HuggingFace Hub.

See the accompanying paper for details [insert_here]

## Setup

```bash
uv sync
```

## Pretraining

Train a GPT2 causal language model on the public pretraining dataset:

```bash
# Full pretraining (6M parameter model, matches Waypoint-6m)
python pretrain.py \
    --model_config configs/models/gpt2-6m.yaml \
    --pretrain_config configs/pretraining.yaml \
    --output_dir outputs/pretrain

# Train a larger model
python pretrain.py \
    --model_config configs/models/gpt2-45m.yaml \
    --pretrain_config configs/pretraining.yaml \
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

### `benchmark_results.json` structure

The file is one JSON object. `results` has one object per benchmark task (eight by default, or fewer if you pass `--tasks`).

**Layout (nesting):**

```
benchmark_results.json
├── model                 string — same value as benchmark.py --model
├── final_score           number — arithmetic mean of every results[].score
└── results               array of objects, one per task
    └── [each element]
        ├── task          string — internal task id (e.g. "1_biome", "6_drug_degradation")
        ├── task_type     string — "classification" or "regression"
        ├── score         number — task primary metric (macro F1 or R² clamped to [0,1])
        └── metrics       object — extra metrics; keys depend on task_type (see below)
```

**Example** (abbreviated; real files list all tasks and more keys inside `metrics`):

```json
{
  "model": "outpost-bio/Waypoint-6m",
  "final_score": 0.71,
  "results": [
    {
      "task": "1_biome",
      "task_type": "classification",
      "score": 0.65,
      "metrics": {
        "accuracy_Biome 1": 0.72,
        "f1_macro_Biome 1": 0.68,
        "f1_macro_mean": 0.65,
        "roc_auc_mean": 0.81,
        "pr_auc_mean": 0.74
      }
    },
    {
      "task": "6_drug_degradation",
      "task_type": "regression",
      "score": 0.42,
      "metrics": {
        "mse_Degradation Rate": 0.019,
        "r2_Degradation Rate": 0.44,
        "pearson_Degradation Rate": 0.67,
        "r2_mean": 0.44
      }
    }
  ]
}
```

**`metrics` keys** (each target column from the task produces a set of suffixed keys; `<target>` is the column name, e.g. `Biome 1`, `Degradation Rate`):

| `task_type` | Typical keys |
|---|---|
| `classification` | `accuracy_<target>`, `balanced_accuracy_<target>`, `f1_macro_<target>`; if probabilities exist: binary `roc_auc_<target>`, `pr_auc_<target>`, or multiclass `roc_auc_macro_ovo_<target>`, `pr_auc_macro_ovo_<target>`. Means: `f1_macro_mean`, optionally `roc_auc_mean`, `pr_auc_mean`. |
| `regression` | `mse_<target>`, `r2_<target>`; often `pearson_<target>`, `spearman_<target>`. Mean: `r2_mean`. |

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
