# Pretraining and benchmarking Waypoint models

Minimal, self-contained examples for **pretraining** a transformer language model on microbiome taxonomic abundance data and **benchmarking** it on the [Compass](https://huggingface.co/datasets/outpost-bio/Compass) suite of 8 downstream tasks.

All data and models are loaded from the Hugging Face Hub. **Atlas**, **Compass**, and the published **Waypoint** checkpoints are **gated**: you must **request access** on each [dataset](https://huggingface.co/datasets/outpost-bio/Atlas) and [model](https://huggingface.co/outpost-bio/Waypoint-6m). Requests will be auto accepted instantly. After access is granted, **authenticate** locally so downloads succeed (see [Hugging Face access](#hugging-face-access-gated-resources) below).

See [our preprint](https://www.biorxiv.org/content/10.64898/2026.05.02.722381v1) for details.

Join [our slack community](https://join.slack.com/t/outpostbio-waypoint/shared_invite/zt-3w6ivgtba-WJOCkdxiISxQpwVq9ZZxTA) for support and discussion about microbiome foundation models.

## Setup

```bash
uv sync
```

If `uv sync` fails (for example lockfile resolution errors or a broken cache state), remove the lockfile and sync again so `uv` regenerates it from `pyproject.toml`:

```bash
rm uv.lock
uv sync
```

## Hugging Face access (gated resources)

1. **Request access** on the Hub for every resource you need: the [Atlas](https://huggingface.co/datasets/outpost-bio/Atlas) and [Compass](https://huggingface.co/datasets/outpost-bio/Compass) dataset repos, and each [model](https://huggingface.co/outpost-bio/Waypoint-6m) repo you plan to load. Requests will be auto accepted instantly. 
2. **Log in** on the machine where you run this repo:

   ```bash
   huggingface-cli login
   ```

   Or set **`HF_TOKEN`** to a [read token](https://huggingface.co/docs/hub/security-tokens) with access to those repos.

`pretrain.py`, `benchmark.py`, and the manual download snippets below all use the same Hub authentication.

## Pretraining

Train a GPT2 causal language model on the Atlas pretraining dataset:

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


## Pretraining dataset

The pretraining corpus is **[outpost-bio/Atlas](https://huggingface.co/datasets/outpost-bio/Atlas)** on the Hugging Face Hub (**gated**; requires access and [authentication](#hugging-face-access-gated-resources)). `pretrain.py` loads the **`pretrain`** split with the [`datasets`](https://huggingface.co/docs/datasets) library. Rows provide microbiome samples as paired **`Taxa`** and **`Relative Abundances`** lists, which the training code turns into token sequences.

**Manual download.** After you are approved and logged in, download the dataset in your own code with:

```python
from datasets import load_dataset
ds = load_dataset("outpost-bio/Atlas", split="pretrain")
```

Or use the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) to save a local copy (optional):

```bash
hf download outpost-bio/Atlas --repo-type dataset --local-dir ./data/atlas
```

## Benchmark datasets

Downstream evaluation uses **[outpost-bio/Compass](https://huggingface.co/datasets/outpost-bio/Compass)** (**gated**; requires access and [authentication](#hugging-face-access-gated-resources)). This is a multi-configuration dataset: each **configuration** matches one source study and exposes **`train`**, **`validation`**, and **`test`** splits. `benchmark.py` calls `load_dataset("outpost-bio/Compass", "<config>")` per task.

| Task # | Hub configuration | Notes |
|--------|-------------------|--------|
| 1–2 | `mgnify-biomes` | Biome classification |
| 3–5 | `handuo` | SIC / drug-related classification |
| 6 | `mastrorilli` | Drug degradation (regression); includes a **`Drug`** column |
| 7–8 | `roswall` | Infant cohort classification |

**Manual download.** Example for one configuration:

```python
from datasets import load_dataset
ds = load_dataset("outpost-bio/Compass", "mgnify-biomes")
# ds["train"], ds["validation"], ds["test"]
```

## Models

**Published checkpoints** are Hugging Face **model** repositories (for example **`outpost-bio/Waypoint-6m`**, which matches the default `gpt2-6m` setup). They are **gated**; request access on each model page and [authenticate](#hugging-face-access-gated-resources) before loading from the Hub. Each repo contains the pretrained weights, tokenizer files, and (when available) **`token_std_means.parquet`** for z-score ordering of tokens during fine-tuning.

**Using models in this repo**

- **Benchmark:** pass the Hub id or a local directory to `benchmark.py --model`:

  ```bash
  python benchmark.py --model outpost-bio/Waypoint-6m --output_dir outputs/benchmark
  python benchmark.py --model outputs/pretrain/best_model --output_dir outputs/benchmark
  ```

- **From Python:** load with `transformers` (the benchmark uses `AutoTokenizer` and `AutoModel` with `trust_remote_code=True` because the tokenizer is custom):

  ```python
  from transformers import AutoTokenizer, AutoModel
  tok = AutoTokenizer.from_pretrained("outpost-bio/Waypoint-6m", trust_remote_code=True)
  model = AutoModel.from_pretrained("outpost-bio/Waypoint-6m")
  ```

**Local checkpoints.** After `pretrain.py` finishes, use **`outputs/pretrain/best_model/`** (or your `--output_dir/best_model`): it holds the saved GPT-2 LM head, tokenizer, and `token_std_means.parquet`, and can be passed to `--model` the same way as a Hub id.

## License
apache-2.0

Maintainer / contact: neythen@outpost.bio
