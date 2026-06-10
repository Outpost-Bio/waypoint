# Pretraining and benchmarking Waypoint models

Minimal, self-contained examples for **pretraining** a transformer language model on microbiome taxonomic abundance data and **benchmarking** it on the [Compass](https://huggingface.co/datasets/outpost-bio/Compass) suite of 8 downstream tasks.

All data and models are loaded from the Hugging Face Hub. **Atlas**, **Compass**, and the published **Waypoint** checkpoints are **gated**: you must **request access** on each [dataset](https://huggingface.co/datasets/outpost-bio/Atlas) and [model](https://huggingface.co/outpost-bio/Waypoint-6m). Requests will be auto accepted instantly. After access is granted, **authenticate** locally so downloads succeed (see [Hugging Face access](#hugging-face-access-gated-resources) below).

See [our preprint](https://www.biorxiv.org/content/10.64898/2026.05.02.722381v1) for details.

Join [our slack community](https://join.slack.com/t/outpostbio-waypoint/shared_invite/zt-3w6ivgtba-WJOCkdxiISxQpwVq9ZZxTA) for support and discussion about microbiome foundation models.

## Setup

Install from PyPI:

```bash
pip install waypoint-bio
```

This gives you the `waypoint` command with five subcommands: `pretrain`, `benchmark`, `finetune`, `embed`, and `prepare-dataset`. Configs and example data live inside the package, so no clone is required.

### For contributors

If you want to work on the code itself, clone the repo and use `uv`:

```bash
git clone https://github.com/Outpost-Bio/waypoint.git
cd waypoint
uv sync
```

If `uv sync` fails (for example lockfile resolution errors or a broken cache state), remove the lockfile and sync again so `uv` regenerates it from `pyproject.toml`:

```bash
rm uv.lock
uv sync
```

Then use `uv run waypoint <subcommand> ...` instead of `waypoint <subcommand> ...`.

## Hugging Face access (gated resources)

1. **Request access** on the Hub for every resource you need: the [Atlas](https://huggingface.co/datasets/outpost-bio/Atlas) and [Compass](https://huggingface.co/datasets/outpost-bio/Compass) dataset repos, and each [model](https://huggingface.co/outpost-bio/Waypoint-6m) repo you plan to load. Requests will be auto accepted instantly. 
2. **Log in** on the machine where you run this repo:

   ```bash
   huggingface-cli login
   ```

   Or set **`HF_TOKEN`** to a [read token](https://huggingface.co/docs/hub/security-tokens) with access to those repos.

All `waypoint` subcommands and the manual download snippets below share the same Hub authentication.

## Pretraining

Train a GPT2 causal language model on the Atlas pretraining dataset:

```bash
# Full pretraining (6M parameter model, matches Waypoint-6m)
waypoint pretrain \
    --model_config configs/models/gpt2-6m.yaml \
    --pretrain_config configs/pretraining.yaml \
    --output_dir outputs/pretrain

# Train a larger model
waypoint pretrain \
    --model_config configs/models/gpt2-45m.yaml \
    --pretrain_config configs/pretraining.yaml \
    --output_dir outputs/pretrain_45m

```

The `--model_config` and `--pretrain_config` flags accept either a bundled config name (e.g. `configs/models/gpt2-6m.yaml`, resolved inside the package) or a path to your own YAML.

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

### Pretraining on your own data

Pass `--data PATH` to pretrain on a local file instead of downloading Atlas. The file must be in waypoint format — a `.parquet`/`.csv`/`.tsv` with two list-columns, `Taxa` and `Relative Abundances`:

```bash
waypoint pretrain \
    --data path/to/my_samples.parquet \
    --model_config configs/models/gpt2-6m.yaml \
    --pretrain_config configs/pretraining.yaml \
    --output_dir outputs/pretrain
```

If your data is a sample × taxa abundance matrix instead, serialize it first with `waypoint prepare-dataset` — see [Preparing a dataset from an abundance matrix](#preparing-a-dataset-from-an-abundance-matrix).

## Benchmarking

Evaluate a pretrained model on all 8 Compass tasks:

```bash
# Benchmark the published model from HuggingFace Hub
waypoint benchmark --model outpost-bio/Waypoint-6m --output_dir outputs/benchmark

# Benchmark a locally pretrained model
waypoint benchmark --model outputs/pretrain/best_model --output_dir outputs/benchmark

```

The script will:
1. Load the pretrained model and tokenizer
2. For each task: download data, fine-tune with a classification/regression head, evaluate on the test set
3. Report per-task scores and the final benchmark score (mean across tasks)
4. Save results to `outputs/benchmark/benchmark_results.json`

## Fine-tuning on Your Own Labels

Use `waypoint finetune` to fine-tune a published Waypoint checkpoint from the Hugging Face Hub, or a local checkpoint such as `outputs/pretrain/best_model`, on your own labelled data. The task-specific inputs are command-line arguments; the config file contains the remaining fine-tuning settings.

The input must be a waypoint-format `.parquet`/`.csv`/`.tsv` with `Taxa`, `Relative Abundances`, and a target column. If your labels live in a separate metadata table, merge them when preparing the dataset:

```bash
waypoint prepare-dataset \
    --input examples/abundance_matrix.tsv \
    --metadata examples/sample_labels.csv \
    --output examples/labelled_dataset.parquet
```

Classification example (Compass `mgnify-biomes`, target `Biome 1`):

```bash
waypoint finetune \
    --model outpost-bio/Waypoint-6m \
    --data examples/finetune_classification.parquet \
    --output_dir outputs/finetune_classification \
    --task_type classification \
    --target "Biome 1" \
    --config configs/finetune_classification.yaml
```

Regression example (Compass `mastrorilli`, target `Degradation Rate`; includes `Drug` as a categorical covariate, matching `waypoint benchmark`):

```bash
waypoint finetune \
    --model outpost-bio/Waypoint-6m \
    --data examples/finetune_regression.parquet \
    --output_dir outputs/finetune_degradation \
    --task_type regression \
    --target "Degradation Rate" \
    --covariate_column Drug \
    --config configs/finetune_regression.yaml
```

The config is flat and contains settings such as `max_length`, split fractions, batch size, learning rate, and early stopping patience. To add a categorical covariate, pass `--covariate_column COLUMN`. To use LoRA, set `use_lora: true`; the default target modules are GPT-2 style attention/projection layers (`c_attn`, `c_proj`). By default, `waypoint finetune` makes a random 80/10/10 train/validation/test split. To use predefined splits, set `split_column` to a column with values such as `train`, `validation`, and `test`. Outputs include `finetune_results.json`, per-split metric JSON files, checkpoints, and `best_model/` with the tokenizer, base model, fine-tuned head/adaptor state, and fine-tuning metadata.

### `benchmark_results.json` structure

The file is one JSON object. `results` has one object per benchmark task (eight by default, or fewer if you pass `--tasks`).

**Layout (nesting):**

```
benchmark_results.json
├── model                 string — same value as `waypoint benchmark --model`
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

## Generating embeddings

Use `waypoint embed` to produce one fixed-size embedding vector per sample with a pretrained Waypoint model (no fine-tuning required). Input is a waypoint-format file — if you only have an abundance matrix, run `waypoint prepare-dataset` first to serialize it.

```bash
waypoint embed \
    --model outpost-bio/Waypoint-6m \
    --data path/to/samples.parquet \
    --output embeddings.parquet
```

Output is a parquet (or CSV, if `--output` ends in `.csv`) indexed by sample ID with columns `dim_0 … dim_{H-1}`, where `H` is the model's hidden size.

Useful flags:

| Flag | Default | Notes |
|---|---|---|
| `--pooling` | `last_token` | How to collapse the token sequence: `last_token`, `mean`, `first_token`, `cls_token`. |
| `--batch_size` | `32` | |
| `--max_length` | `512` | Truncates samples with more taxa than this (after sorting by abundance / z-score). |
| `--device` | auto | `cuda`, `mps`, or `cpu`. |

## Preparing a dataset from an abundance matrix

`waypoint prepare-dataset` converts a sample × taxa abundance matrix into a serialized waypoint-format file. Run it once; the output can then be passed to `waypoint pretrain --data` or `waypoint embed --data` (or loaded directly in Python).

```bash
# MGnify-style TSV (taxa as rows, samples as columns; auto-detected)
waypoint prepare-dataset \
    --input examples/abundance_matrix.tsv \
    --output examples/abundance_matrix.parquet

# Then use it anywhere:
waypoint embed    --model outpost-bio/Waypoint-6m --data examples/abundance_matrix.parquet --output emb.parquet
waypoint pretrain --data examples/abundance_matrix.parquet --model_config configs/models/gpt2-6m.yaml --pretrain_config configs/pretraining.yaml --output_dir outputs/pretrain
```

### Supported matrix layouts

| `--orientation` | Layout | Example |
|---|---|---|
| `samples_as_rows` | Rows = samples, columns = taxa, first column = sample ID. | A CSV exported from a phyloseq OTU table. |
| `taxa_as_rows` | Rows = taxa, columns = samples, first column = taxonomy lineage. | MGnify amplicon abundance TSVs. |
| `auto` (default) | Detected from the first column header (treated as `taxa_as_rows` if the header is `taxonomy`, `lineage`, `taxon`, `otu`, or `#otu id`). | |

Taxa identifiers should be **full lineage strings** (`k__Bacteria; p__Firmicutes; … ; g__Lactobacillus`) so the tokenizer can extract whichever rank the model was trained at (genus by default) and fall back to a higher rank when a lineage is shorter. If your column / row headers are bare names instead (e.g. just `Lactobacillus`), pass `--taxonomy_format genus` (or `species`, `family`, …) to prefix them with the rank tag — but be aware this disables higher-rank fallback.

### Other flags

| Flag | Default | Notes |
|---|---|---|
| `--no_normalize` | off | Skip row-normalization (use if the matrix already holds relative abundances). |
| `--keep_zeros` | off | Keep zero-abundance entries in each sample's lists. |
| `--metadata PATH` | none | CSV/TSV/parquet of per-sample metadata (indexed by sample ID); columns are merged into the output for use as labels/targets. |

A tiny MGnify-style example lives at `examples/abundance_matrix.tsv` (6 samples, 11 lineages at varying depths).

### Using the converter from Python

```python
from waypoint_bio import load_abundance_matrix, matrix_to_waypoint_df

matrix = load_abundance_matrix("examples/abundance_matrix.tsv")  # samples x taxa
df = matrix_to_waypoint_df(matrix)
df.to_parquet("my_dataset.parquet")
# df has columns: 'Taxa' (list[str]) and 'Relative Abundances' (list[float]),
# indexed by sample ID. Feed it to MicrobiomePretrainingDataset /
# MicrobiomeBenchmarkDataset directly, or save it for the CLI scripts.
```

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
├── examples/
│   ├── abundance_matrix.tsv             # MGnify-style example input for `waypoint prepare-dataset`
│   ├── sample_labels.csv                # per-sample labels to merge via `prepare-dataset --metadata`
│   ├── finetune_classification.parquet  # labelled example for `waypoint finetune` (classification)
│   └── finetune_regression.parquet      # labelled example for `waypoint finetune` (regression)
├── src/
│   └── waypoint_bio/
│       ├── cli.py                 # `waypoint` command dispatcher
│       ├── pretrain.py            # Pretraining command
│       ├── benchmark.py           # Compass benchmark command
│       ├── finetune.py            # User-provided labelled-data fine-tuning command
│       ├── embed.py               # Generate per-sample embeddings
│       ├── prepare_dataset.py     # Convert abundance matrices into waypoint format
│       ├── tokenizer.py           # TaxonomicTokenizer
│       ├── dataset.py             # Torch datasets + waypoint-format I/O helpers
│       ├── abundance_matrix.py    # Matrix conversion helpers
│       ├── models.py              # Classification/regression heads
│       ├── scoring.py             # Metric computation and task scoring
│       └── configs/               # Bundled model/training/fine-tuning configs
├── pyproject.toml
└── README.md
```


## Pretraining dataset

The pretraining corpus is **[outpost-bio/Atlas](https://huggingface.co/datasets/outpost-bio/Atlas)** on the Hugging Face Hub (**gated**; requires access and [authentication](#hugging-face-access-gated-resources)). `waypoint pretrain` loads the **`pretrain`** split with the [`datasets`](https://huggingface.co/docs/datasets) library. Rows provide microbiome samples as paired **`Taxa`** and **`Relative Abundances`** lists, which the training code turns into token sequences.

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

Downstream evaluation uses **[outpost-bio/Compass](https://huggingface.co/datasets/outpost-bio/Compass)** (**gated**; requires access and [authentication](#hugging-face-access-gated-resources)). This is a multi-configuration dataset: each **configuration** matches one source study and exposes **`train`**, **`validation`**, and **`test`** splits. `waypoint benchmark` calls `load_dataset("outpost-bio/Compass", "<config>")` per task.

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

- **Benchmark:** pass the Hub id or a local directory to `waypoint benchmark --model`:

  ```bash
  waypoint benchmark --model outpost-bio/Waypoint-6m --output_dir outputs/benchmark
  waypoint benchmark --model outputs/pretrain/best_model --output_dir outputs/benchmark
  ```

- **From Python:** load with `transformers` (the benchmark uses `AutoTokenizer` and `AutoModel` with `trust_remote_code=True` because the tokenizer is custom):

  ```python
  from transformers import AutoTokenizer, AutoModel
  tok = AutoTokenizer.from_pretrained("outpost-bio/Waypoint-6m", trust_remote_code=True)
  model = AutoModel.from_pretrained("outpost-bio/Waypoint-6m")
  ```

**Local checkpoints.** After `waypoint pretrain` finishes, use **`outputs/pretrain/best_model/`** (or your `--output_dir/best_model`): it holds the saved GPT-2 LM head, tokenizer, and `token_std_means.parquet`, and can be passed to `--model` the same way as a Hub id.

## License
apache-2.0

Maintainer / contact: neythen@outpost.bio
