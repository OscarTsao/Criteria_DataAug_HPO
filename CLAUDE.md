# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DSM-5 criteria matching project using Natural Language Inference (NLI) with DeBERTa models. The repository consolidates all active code in `src/criteria_bge_hpo/`, which contains the Hydra CLI plus data, model, training, evaluation, and utility modules built on PyTorch, Transformers, MLflow, and Optuna.

**Primary Model**: DeBERTa-v3-base (`microsoft/deberta-v3-base`) for sequence classification with text augmentation support.

## Setup and Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## Running Commands

### Quick Start with Makefile

```bash
make help                         # Show all available commands
make setup                        # Install dependencies
make train                        # Run K-fold training (100 epochs, patience 20)
make hpo_deberta_base_aug         # HPO with DeBERTa + augmentation (500 trials)
make hpo_deberta_base_noaug       # HPO with DeBERTa without augmentation
```

### DSM-5 NLI Training

The main CLI is `src/criteria_bge_hpo/cli.py` which uses Hydra for configuration management:

```bash
# K-fold cross-validation training (100 epochs, patience 20)
python -m criteria_bge_hpo.cli command=train training.num_epochs=100 training.early_stopping_patience=20

# Hyperparameter optimization (500 trials) - DeBERTa with augmentation
make hpo_deberta_base_aug N_TRIALS=500

# Or use direct CLI command
MLFLOW_TRACKING_URI=file:mlruns OPTUNA_STORAGE=sqlite:///optuna.db \
python -m criteria_bge_hpo.cli command=hpo n_trials=500 \
  model=deberta_nli model.model_name=microsoft/deberta-v3-base \
  experiment_name=deberta_v3_base_aug \
  hpo.study_name=pc_ce_debv3_base_aug \
  augmentation.enable=true \
  hpo.search_space.aug_enable.choices=[true] \
  training.num_epochs=100 training.early_stopping_patience=20

# Check HPO progress
python -c "import sqlite3; conn = sqlite3.connect('optuna.db'); \
cursor = conn.cursor(); \
cursor.execute('SELECT study_name, COUNT(*) FROM trials JOIN studies ON trials.study_id = studies.study_id WHERE state=\"COMPLETE\" GROUP BY study_name'); \
print('Completed trials:', dict(cursor.fetchall())); conn.close()"

# Evaluate specific fold
python -m criteria_bge_hpo.cli command=eval fold=0
```

### Development

```bash
# Linting and formatting
ruff check src tests
black src tests

# Run tests
pytest

# Clean outputs
make clean
```

## Configuration System

Uses Hydra with composition pattern. Main config: `configs/config.yaml`

**Config components:**
- `configs/model/deberta_nli.yaml` - DeBERTa model config (model_name, num_labels, freeze_backbone)
- `configs/training/default.yaml` - Training hyperparameters, optimization flags (bf16, tf32, compile)
- `configs/hpo/optuna.yaml` - Optuna study settings, basic search spaces
- `configs/augmentation/default.yaml` - Text augmentation settings (enable, type, prob)

**Override configs via CLI:**
```bash
python -m criteria_bge_hpo.cli command=train \
  training.learning_rate=1e-5 \
  training.num_epochs=100 \
  training.early_stopping_patience=20 \
  augmentation.enable=true \
  augmentation.prob=0.3
```

## Architecture

### DSM-5 NLI Pipeline (src/criteria_bge_hpo/)

The CLI (`cli.py`) orchestrates the full training pipeline:

1. **Data Loading** (`data/preprocessing.py`) - Loads posts, annotations, and DSM-5 criteria from CSV/JSON
2. **K-fold Splits** (`training/kfold.py`) - Stratified splits grouped by post (prevents data leakage)
3. **Dataset** (`data/dataset.py`) - Tokenizes post-criterion pairs with optional text augmentation
4. **Model** (`models/bert_classifier.py`) - DeBERTa wrapper for sequence classification
5. **Training** (`training/trainer.py`) - Training loop with gradient accumulation, mixed precision
6. **Evaluation** (`evaluation/evaluator.py`) - Per-criterion and aggregate metrics
7. **MLflow Logging** (`utils/mlflow_setup.py`) - Experiment tracking

**Key workflow pattern:** Each fold runs as a separate MLflow run, with overall summary logged after K-fold completion.

### Text Augmentation

The dataset supports evidence-based text augmentation for positive samples:
- **Synonym replacement** - WordNet-based synonym substitution
- **Contextual substitution** - BERT-based contextual word replacement
- **Configurable probability** - Control augmentation frequency (default: 0.0)
- **Evidence-only** - Only augments text spans marked as evidence

### Hyperparameter Optimization

Optuna with MedianPruner for early stopping. Each study targets 500 trials by default and stores results in `optuna.db` (SQLite).

**Current active search space** (when augmentation enabled):
- learning_rate (loguniform: 5e-6 to 3e-5)
- batch_size (categorical: [4, 8, 16])
- weight_decay (loguniform: 0.001 to 0.1)
- warmup_ratio (uniform: 0.05 to 0.15)
- aug_prob (uniform: 0.10 to 0.50) - when aug_enable=true
- aug_method (categorical: [synonym, contextual]) - when aug_enable=true

HPO runs 100-epoch K-fold CV (patience 20) to stay aligned with full-training defaults.

**Parallel execution**: HPO supports parallel trials but uses SQLite by default (limited concurrency). For better performance with parallel workers, use PostgreSQL: `hpo.storage=postgresql://user:pass@host/db`

## GPU Optimization (Ampere+ GPUs)

Training config enables aggressive optimizations:
- `use_bf16: true` - bfloat16 mixed precision (2x speedup, requires Ampere+)
- `use_tf32: true` - TensorFloat-32 operations (2-3x speedup on Ampere+)
- `use_torch_compile: false` - JIT compilation (currently disabled)
- `fused_adamw: true` - Fused optimizer kernel when CUDA is available

Set `reproducibility.tf32: true` in config for deterministic TF32 behavior.

## Data Paths

- `data/groundtruth/criteria_matching_groundtruth.csv` - Main training data (post-criterion pairs with labels)
- `data/redsm5/redsm5_annotations.csv` - Annotations with evidence spans (used for augmentation)
- `data/DSM5/MDD_Criteria.json` - DSM-5 Major Depressive Disorder criteria definitions

## Important Implementation Details

**K-fold Grouping:** Splits group by `post_id` to prevent train/val leakage when a single post has multiple criterion annotations.

**Tokenization:** Max length 512 tokens (configurable via `data.max_length`). Dataset class handles proper attention masking.

**Reproducibility:** Set seed via config, enable deterministic operations. Use `utils/reproducibility.py` helpers.

**Per-Criterion Evaluation:** Beyond aggregate F1/accuracy, track performance per individual DSM-5 criterion to identify problematic criteria.

**Augmentation Targets:** Only positive samples with evidence spans are augmented. Augmentation applies to the evidence text within the post, not the DSM-5 criterion text.

## Active HPO Studies

Monitor active studies:
```bash
# List all studies with trial counts
python -c "import sqlite3; conn = sqlite3.connect('optuna.db'); \
cursor = conn.cursor(); \
cursor.execute('SELECT study_name, direction, COUNT(trial_id) FROM studies LEFT JOIN trials ON studies.study_id = trials.study_id GROUP BY study_name'); \
for row in cursor.fetchall(): print(f'{row[0]}: {row[2]} trials ({row[1]})'); conn.close()"

# Check specific study progress
python -c "import sqlite3; conn = sqlite3.connect('optuna.db'); \
cursor = conn.cursor(); \
cursor.execute(\"SELECT state, COUNT(*) FROM trials WHERE study_id = (SELECT study_id FROM studies WHERE study_name = 'pc_ce_debv3_base_aug') GROUP BY state\"); \
print('Study: pc_ce_debv3_base_aug'); \
for state, count in cursor.fetchall(): print(f'  {state}: {count}'); conn.close()"
```

## Output Files

- `mlruns/` - MLflow experiment tracking (local SQLite backend)
- `optuna.db` - Optuna trial history (SQLite database)
- `outputs/` - Model checkpoints and training artifacts
- `*.log` - Training and HPO logs
