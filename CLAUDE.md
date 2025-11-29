# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a DSM-5 criteria matching project using Natural Language Inference (NLI) with BERT-based models. The repository now consolidates all active code in `src/criteria_bge_hpo/`, which contains the Hydra CLI plus data, model, training, evaluation, and utility modules built on PyTorch, Transformers, MLflow, and Optuna.

## Setup and Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

## Running Commands

### DSM-5 NLI Training

The main CLI is `src/criteria_bge_hpo/cli.py` which uses Hydra for configuration management:

```bash
# K-fold cross-validation training (100 epochs, patience 20)
python -m criteria_bge_hpo.cli train training.num_epochs=100 training.early_stopping_patience=20

# Hyperparameter optimization (500 trials)
python -m criteria_bge_hpo.cli hpo --n-trials 500

# Evaluate specific fold
python -m criteria_bge_hpo.cli eval --fold 0
```

### Development

```bash
# Linting and formatting
ruff check src tests
black src tests

# Run tests
pytest
```

## Configuration System

Uses Hydra with composition pattern. Main config: `configs/config.yaml`

**Config components:**
- `configs/model/bge_reranker.yaml` - Reranker architecture (model_name, num_labels, threshold)
- `configs/training/default.yaml` - Training hyperparameters, optimization flags
- `configs/hpo/optuna.yaml` - Optuna study settings, search spaces

**Override configs via CLI:**
```bash
python -m criteria_bge_hpo.cli train training.learning_rate=1e-5 training.num_epochs=100 training.early_stopping_patience=20
```

## Architecture

### DSM-5 NLI Pipeline (src/criteria_bge_hpo/)

The CLI (`cli.py`) orchestrates the full training pipeline:

1. **Data Loading** (`data/preprocessing.py`) - Loads posts, annotations, and DSM-5 criteria from CSV/JSON
2. **K-fold Splits** (`training/kfold.py`) - Stratified splits grouped by post (prevents data leakage)
3. **Dataset** - Tokenizes post-criterion pairs for binary classification
4. **Model** (`models/bert_classifier.py`) - BGE reranker wrapper for sequence classification
5. **Training** (`training/trainer.py`) - Training loop with gradient accumulation, mixed precision
6. **Evaluation** (`evaluation/evaluator.py`) - Per-criterion and aggregate metrics
7. **MLflow Logging** (`utils/mlflow_setup.py`) - Experiment tracking

**Key workflow pattern:** Each fold runs as a separate MLflow run, with overall summary logged after K-fold completion.

### Hyperparameter Optimization

Optuna with MedianPruner for early stopping. Each study targets 500 trials by default and stores results in `optuna.db` (SQLite). Search space defined in `configs/hpo/optuna.yaml`:
- learning_rate (loguniform)
- batch_size (categorical)
- dropout (uniform)
- weight_decay (loguniform)
- warmup_ratio (uniform)

HPO runs 100-epoch K-fold CV (patience 20) to stay aligned with full-training defaults.

## GPU Optimization (Ampere+ GPUs)

Training config enables aggressive optimizations:
- `use_bf16: true` - bfloat16 mixed precision (2x speedup, requires Ampere+)
- `use_tf32: true` - TensorFloat-32 operations (2-3x speedup on Ampere+)
- `use_torch_compile: true` - JIT compilation (10-20% speedup on PyTorch 2+)
- `fused_adamw: true` - Fused optimizer kernel when CUDA is available

Set `reproducibility.tf32: true` in config for deterministic TF32 behavior.

## Data Paths

- `data/redsm5/redsm5_posts.csv` - Social media posts
- `data/redsm5/redsm5_annotations.csv` - Annotations linking posts to criteria
- `data/DSM5/MDD_Criteria.json` - DSM-5 Major Depressive Disorder criteria definitions

## Important Implementation Details

**K-fold Grouping:** Splits group by `post_id` to prevent train/val leakage when a single post has multiple criterion annotations.

**Tokenization:** Max length 512 tokens (configurable via `data.max_length`). Dataset class handles proper attention masking.

**Reproducibility:** Set seed via config, enable deterministic operations. Use `utils/reproducibility.py` helpers.

**Per-Criterion Evaluation:** Beyond aggregate F1/accuracy, track performance per individual DSM-5 criterion to identify problematic criteria.
