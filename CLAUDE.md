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

Optuna with MedianPruner for early stopping. Each study targets 500 trials by default and stores results in `optuna.db` (SQLite). The search space is defined in `configs/hpo/optuna.yaml` and includes:

- `target_effective_batch_size` (categorical: 32, 64, 128) - Effective batch size after gradient accumulation
- `scheduler_type` (categorical: linear, cosine, cosine_with_restarts) - Learning rate scheduler
- learning_rate (loguniform)
- dropout (uniform)
- weight_decay (loguniform)
- warmup_ratio (uniform)

HPO runs 100-epoch K-fold CV (patience 20) to stay aligned with full-training defaults.

**Note:** The legacy `batch_size` parameter has been replaced by `target_effective_batch_size` to support dynamic batch sizing with gradient accumulation.

## GPU Optimization (Ampere+ GPUs)

Training config enables aggressive optimizations:
- `use_bf16: true` - bfloat16 mixed precision (2x speedup, requires Ampere+)
- `use_tf32: true` - TensorFloat-32 operations (2-3x speedup on Ampere+)
- `use_torch_compile: false` - JIT compilation (10-20% speedup on PyTorch 2+)
  - **Default: disabled** for HPO stability (compilation overhead per trial)
  - **Enable for final training:** `training.optimization.use_torch_compile=true`
  - First epoch will be slower (compilation), subsequent epochs faster
- `fused_adamw: true` - Fused optimizer kernel when CUDA is available

**torch.compile Usage:**
```bash
# HPO mode - keep disabled (default)
python -m criteria_bge_hpo.cli hpo_fast --n-trials 500

# Final training - enable for 10-20% speedup
python -m criteria_bge_hpo.cli train training.optimization.use_torch_compile=true
```

Set `reproducibility.tf32: true` in config for deterministic TF32 behavior.

## Dynamic Batch Size Detection

The training system automatically detects the maximum physical batch size your GPU can handle, preventing OOM errors while maximizing GPU utilization.

**How it works:**

1. **Binary Search** - The `find_max_physical_batch_size()` function in `utils/batch_size_finder.py` performs a binary search (1-256 range) to find the largest batch size that fits in GPU memory
2. **Safety Margin** - Applies a 90% safety factor to the detected maximum to leave headroom for memory fluctuations during training
3. **Gradient Accumulation** - Automatically calculates gradient accumulation steps to reach your target effective batch size:
   ```
   gradient_accumulation_steps = target_effective_batch_size / physical_batch_size
   ```

**Physical vs Effective Batch Size:**

- **Physical Batch Size** - Number of samples processed in a single forward/backward pass (limited by GPU memory)
- **Effective Batch Size** - Total number of samples accumulated before optimizer step (controls optimization dynamics)

**Configuration:**

```yaml
training:
  target_effective_batch_size: 64  # Effective batch size for optimization
  scheduler_type: linear           # Learning rate scheduler
  batch_size: 8                    # Legacy fallback (deprecated)
```

During HPO, `target_effective_batch_size` is searched over [32, 64, 128] to find the optimal effective batch size for model convergence.

## OOM Handling

The training system includes robust out-of-memory error handling:

- **Automatic Detection** - Catches CUDA OOM errors during training loop
- **Memory Cleanup** - Clears CUDA cache and triggers garbage collection
- **HPO Integration** - In HPO mode, OOM errors prune the trial instead of crashing the entire study
- **Graceful Continuation** - Study continues with next trial, allowing exploration of different hyperparameter combinations

This ensures that aggressive hyperparameter searches (e.g., large batch sizes) don't crash the entire optimization run when they exceed GPU memory limits.

## Data Paths

- `data/redsm5/redsm5_posts.csv` - Social media posts
- `data/redsm5/redsm5_annotations.csv` - Annotations linking posts to criteria
- `data/DSM5/MDD_Criteria.json` - DSM-5 Major Depressive Disorder criteria definitions

## Important Implementation Details

**K-fold Grouping:** Splits group by `post_id` to prevent train/val leakage when a single post has multiple criterion annotations.

**Tokenization:** Max length 512 tokens (configurable via `data.max_length`). Dataset class handles proper attention masking.

**Reproducibility:** Set seed via config, enable deterministic operations. Use `utils/reproducibility.py` helpers.

**Per-Criterion Evaluation:** Beyond aggregate F1/accuracy, track performance per individual DSM-5 criterion to identify problematic criteria.
