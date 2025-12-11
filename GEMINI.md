# DSM-5 Criteria Matching (Gemini Context)

## Project Overview
This project implements a system for classifying DSM-5 symptoms using Natural Language Inference (NLI) with BGE-based reranker models. It features a complete end-to-end pipeline including data preprocessing, K-fold cross-validation training, hyperparameter optimization (HPO) with Optuna, and experiment tracking via MLflow.

**Key Technologies:**
- **Frameworks:** PyTorch, Hugging Face Transformers, Hydra (Configuration)
- **Tracking/HPO:** MLflow, Optuna
- **Code Quality:** Ruff, Black, Mypy, Pytest

## Environment Setup
The project uses a standard Python virtual environment workflow.

```bash
# 1. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies (editable mode with dev tools)
pip install -e '.[dev]'

# 3. Environment Variables
# Copy env.example to .env for MLflow/Optuna persistence configuration
cp env.example .env
```

## Core Commands (via Makefile)
The `Makefile` is the primary interface for common tasks.

- **Setup:** `make setup` (Installs dependencies)
- **Training:** `make train` (Runs 5-fold CV, 100 epochs, default configs)
- **HPO:** `make hpo` (Runs Optuna search, default 500 trials)
- **Evaluation:** `make eval` (Evaluates fold 0 of a trained model)
- **Testing:** `make test` (Runs pytest suite)
- **Linting:** `make lint` (Runs Ruff)
- **Formatting:** `make format` (Runs Black + Ruff fixes)
- **Cleaning:** `make clean` (Removes outputs, cache, and logs)

## CLI Usage (Direct)
The application entry point is `src/criteria_bge_hpo/cli.py`. It uses Hydra for configuration.

**Syntax:** `python -m criteria_bge_hpo.cli command=<cmd> [overrides]`

### Common Examples
```bash
# Train with custom parameters
python -m criteria_bge_hpo.cli command=train training.num_epochs=50 training.learning_rate=2e-5

# Hyperparameter Optimization
python -m criteria_bge_hpo.cli command=hpo n_trials=100

# Evaluate a specific fold
python -m criteria_bge_hpo.cli command=eval fold=2
```

## Configuration System
Configuration is managed by **Hydra** in the `configs/` directory.
- `configs/config.yaml`: Root configuration.
- `configs/model/`: Model architecture settings.
- `configs/training/`: Training hyperparameters.
- `configs/hpo/`: Optuna search spaces.

**Note:** Always prefer overriding configuration via CLI arguments or creating new YAML files rather than hardcoding values in Python files.

## Project Structure
- `src/criteria_bge_hpo/`: Main source code.
    - `cli.py`: Entry point.
    - `data/`: Dataset loading and preprocessing.
    - `models/`: Model definitions (BERTClassifier).
    - `training/`: Trainer loop and K-fold logic.
    - `evaluation/`: Metrics calculation.
- `configs/`: Hydra configuration files.
- `data/`: Input datasets (DSM-5 criteria, Ground truth CSVs).
- `outputs/`: Training artifacts (checkpoints, logs).
- `mlruns/`: MLflow tracking data.
- `tests/`: Pytest suite.

## Development Conventions
- **Style:** Adhere to `black` (formatting) and `ruff` (linting). Line length is 100.
- **Testing:** Write unit tests for new logic in `tests/`. Ensure `make test` passes.
- **Commits:** Use imperative, scope-prefixed commit messages (e.g., `train: add bf16 support`).
- **Paths:** Use `hydra.utils.to_absolute_path()` when dealing with file paths in the code to handle Hydra's working directory management.
