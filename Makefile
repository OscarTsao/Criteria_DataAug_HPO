# ============================================================================
# DSM-5 NLI Binary Classification - Makefile
# ============================================================================
# Automation for common development and training tasks
# Usage: make <target>
# Example: make setup && make train
# ============================================================================

# Declare all targets as phony (not actual files)
.PHONY: help setup train hpo hpo_deberta_base_noaug hpo_deberta_base_aug hpo_deberta_base_nli_noaug hpo_deberta_base_nli_aug hpo_status eval clean clean_logs test lint format

# Default tracking backends (override via env if needed)
MLFLOW_URI ?= file:mlruns
OPTUNA_URI ?= sqlite:///optuna.db
PYTHON ?= python3
N_TRIALS ?= 2000
EXTRA_ARGS ?=

# Common override bundles for HPO launchers
HPO_COMMON := training.num_epochs=100 training.early_stopping_patience=20
HPO_AUG_OFF := augmentation.enable=false hpo.search_space.aug_enable.choices=[false]
HPO_AUG_ON := augmentation.enable=true hpo.search_space.aug_enable.choices=[true]
DEBERTA_V3_BASE := model=deberta_nli model.model_name=microsoft/deberta-v3-base
DEBERTA_V3_BASE_NLI := model=deberta_nli model.model_name=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli

# Ensure this repo's src/ is first on PYTHONPATH (avoids picking up other editable installs)
PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH),)
export PYTHONPATH

# ============================================================================
# HELP - Display available targets and their descriptions
# ============================================================================
help:
	@echo "DSM-5 NLI Criteria Matching - Makefile Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  setup    - Install dependencies and setup environment"
	@echo "  train    - Run 5-fold cross-validation training"
	@echo "  hpo      - Run hyperparameter optimization (500 trials)"
	@echo "  hpo_deberta_base_noaug      - HPO with DeBERTa v3 base, augmentation off"
	@echo "  hpo_deberta_base_aug        - HPO with DeBERTa v3 base, augmentation on"
	@echo "  hpo_deberta_base_nli_noaug  - HPO with DeBERTa v3 base (NLI ckpt), augmentation off"
	@echo "  hpo_deberta_base_nli_aug    - HPO with DeBERTa v3 base (NLI ckpt), augmentation on"
	@echo "  hpo_status                  - Check HPO study progress"
	@echo "  eval     - Evaluate fold 0"
	@echo "  clean    - Clean outputs, cache, and test artifacts"
	@echo "  clean_logs                  - Clean old log files"
	@echo "  test     - Run tests"
	@echo "  lint     - Run linting checks"
	@echo "  format   - Format code with black"

# ============================================================================
# SETUP - Install dependencies in editable mode
# ============================================================================
# Upgrades pip and installs the package with development dependencies
# Run this once after cloning the repository
# Creates: .venv/lib/python3.10/site-packages/criteria_bge_hpo.egg-link
setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e '.[dev]'  # Editable install with dev dependencies (pytest, ruff, black)
	@echo "✓ Setup complete!"

# ============================================================================
# TRAIN - Run full K-fold cross-validation training
# ============================================================================
# Trains 5 separate models (one per fold) with default hyperparameters
# Logs results to MLflow (mlruns/ directory)
# Runtime: ~30-60 minutes depending on GPU
# Output: mlruns/, outputs/dsm5_criteria_matching/checkpoints/
train:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m criteria_bge_hpo.cli command=train training.num_epochs=100 training.early_stopping_patience=20 $(EXTRA_ARGS)

# ============================================================================
# HPO - Run hyperparameter optimization with Optuna
# ============================================================================
# Uses configs/hpo/pc_ce.yaml (LoRA/QLoRA + threshold tuning search space)
# Defaults to 500 trials; override N_TRIALS for smaller/faster searches
# Results stored in SQLite (Optuna) and MLflow by default
# Pass EXTRA_ARGS for Hydra overrides (e.g., hpo.search_space.threshold_mode=global)
hpo:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m criteria_bge_hpo.cli command=hpo n_trials=$(N_TRIALS) $(HPO_COMMON) $(EXTRA_ARGS)

hpo_deberta_base_noaug:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m criteria_bge_hpo.cli command=hpo n_trials=$(N_TRIALS) $(DEBERTA_V3_BASE) experiment_name=deberta_v3_base_noaug hpo.study_name=pc_ce_debv3_base_noaug $(HPO_AUG_OFF) $(HPO_COMMON) $(EXTRA_ARGS)

hpo_deberta_base_aug:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m criteria_bge_hpo.cli command=hpo n_trials=$(N_TRIALS) $(DEBERTA_V3_BASE) experiment_name=deberta_v3_base_aug hpo.study_name=pc_ce_debv3_base_aug $(HPO_AUG_ON) $(HPO_COMMON) $(EXTRA_ARGS)

hpo_deberta_base_nli_noaug:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m criteria_bge_hpo.cli command=hpo n_trials=$(N_TRIALS) $(DEBERTA_V3_BASE_NLI) experiment_name=deberta_v3_base_nli_noaug hpo.study_name=pc_ce_debv3_base_nli_noaug $(HPO_AUG_OFF) $(HPO_COMMON) $(EXTRA_ARGS)

hpo_deberta_base_nli_aug:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) OPTUNA_STORAGE=$(OPTUNA_URI) $(PYTHON) -m criteria_bge_hpo.cli command=hpo n_trials=$(N_TRIALS) $(DEBERTA_V3_BASE_NLI) experiment_name=deberta_v3_base_nli_aug hpo.study_name=pc_ce_debv3_base_nli_aug $(HPO_AUG_ON) $(HPO_COMMON) $(EXTRA_ARGS)

# ============================================================================
# HPO STATUS - Check hyperparameter optimization progress
# ============================================================================
# Displays trial counts and completion status for all HPO studies
# Shows COMPLETE, RUNNING, PRUNED, and FAILED trial counts per study
hpo_status:
	@echo "═══════════════════════════════════════════════════════"
	@echo "              HPO Study Status"
	@echo "═══════════════════════════════════════════════════════"
	@$(PYTHON) tools/hpo_status.py
	@echo ""

# ============================================================================
# EVAL - Evaluate a specific fold
# ============================================================================
# Loads trained model from fold 0 and runs evaluation
# Displays per-criterion metrics and aggregate performance
eval:
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) $(PYTHON) -m criteria_bge_hpo.cli command=eval fold=0 $(EXTRA_ARGS)

# ============================================================================
# CLEAN - Remove all generated files, outputs, and cache
# ============================================================================
# Deletes:
#   - outputs/ - Model checkpoints and training artifacts
#   - .pytest_cache/ - pytest cache
#   - .coverage - Coverage report data
#   - __pycache__/ - Python bytecode cache (all directories)
#   - *.pyc - Compiled Python files
# WARNING: This does NOT delete mlruns/ or optuna.db (use with caution)
# Note: Use 'make clean_logs' to remove old log files
clean:
	rm -rf outputs/
	rm -rf .pytest_cache/
	rm -rf .coverage htmlcov/
	rm -rf __pycache__/
	# Find and remove all __pycache__ directories recursively (ignore errors)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	# Find and remove all .pyc files recursively
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned outputs, cache, and test artifacts"

# ============================================================================
# CLEAN LOGS - Remove old log files
# ============================================================================
# Deletes old *.log files in the root directory
# Preserves mlruns/ and optuna.db
# Use this to clean up accumulated log files from training/HPO runs
clean_logs:
	@echo "Removing old log files..."
	rm -f *.log
	rm -f nohup.out
	@echo "✓ Cleaned log files"

# ============================================================================
# TEST - Run pytest test suite with coverage reporting
# ============================================================================
# Runs all tests in tests/ directory
# Generates HTML coverage report in htmlcov/
# Flags:
#   -v: Verbose output (show individual test results)
#   --cov: Measure code coverage for src/criteria_bge_hpo
#   --cov-report=html: Generate HTML coverage report
# Output: htmlcov/index.html (open in browser to view coverage)
test:
	pytest tests/ -v --cov=src/criteria_bge_hpo --cov-report=html

# ============================================================================
# LINT - Run code quality checks with ruff
# ============================================================================
# Checks for:
#   - PEP 8 style violations
#   - Common bugs and code smells
#   - Import sorting issues
#   - Unused imports and variables
# Does NOT modify files (use 'make format' to auto-fix)
# Exit code: 0 if clean, 1 if issues found
lint:
	ruff check src tests
	@echo "✓ Linting complete"

# ============================================================================
# FORMAT - Auto-format code with black and fix linting issues
# ============================================================================
# Steps:
#   1. black: Formats all Python files to consistent style (line length 100)
#   2. ruff --fix: Auto-fixes safe linting issues (imports, unused vars, etc.)
# Modifies files in-place
# Always run before committing code
format:
	black src tests                # Format with black (line length from pyproject.toml)
	ruff check --fix src tests     # Auto-fix safe linting issues
	@echo "✓ Code formatted"
