# DSM-5 Criteria Matching with DeBERTa

End-to-end experimentation framework for classifying DSM-5 Major Depressive Disorder (MDD) symptoms via natural language inference (NLI). Built on PyTorch + Hugging Face Transformers with Hydra configuration, MLflow tracking, and Optuna hyperparameter optimization.

**Primary Model**: DeBERTa-v3-base (`microsoft/deberta-v3-base`) with optional text augmentation support.

## Features

- **K-fold Cross-Validation**: Stratified 5-fold CV with post-level grouping to prevent data leakage
- **Text Augmentation**: Evidence-based synonym and contextual word substitution for positive samples
- **Hyperparameter Optimization**: Optuna-based search with parallel trial support
- **GPU Acceleration**: bfloat16 mixed precision, TF32 operations, fused AdamW optimizer
- **Experiment Tracking**: MLflow integration with per-fold and aggregate metrics
- **Per-Criterion Metrics**: Detailed performance tracking for each DSM-5 criterion

## Quick Start

```bash
# Setup environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'

# Run training (100 epochs, patience 20)
make train

# Run HPO with augmentation (500 trials)
make hpo_deberta_base_aug N_TRIALS=500

# Check HPO progress
python -c "import sqlite3; conn = sqlite3.connect('optuna.db'); \
cursor = conn.cursor(); \
cursor.execute('SELECT study_name, COUNT(*) FROM trials JOIN studies ON trials.study_id = studies.study_id WHERE state=\"COMPLETE\" GROUP BY study_name'); \
print('Completed trials:', dict(cursor.fetchall())); conn.close()"
```

## Dataset

The project uses a unified ground-truth file with evidence-span annotations:

- **`data/groundtruth/criteria_matching_groundtruth.csv`** - Main training data
  - Columns: `post_id`, `post`, `DSM5_symptom`, `groundtruth`
  - Each row represents a `(post, criterion)` pair with binary label
  - 13,602 NLI pairs from 1,484 unique posts across 9 DSM-5 criteria

- **`data/redsm5/redsm5_annotations.csv`** - Evidence span annotations
  - Links positive samples to specific text spans within posts
  - Used for evidence-based text augmentation

- **`data/DSM5/MDD_Criteria.json`** - DSM-5 criterion definitions
  - Canonical text for each of the 9 MDD criteria

## Running Experiments

### Training

```bash
# Full K-fold training with Makefile
make train

# Or use CLI directly
python -m criteria_bge_hpo.cli command=train \
  training.num_epochs=100 \
  training.early_stopping_patience=20

# Training with augmentation
python -m criteria_bge_hpo.cli command=train \
  training.num_epochs=100 \
  training.early_stopping_patience=20 \
  augmentation.enable=true \
  augmentation.prob=0.3 \
  augmentation.type=synonym
```

### Hyperparameter Optimization

```bash
# HPO with DeBERTa + augmentation search
make hpo_deberta_base_aug N_TRIALS=500

# HPO without augmentation
make hpo_deberta_base_noaug N_TRIALS=500

# HPO with NLI-pretrained DeBERTa + augmentation
make hpo_deberta_base_nli_aug N_TRIALS=500

# Custom HPO with CLI
MLFLOW_TRACKING_URI=file:mlruns OPTUNA_STORAGE=sqlite:///optuna.db \
python -m criteria_bge_hpo.cli command=hpo n_trials=500 \
  model=deberta_nli \
  model.model_name=microsoft/deberta-v3-base \
  experiment_name=my_experiment \
  hpo.study_name=my_study \
  augmentation.enable=true \
  hpo.search_space.aug_enable.choices=[true] \
  training.num_epochs=100 \
  training.early_stopping_patience=20
```

**HPO Search Space** (when augmentation enabled):
- `learning_rate`: loguniform [5e-6, 3e-5]
- `batch_size`: categorical [4, 8, 16]
- `weight_decay`: loguniform [0.001, 0.1]
- `warmup_ratio`: uniform [0.05, 0.15]
- `aug_prob`: uniform [0.10, 0.50]
- `aug_method`: categorical [synonym, contextual]

### Evaluation

```bash
# Evaluate a specific fold
python -m criteria_bge_hpo.cli command=eval fold=0

# Requires checkpoint at outputs/<experiment>/checkpoints/fold_0_best.pt
```

## Configuration

Hydra drives all CLI commands via `configs/config.yaml`. Key config files:

- **`configs/model/deberta_nli.yaml`** - Model architecture settings
- **`configs/training/default.yaml`** - Training hyperparameters and GPU optimizations
- **`configs/hpo/optuna.yaml`** - HPO study settings and search space
- **`configs/augmentation/default.yaml`** - Text augmentation configuration

**Override configs via CLI:**
```bash
python -m criteria_bge_hpo.cli command=train \
  training.learning_rate=1e-5 \
  training.batch_size=16 \
  model.freeze_backbone=false \
  augmentation.enable=true \
  augmentation.prob=0.3
```

## Text Augmentation

Evidence-based augmentation for positive samples only:

- **Synonym Replacement**: WordNet-based synonym substitution (`augmentation.type=synonym`)
- **Contextual Substitution**: BERT-based word replacement (`augmentation.type=contextual`)
- **Evidence-Only**: Only augments text spans marked as evidence in annotations
- **Configurable Probability**: Control augmentation frequency (`augmentation.prob`)

Augmentation is applied during training to the post text containing evidence spans, never to the DSM-5 criterion text.

## GPU Optimization

Optimized for NVIDIA Ampere+ GPUs (RTX 30xx/40xx, A100):

- **bfloat16 Mixed Precision**: `training.optimization.use_bf16=true` (2x speedup)
- **TensorFloat-32**: `training.optimization.use_tf32=true` (2-3x speedup)
- **Fused AdamW**: `training.optimization.fused_adamw=true` (automatic when CUDA available)
- **Gradient Accumulation**: `training.gradient_accumulation_steps` for effective larger batches

## Development Workflow

```bash
# Install with development dependencies
make setup

# Format code
make format  # Runs black + ruff --fix

# Lint code
make lint

# Run tests
make test

# Clean outputs and cache
make clean
```

## Outputs & Tracking

- **`mlruns/`** - MLflow experiment tracking (local SQLite backend)
- **`optuna.db`** - Optuna trial history (SQLite database)
- **`outputs/<experiment>/checkpoints/`** - Saved model checkpoints per fold
- **`*.log`** - Training and HPO logs

### Monitor HPO Progress

```bash
# List all studies
python -c "import sqlite3; conn = sqlite3.connect('optuna.db'); \
cursor = conn.cursor(); \
cursor.execute('SELECT study_name, direction, COUNT(trial_id) FROM studies LEFT JOIN trials ON studies.study_id = trials.study_id GROUP BY study_name'); \
for row in cursor.fetchall(): print(f'{row[0]}: {row[2]} trials ({row[1]})'); \
conn.close()"

# Check specific study progress
python -c "import sqlite3; conn = sqlite3.connect('optuna.db'); \
cursor = conn.cursor(); \
cursor.execute(\"SELECT state, COUNT(*) FROM trials WHERE study_id = (SELECT study_id FROM studies WHERE study_name = 'pc_ce_debv3_base_aug') GROUP BY state\"); \
print('Study: pc_ce_debv3_base_aug'); \
for state, count in cursor.fetchall(): print(f'  {state}: {count}'); \
conn.close()"
```

## Project Structure

```
configs/                          # Hydra configuration files
├── config.yaml                   # Root configuration
├── model/                        # Model configs (deberta_nli, bge_reranker)
├── training/                     # Training hyperparameters
├── hpo/                          # HPO search space definitions
└── augmentation/                 # Text augmentation settings

data/                             # Dataset files
├── groundtruth/                  # Training data
│   └── criteria_matching_groundtruth.csv
├── redsm5/                       # Evidence annotations
│   ├── redsm5_posts.csv
│   └── redsm5_annotations.csv
└── DSM5/                         # DSM-5 criterion definitions
    └── MDD_Criteria.json

src/criteria_bge_hpo/             # Main package
├── cli.py                        # Hydra CLI entrypoint
├── data/                         # Data loading and preprocessing
│   ├── preprocessing.py          # Load groundtruth and annotations
│   └── dataset.py                # PyTorch dataset with augmentation
├── models/                       # Model architectures
│   └── bert_classifier.py        # DeBERTa classifier wrapper
├── training/                     # Training logic
│   ├── trainer.py                # Training loop with early stopping
│   └── kfold.py                  # K-fold split generation
├── evaluation/                   # Evaluation metrics
│   └── evaluator.py              # Per-criterion and aggregate metrics
└── utils/                        # Utilities
    ├── mlflow_setup.py           # MLflow configuration
    ├── reproducibility.py        # Seed and determinism
    └── device.py                 # CUDA setup

tests/                            # Pytest test suite
```

## Makefile Commands

```bash
make help                         # Show all available commands
make setup                        # Install dependencies
make train                        # Run K-fold training (100 epochs, patience 20)
make hpo_deberta_base_aug         # HPO with DeBERTa + augmentation (500 trials)
make hpo_deberta_base_noaug       # HPO with DeBERTa without augmentation
make hpo_deberta_base_nli_aug     # HPO with NLI-pretrained DeBERTa + augmentation
make hpo_deberta_base_nli_noaug   # HPO with NLI-pretrained DeBERTa without augmentation
make eval                         # Evaluate fold 0
make clean                        # Clean outputs, cache, and logs
make test                         # Run pytest test suite
make lint                         # Run ruff linting
make format                       # Format code with black + ruff
```

## Key Implementation Details

- **K-fold Grouping**: Splits group by `post_id` to prevent train/val leakage when a single post has multiple criterion annotations
- **Stratification**: Maintains class balance across folds despite severe class imbalance (11.3% positive rate)
- **Evidence-Based Augmentation**: Only positive samples with evidence span annotations are augmented
- **Per-Criterion Metrics**: Tracks F1, precision, recall, and accuracy for each of the 9 DSM-5 criteria
- **Early Stopping**: Monitors validation F1 score with configurable patience (default: 20 epochs)
- **MLflow Integration**: Each fold runs as a separate MLflow run with summary metrics logged after K-fold completion

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA 11.8+ (for GPU acceleration)
- NVIDIA GPU with Compute Capability 7.0+ (recommended: Ampere+ for bfloat16 support)
- 16GB+ GPU VRAM (for batch size 16 with DeBERTa-v3-base)

See `pyproject.toml` for full dependency list.

## License & Citation

This project is for research purposes. If you use this code, please cite appropriately.
