# HPO Utility Scripts

This directory contains utility scripts for monitoring and analyzing hyperparameter optimization runs.

## Scripts

### `inspect_optuna.py`
Quick inspection of all Optuna studies in `optuna.db`.

```bash
python tools/inspect_optuna.py
```

**Output**: Lists all studies with trial counts, best values, and optimization direction.

### `analyze_hpo.py`
Detailed analysis of a specific Optuna study.

```bash
python tools/analyze_hpo.py
```

**Features**:
- Lists all trials sorted by performance
- Shows hyperparameters for each trial
- Identifies failed trials with error details
- Displays best trial parameters

**Note**: Update `study_name` variable in the script to match your study.

### `monitor_hpo.sh`
Real-time monitoring script for active HPO runs.

```bash
./tools/monitor_hpo.sh
```

**Features**:
- Watches GPU utilization
- Tracks trial progress
- Monitors log file output

## Quick Status Check (Makefile)

For quick HPO status without scripts:

```bash
make hpo_status
```

This displays all studies with trial counts by state (COMPLETE, RUNNING, PRUNED, FAILED).
