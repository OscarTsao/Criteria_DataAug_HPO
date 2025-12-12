# HPO Utility Scripts

This directory contains utility scripts for monitoring and analyzing hyperparameter optimization runs.

## Scripts

### `monitor_hpo.py`
Monitor HPO progress and estimate completion time.

```bash
python3 tools/monitor_hpo.py
```

**Features**:
- Shows trial status (completed, pruned, running, failed)
- Displays current progress percentage
- Shows best F1 score achieved so far
- Calculates average trial durations
- **Estimates time to completion and ETA**
- Warns if completion time is excessive (>60 days)

**Output Example**:
```
======================================================================
 HPO PROGRESS: pc_ce_debv3_base_aug_v2
======================================================================

Trial Status:
  Total trials: 45/2000 (2.2%)
  Completed: 30
  Pruned: 14
  Running: 1
  Failed: 0

  Best F1 Score: 0.7854

Trial Durations:
  Average (all): 3.2 hours
  Average (completed): 8.5 hours
  Average (pruned): 1.1 hours
  Min: 0.5 hours
  Max: 11.2 hours

Estimated Completion:
  Remaining time: 156 days, 8 hours
  ETA: 2026-05-18 07:23:15

⚠️  WARNING: Estimated completion is 156 days away!
   Consider reducing n_trials to 100-200 for faster completion.

======================================================================
```

**Note**: Study name is hardcoded to `pc_ce_debv3_base_aug_v2`. Edit the script to monitor different studies.

## Quick Status Check (Makefile)

For quick HPO status without scripts:

```bash
make hpo_status
```

This displays all studies with trial counts by state (COMPLETE, RUNNING, PRUNED, FAILED).
