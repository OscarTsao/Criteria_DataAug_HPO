# HPO Status - DeBERTa-v3-base with Augmentation

**Last Updated:** 2025-12-12 21:35

## Current Running Study

**Study:** `pc_ce_debv3_base_aug_v2`
**Model:** DeBERTa-v3-base (microsoft/deberta-v3-base)
**Augmentation:** ENABLED
**Trials:** 2000
**Status:** ✅ RUNNING

### Process Information
- **PID:** 2384639
- **Started:** 21:27
- **Runtime:** 06:31 elapsed
- **GPU:** NVIDIA GeForce RTX 4090
- **VRAM Usage:** 17.5 GB / 24.5 GB (71%)
- **GPU Utilization:** 99%

### Configuration
- **Max Physical Batch Size:** 19 (auto-detected with 10% VRAM headroom)
- **Gradient Accumulation:** Dynamic (based on HPO sampled batch size)
- **Pruning Strategy:**
  - **HyperbandPruner:** Successive halving (min_resource=1, max_resource=5, reduction_factor=4)
  - **Patience Pruning:** 3 consecutive folds without improvement
- **Early Stopping:** 20 epochs patience per fold
- **Training:** 100 epochs max per fold, 5-fold CV

### Current Trial
- **Trial:** 0
- **Fold:** 0
- **Epoch:** 1/100 completed
- **Learning Rate:** 5.72e-06
- **Effective Batch Size:** 64 (Physical: 19, Accumulation: 3 steps)
- **Scheduler:** cosine
- **Weight Decay:** 3.16e-02
- **Warmup Ratio:** 0.024

## Enhanced Features Implemented

### ✅ Automatic VRAM Detection
- Binary search to find max safe batch size
- 10% headroom to prevent OOM
- Per-run detection ensures optimal batch sizes

### ✅ Gradient Accumulation
- Automatically calculated when HPO samples batch size > max physical
- Formula: `accumulation_steps = ceil(sampled_batch / max_physical)`
- Maintains effective batch size while staying within memory limits

### ✅ HyperbandPruner + Patience Pruning
- **HyperbandPruner:** Aggressive early trial elimination (30-50% speedup)
- **Patience Pruning:** Stops trials with 3 consecutive non-improving folds
- Both pruning strategies work together for optimal efficiency

### ✅ Expanded Hyperparameter Search Space
- **Batch Size:** [16, 32, 64, 128, 256] (was [4, 8, 16])
- **Learning Rate:** [1e-6, 5e-5] (expanded range)
- **Warmup Ratio:** [0.0, 0.2] (expanded range)
- **Weight Decay:** [0.0, 0.1] (expanded range)
- **NEW:** classifier_dropout [0.1, 0.5]
- **NEW:** hidden_dropout [0.0, 0.3]
- **NEW:** attention_dropout [0.0, 0.3]
- **NEW:** focal_gamma [1.0, 2.0, 3.0]
- **NEW:** scheduler_type [linear, cosine, cosine_with_restarts]

## Repository Cleanup

### Removed Files
- ✅ AGENTS.md (obsolete documentation)
- ✅ AUGMENTATION_LOGGING_COMPLETE.md (obsolete documentation)
- ✅ GEMINI.md (obsolete documentation)
- ✅ analyze_hpo.py (old analysis script)
- ✅ inspect_optuna.py (old analysis script)
- ✅ mlruns_test/ (test directory)
- ✅ Criteria_Baseline_5Fold/ (submodule)
- ✅ hpo_noaug_v2.* (no-augmentation study files)

### .gitignore Updates
- Added `*.pid` to exclude process ID files
- Already excludes `*.log` and `*.db`

## Monitoring Commands

```bash
# Check HPO progress
tail -f hpo_aug_v2.log

# Check GPU usage
nvidia-smi

# Check process status
ps -p $(cat hpo_aug_v2.pid) -o pid,etime,pmem,cmd

# Query Optuna study
sqlite3 optuna.db "SELECT trial_id, state, value FROM trials WHERE study_id = (SELECT study_id FROM studies WHERE study_name='pc_ce_debv3_base_aug_v2') ORDER BY trial_id DESC LIMIT 10;"
```

## Expected Timeline

With 2000 trials and aggressive pruning:
- **Trial Duration:** ~5-30 minutes (depending on pruning)
- **Estimated Total:** ~3-7 days continuous run
- **Early trials:** Bootstrap period (first 30 trials run full 5-fold CV)
- **Later trials:** Aggressive pruning kicks in (50-70% trials pruned early)

## Next Steps

1. Monitor first 50 trials to ensure pruning is working correctly
2. Check for any patterns in pruned trials
3. Verify best trial hyperparameters after ~200 trials
4. Consider launching no-augmentation study after this completes

---

**Repository:** https://github.com/OscarTsao/Criteria_DataAug_HPO
**Last Commit:** 82ab3fd - chore: remove unused files and clean up repository
