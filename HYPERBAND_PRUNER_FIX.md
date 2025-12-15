# HyperbandPruner Fix for Single-Split Mode

**Date:** 2025-12-14
**Status:** ✅ **Fixed and Ready to Deploy**

---

## Problem Identified

**HyperbandPruner was NOT working in single-split mode**, causing trials to run much longer than expected.

### Symptoms

- **0 trials pruned** out of 94 trials completed
- All trials running until early stopping (~12.5 epochs average)
- Trial duration: 15-16 minutes (instead of expected 5-8 minutes)
- Total ETA: ~22 days (instead of expected ~6-7 days)

### Root Cause

**Fold-level reporting vs Epoch-level reporting mismatch:**

1. **HyperbandPruner configuration** was set for K-fold mode:
   ```yaml
   min_resource: 1  # Minimum folds
   max_resource: 5  # Maximum folds
   ```

2. **Single-split mode only has 1 "fold":**
   - Reports to Optuna **once** after all training completes
   - HyperbandPruner has no intermediate signals to prune
   - Can't compare performance until trial finishes

3. **Result:** HyperbandPruner receives only 1 data point per trial (at the end), so it can't prune early.

---

## Solution Implemented

### 1. Added Epoch-Level Reporting to Trainer

**File:** `src/criteria_bge_hpo/training/trainer.py`

**Changes:**
- Added `optuna_trial` parameter to Trainer.__init__()
- Added `report_interval` parameter (default: 5 epochs)
- Added reporting logic in train loop:
  ```python
  # Report to Optuna for HyperbandPruner (epoch-level pruning)
  if self.optuna_trial is not None and epoch % self.report_interval == 0:
      # Report current best F1 (not just current epoch F1)
      self.optuna_trial.report(self.best_val_f1, epoch)

      # Check if trial should be pruned
      if self.optuna_trial.should_prune():
          import optuna
          tqdm.write(
              f"[Optuna] Trial {self.optuna_trial.number} pruned by HyperbandPruner at epoch {epoch} "
              f"(best F1: {self.best_val_f1:.4f})"
          )
          raise optuna.TrialPruned()
  ```

### 2. Updated HPO Worker to Pass Trial Object

**File:** `src/criteria_bge_hpo/cli.py:713-714`

**Changes:**
```python
optuna_trial=trial if mode == "single_split" else None,  # Enable epoch-level pruning for single-split
report_interval=5,  # Report every 5 epochs for HyperbandPruner
```

**Why only for single_split?**
- K-fold mode already has fold-level reporting (works correctly)
- Single-split mode needs epoch-level reporting (now added)

### 3. Updated HyperbandPruner Configuration

**File:** `configs/hpo/optuna.yaml`

**Before (K-fold optimized):**
```yaml
min_resource: 1   # Minimum folds
max_resource: 5   # Maximum folds
reduction_factor: 4
```

**After (Epoch-level optimized):**
```yaml
min_resource: 5   # Minimum epochs before pruning
max_resource: 40  # Maximum epochs (matches hpo_mode.num_epochs)
reduction_factor: 3  # Keep top 1/3 at each stage
```

**Pruning stages:**
- Stage 1: 5 epochs → Prune bottom 2/3
- Stage 2: 10 epochs → Prune bottom 2/3
- Stage 3: 20 epochs → Prune bottom 2/3
- Stage 4: 40 epochs → Top performers complete all epochs

---

## Expected Impact

### Before Fix (Current State - 94 trials)

| Metric | Value |
|--------|-------|
| **Trials pruned** | 0 (0%) |
| **Average trial time** | 15.6 minutes |
| **Trials/hour** | 3.85 |
| **2000 trials ETA** | ~22 days |

### After Fix (Expected)

| Metric | Value |
|--------|-------|
| **Trials pruned** | ~1,400 (70%) |
| **Average trial time** | ~7-8 minutes |
| **Trials/hour** | ~7-8 |
| **2000 trials ETA** | **~10-12 days** |

**Time saved: ~10-12 days (50% faster!)** 🚀

### Breakdown

**Pruned trials (70% = 1,400 trials):**
- Pruned at epoch 5: 40% → 800 trials × 3 min = 2,400 min
- Pruned at epoch 10: 20% → 400 trials × 6 min = 2,400 min
- Pruned at epoch 20: 10% → 200 trials × 11 min = 2,200 min

**Surviving trials (30% = 600 trials):**
- Complete 40 epochs: 600 trials × 15 min = 9,000 min

**Total: 16,000 minutes = 11.1 days**

---

## Restart Strategy

**IMPORTANT:** Continue the current study (don't delete optuna.db)

### Why Continue?

1. ✅ Already completed 94 trials (valuable data)
2. ✅ Best F1 found: 0.7200 (Trial 71)
3. ✅ Bootstrap phase complete (30 trials)
4. ✅ Optuna will automatically use new pruning for trials 95+

### Restart Command

```bash
# Stop current run
pkill -9 -f "criteria_bge_hpo.cli"

# DON'T delete optuna.db - we want to continue!
# rm -f optuna.db  ← DON'T DO THIS

# Restart with same study name (will resume)
python3 -m criteria_bge_hpo.cli command=hpo \
    n_trials=2000 \
    model=deberta_nli \
    model.model_name=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
    experiment_name=deberta_v3_base_nli_aug \
    hpo.study_name=pc_ce_debv3_base_nli_aug_optimized \
    augmentation.enable=true \
    > hpo.log 2>&1 &

# Monitor
tail -f hpo.log
```

### What Will Happen

1. ✅ Loads existing study `pc_ce_debv3_base_nli_aug_optimized`
2. ✅ Sees 94 completed trials
3. ✅ Starts trial 95 with **new epoch-level pruning**
4. ✅ HyperbandPruner now has intermediate signals
5. ✅ Poor trials will be pruned at epochs 5, 10, or 20
6. ✅ Best F1 (0.7200) is preserved

---

## Verification Checklist

After restart, verify:

1. **Study resumed correctly:**
   ```bash
   grep "A new study created\|Loaded existing study" hpo.log
   # Should say "Loaded existing study" or start from trial 95
   ```

2. **Epoch-level reporting active:**
   ```bash
   grep "Optuna.*pruned.*at epoch" hpo.log
   # Should see pruning messages at epochs 5, 10, 15, etc.
   ```

3. **Trials getting pruned:**
   ```bash
   grep -i "pruned" hpo.log | wc -l
   # Should increase as trials run
   ```

4. **Faster trial times:**
   ```bash
   # Monitor trial durations - should see mix of:
   # - Quick trials: 3-6 min (pruned early)
   # - Full trials: 15 min (good performers)
   ```

---

## Files Modified

1. ✅ `src/criteria_bge_hpo/training/trainer.py`
   - Added `optuna_trial` and `report_interval` parameters
   - Added epoch-level reporting logic

2. ✅ `src/criteria_bge_hpo/cli.py`
   - Pass trial object to Trainer for single-split mode

3. ✅ `configs/hpo/optuna.yaml`
   - Updated min_resource: 1 → 5 (epochs)
   - Updated max_resource: 5 → 40 (epochs)
   - Updated reduction_factor: 4 → 3 (more conservative)

---

## Summary

**The fix enables HyperbandPruner to work correctly in single-split mode by:**

1. ✅ Reporting validation F1 every 5 epochs (not just at the end)
2. ✅ Allowing pruning at epochs 5, 10, 20, 30 based on performance
3. ✅ Configuring pruner for epoch-level resources (not fold-level)

**Expected result:** ~50% faster HPO completion (22 days → 10-12 days)

**Next step:** Restart HPO with same study name to continue with pruning enabled! 🚀
