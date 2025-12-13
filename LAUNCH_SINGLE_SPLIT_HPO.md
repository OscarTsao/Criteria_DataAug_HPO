# Launch Single-Split HPO - Command Reference

**Date:** 2025-12-13
**Mode:** Single-Split (10-15x faster than K-fold)
**Expected Completion:** ~6 days for 2000 trials

---

## Pre-Launch Checklist

### 1. Verify Configuration

```bash
# Check HPO mode is set to single_split
python3 -m criteria_bge_hpo.cli command=hpo --cfg job | grep -A 5 "hpo_mode"

# Expected output:
#   mode: single_split
#   train_split: 0.8
#   num_epochs: 40
#   early_stopping_patience: 10
```

### 2. Check GPU Availability

```bash
nvidia-smi

# Verify:
# ✓ GPU detected (RTX 4090 or similar)
# ✓ VRAM available (>20 GB free recommended)
# ✓ No other processes using GPU
```

### 3. Verify System Status

```bash
# Check no other HPO running
ps aux | grep criteria_bge_hpo

# Check disk space for database
df -h .

# Expected: >10 GB free for optuna.db growth
```

---

## Launch Commands

### Option 1: DeBERTa-v3-base with Augmentation Search (RECOMMENDED)

```bash
# Launch 2000-trial HPO with single-split mode
nohup python3 -m criteria_bge_hpo.cli command=hpo \
    model=deberta_nli \
    hpo=pc_ce \
    hpo.study_name=deberta_v3_base_single_split_hpo \
    augmentation.enable=true \
    n_trials=2000 \
    > hpo_single_split_deberta_aug.log 2>&1 &

# Save PID
echo $! > hpo_single_split.pid

# Monitor launch
tail -f hpo_single_split_deberta_aug.log
```

**Expected Timeline:**
- Bootstrap phase (30 trials): ~3-4 hours
- Pruned trials (1970 trials): ~5.5 days
- **Total: ~6 days**

**Expected Completion:** Dec 19, 2025

### Option 2: DeBERTa-v3-base without Augmentation (Baseline)

```bash
# Launch without augmentation search
nohup python3 -m criteria_bge_hpo.cli command=hpo \
    model=deberta_nli \
    hpo=pc_ce \
    hpo.study_name=deberta_v3_base_single_split_no_aug \
    augmentation.enable=false \
    n_trials=2000 \
    > hpo_single_split_deberta_no_aug.log 2>&1 &

# Save PID
echo $! > hpo_single_split_no_aug.pid
```

### Option 3: Extended HPO (4000 trials)

For even more comprehensive search:

```bash
nohup python3 -m criteria_bge_hpo.cli command=hpo \
    model=deberta_nli \
    hpo=pc_ce \
    hpo.study_name=deberta_v3_base_extended_hpo \
    augmentation.enable=true \
    n_trials=4000 \
    > hpo_extended.log 2>&1 &

# Expected time: ~12 days
```

---

## Monitoring

### Real-Time Log Monitoring

```bash
# Follow live log
tail -f hpo_single_split_deberta_aug.log

# Search for specific events
grep "Trial.*pruned" hpo_single_split_deberta_aug.log | tail -20
grep "Val F1" hpo_single_split_deberta_aug.log | tail -20
```

### GPU Monitoring

```bash
# Real-time GPU usage
watch -n 1 nvidia-smi

# Expected:
# ✓ GPU Utilization: 95-99%
# ✓ VRAM Usage: 17-20 GB
# ✓ Temperature: <80°C
```

### Study Progress

```bash
# Check trial counts
sqlite3 optuna.db "
SELECT
    COUNT(CASE WHEN state='COMPLETE' THEN 1 END) as completed,
    COUNT(CASE WHEN state='PRUNED' THEN 1 END) as pruned,
    COUNT(CASE WHEN state='RUNNING' THEN 1 END) as running,
    COUNT(CASE WHEN state='FAIL' THEN 1 END) as failed
FROM trials
WHERE study_id = (SELECT study_id FROM studies WHERE study_name='deberta_v3_base_single_split_hpo');
"

# Check best F1 so far
sqlite3 optuna.db "
SELECT MAX(value) as best_f1
FROM trial_values
JOIN trials USING(trial_id)
JOIN studies USING(study_id)
WHERE study_name='deberta_v3_base_single_split_hpo'
  AND state='COMPLETE';
"
```

### Process Status

```bash
# Check if process is running
ps -p $(cat hpo_single_split.pid) -o pid,etime,cmd

# CPU/Memory usage
top -p $(cat hpo_single_split.pid)
```

---

## Post-Launch Verification

### First Hour Checkpoints

**After 10 minutes:**
- ✅ Process still running
- ✅ GPU at 95%+ utilization
- ✅ Log shows "HPO Mode: SINGLE_SPLIT"
- ✅ Log shows "Train/Val Split: 80%/20%"
- ✅ First trial started

**After 30 minutes:**
- ✅ First trial completed or pruned
- ✅ Trial time: 3-10 minutes
- ✅ No OOM errors
- ✅ Optuna database created/updated

**After 1 hour:**
- ✅ 5-10 trials completed
- ✅ Some trials pruned (30-50%)
- ✅ Best F1 score >0.60
- ✅ Consistent GPU utilization

### Daily Checkpoints

**Day 1 (24 hours):**
- Completed trials: ~200-300
- Best F1: Should be >0.70
- Pruning rate: 50-70%
- GPU uptime: Consistent

**Day 3 (72 hours):**
- Completed trials: ~600-900
- Best F1: Should be >0.75
- Study health: No crashes

**Day 6 (Completion):**
- Total trials: ~2000
- Completed: ~400-600
- Pruned: ~1400-1600
- Best F1: Target >0.80

---

## Troubleshooting

### Process Died Early

**Check logs:**
```bash
tail -100 hpo_single_split_deberta_aug.log
```

**Common issues:**
- OOM errors: Check VRAM usage, may need to adjust batch size detection
- Disk full: Check disk space for optuna.db
- CUDA errors: Check CUDA/driver compatibility

**Recovery:**
```bash
# Resume from where it stopped (Optuna will skip completed trials)
python3 -m criteria_bge_hpo.cli command=hpo \
    model=deberta_nli \
    hpo=pc_ce \
    hpo.study_name=deberta_v3_base_single_split_hpo \
    n_trials=2000
```

### GPU Utilization Low (<80%)

**Possible causes:**
- DataLoader bottleneck: Check num_workers
- Disk I/O bottleneck: Check disk speed
- CPU bottleneck: Check CPU usage

**Solutions:**
```bash
# Increase DataLoader workers (if CPU has capacity)
# Edit configs/training/default.yaml:
num_workers: auto  # or set to cpu_count - 2
```

### Trials Taking Too Long

**Expected times:**
- Bootstrap trials: 6-8 minutes
- Pruned trials: 3-5 minutes
- Average: ~4.5 minutes

**If trials >10 minutes:**
- Check torch.compile overhead (first epoch is slower)
- Check if early stopping is working
- Verify patience is set to 10 (not 20)

### Best F1 Not Improving

**After 100 trials, if best F1 <0.65:**
- May indicate search space issues
- Check if any trials are completing (not all pruned)
- Verify augmentation parameters if enabled

**After 500 trials, if best F1 <0.75:**
- May need to adjust search space
- Check logs for consistent failures
- Consider narrowing search ranges

---

## Stop/Restart

### Graceful Stop

```bash
# Send SIGTERM to allow cleanup
kill $(cat hpo_single_split.pid)

# Wait for current trial to complete (may take a few minutes)
# Check log for "Worker Complete!"
```

### Force Stop

```bash
# Use only if graceful stop fails
kill -9 $(cat hpo_single_split.pid)
```

### Resume After Stop

```bash
# Optuna automatically resumes from database
python3 -m criteria_bge_hpo.cli command=hpo \
    model=deberta_nli \
    hpo=pc_ce \
    hpo.study_name=deberta_v3_base_single_split_hpo \
    n_trials=2000

# Will skip completed trials and continue
```

---

## After Completion

### 1. Extract Best Hyperparameters

```bash
python3 << 'EOF'
import optuna

study = optuna.load_study(
    study_name='deberta_v3_base_single_split_hpo',
    storage='sqlite:///optuna.db'
)

print(f"Study completed with {len(study.trials)} trials")
print(f"Best F1: {study.best_value:.4f}")
print("\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Show top-5
print("\n=== Top-5 Configurations ===")
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
top_5 = sorted(completed, key=lambda t: t.value, reverse=True)[:5]
for i, trial in enumerate(top_5, 1):
    print(f"\n{i}. Trial {trial.number}: F1 = {trial.value:.4f}")
    for key, value in trial.params.items():
        print(f"   {key}: {value}")
EOF
```

### 2. Run Final K-Fold Validation

```bash
# Use best hyperparameters from above
python3 -m criteria_bge_hpo.cli command=train \
    model=deberta_nli \
    training.learning_rate=<best_lr> \
    training.target_effective_batch_size=<best_batch> \
    training.scheduler_type=<best_scheduler> \
    training.weight_decay=<best_wd> \
    training.warmup_ratio=<best_warmup> \
    training.num_epochs=100 \
    training.early_stopping_patience=20

# Expected time: ~12 hours
# Result: 5-fold CV metrics for production model
```

### 3. Archive Results

```bash
# Create results directory
mkdir -p results/single_split_hpo_$(date +%Y%m%d)

# Copy important files
cp optuna.db results/single_split_hpo_$(date +%Y%m%d)/
cp hpo_single_split_deberta_aug.log results/single_split_hpo_$(date +%Y%m%d)/
cp hpo_single_split.pid results/single_split_hpo_$(date +%Y%m%d)/

# Export best params
python3 << 'EOF' > results/single_split_hpo_$(date +%Y%m%d)/best_params.txt
import optuna
study = optuna.load_study(
    study_name='deberta_v3_base_single_split_hpo',
    storage='sqlite:///optuna.db'
)
print(f"Best F1: {study.best_value:.4f}")
print("\nBest Parameters:")
for key, value in study.best_params.items():
    print(f"{key}: {value}")
EOF
```

---

## Expected Outcomes

### Performance Targets

**Minimum Acceptable:**
- Best F1: >0.75
- Pruning rate: 40-80%
- OOM rate: <5%

**Good Performance:**
- Best F1: >0.80
- Pruning rate: 60-70%
- OOM rate: <2%

**Excellent Performance:**
- Best F1: >0.85
- Pruning rate: 65-75%
- OOM rate: <1%

### Timeline Summary

| Milestone | Expected Date | Trials Completed |
|-----------|--------------|------------------|
| Launch | 2025-12-13 | 0 |
| Bootstrap Complete | 2025-12-13 (4h) | 30 |
| 25% Complete | 2025-12-15 | 500 |
| 50% Complete | 2025-12-16 | 1000 |
| 75% Complete | 2025-12-18 | 1500 |
| **HPO Complete** | **2025-12-19** | **2000** |
| K-Fold Validation | 2025-12-20 | - |
| **Final Model** | **2025-12-20** | - |

---

## Ready to Launch!

**Recommended Command:**
```bash
nohup python3 -m criteria_bge_hpo.cli command=hpo \
    model=deberta_nli \
    hpo=pc_ce \
    hpo.study_name=deberta_v3_base_single_split_hpo \
    augmentation.enable=true \
    n_trials=2000 \
    > hpo_single_split_deberta_aug.log 2>&1 &

echo $! > hpo_single_split.pid

echo "HPO launched! PID: $(cat hpo_single_split.pid)"
echo "Monitor with: tail -f hpo_single_split_deberta_aug.log"
echo "Expected completion: 2025-12-19 (~6 days)"
```

**Good luck! 🚀**
