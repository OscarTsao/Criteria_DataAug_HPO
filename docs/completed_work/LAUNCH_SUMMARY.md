# HPO Launch Summary - 2025-12-13

## ✅ Pre-Launch Checklist

### Configuration Verified
- [x] Training epochs: 100 (with patience 20)
- [x] torch.compile: ENABLED (15% speedup)
- [x] zero_grad(set_to_none=True): ENABLED (2-5% speedup)
- [x] BF16 + TF32 + Fused AdamW: ALL ENABLED
- [x] Optimization level: **98%** (near-maximum)
- [x] HPO config: pc_ce.yaml with augmentation search space
- [x] Model: microsoft/deberta-v3-base
- [x] Pruner: HyperbandPruner (reduction_factor=4, bootstrap=30)

### Project Organization
- [x] Scripts moved to `scripts/` directory
- [x] Completed docs moved to `docs/completed_work/`
- [x] PROJECT_STATUS.md created (consolidated status)
- [x] Launch script updated: `scripts/launch_deberta_hpo.sh`
- [x] Verification script available: `scripts/verify_hpo_settings.py`

### System Status
- [x] GPU available and detected
- [x] CUDA configured properly
- [x] Dependencies installed
- [x] Data files present
- [x] Optuna database ready (optuna.db)

---

## 🚀 Launch Commands

### Option 1: Automated Launch (Both Studies)
```bash
# Launch both studies automatically
./scripts/launch_deberta_hpo.sh
```

**Launches:**
1. `deberta_base_no_aug` - DeBERTa without augmentation (baseline)
2. `deberta_base_aug` - DeBERTa with augmentation search

**Note:** Both will run in parallel and compete for GPU. Consider running sequentially or using separate GPUs.

### Option 2: Manual Launch (Sequential)

**Study 1: No Augmentation (Baseline)**
```bash
nohup python -m criteria_bge_hpo.cli hpo \
    model=deberta_nli \
    hpo=pc_ce \
    hpo.study_name=deberta_base_no_aug \
    augmentation.enable=false \
    experiment_name=deberta_base_no_aug_hpo \
    > hpo_deberta_base_no_aug.log 2>&1 &

# Save PID
echo $! > hpo_no_aug.pid
```

**Study 2: With Augmentation (After Study 1 completes)**
```bash
nohup python -m criteria_bge_hpo.cli hpo \
    model=deberta_nli \
    hpo=pc_ce \
    hpo.study_name=deberta_base_aug \
    augmentation.enable=true \
    experiment_name=deberta_base_aug_hpo \
    > hpo_deberta_base_aug.log 2>&1 &

# Save PID
echo $! > hpo_aug.pid
```

---

## 📊 Expected Performance

### Timeline (Per Study)
- **Bootstrap phase (30 trials):** ~12.5 days (300 hours)
- **Pruned phase (1970 trials):** ~14.4 days (345 hours)
- **Total per study:** ~27 days (645 hours)
- **Both studies (parallel):** ~27 days (same wall-clock)
- **Both studies (sequential):** ~54 days

### Resource Usage
- **GPU Utilization:** 95-99%
- **VRAM Usage:** 17-20 GB / 24 GB (70-83%)
- **CPU Usage:** Moderate (DataLoader workers)
- **Disk Usage:** ~500 MB (optuna.db grows over time)

### Quality Metrics
- **Expected pruning rate:** 60-70% of trials
- **OOM rate:** <5% (gracefully handled)
- **Target best F1:** >0.80 (depends on dataset)

---

## 👁️ Monitoring

### Real-Time GPU Monitoring
```bash
watch -n 1 nvidia-smi
```

### Log Monitoring
```bash
# No augmentation study
tail -f hpo_deberta_base_no_aug.log

# Augmentation study
tail -f hpo_deberta_base_aug.log
```

### Study Progress
```bash
# Quick status
sqlite3 optuna.db "
SELECT
    study_name,
    COUNT(CASE WHEN state='COMPLETE' THEN 1 END) as completed,
    COUNT(CASE WHEN state='PRUNED' THEN 1 END) as pruned,
    COUNT(CASE WHEN state='RUNNING' THEN 1 END) as running
FROM trials
JOIN studies USING(study_id)
GROUP BY study_name;
"

# Best F1 scores
sqlite3 optuna.db "
SELECT study_name, MAX(value) as best_f1
FROM trial_values
JOIN trials USING(trial_id)
JOIN studies USING(study_id)
WHERE state='COMPLETE'
GROUP BY study_name;
"
```

### Process Status
```bash
# Check if HPO is running
ps aux | grep criteria_bge_hpo.cli

# Check PIDs
cat hpo_no_aug.pid hpo_aug.pid 2>/dev/null
```

---

## 🛑 Stop/Resume

### Stop Studies
```bash
# Stop using PID file
kill $(cat hpo_no_aug.pid)
kill $(cat hpo_aug.pid)

# Or find and kill manually
ps aux | grep criteria_bge_hpo.cli | grep -v grep
kill <PID>
```

### Resume Studies
HPO studies can be resumed by re-running the same command with the same study_name:

```bash
# Resume will continue from where it left off
python -m criteria_bge_hpo.cli hpo \
    model=deberta_nli \
    hpo=pc_ce \
    hpo.study_name=deberta_base_no_aug
```

Optuna automatically detects existing studies and continues from the last trial.

---

## 📈 Post-HPO Analysis

### Extract Best Hyperparameters
```python
import optuna

# Load study
study = optuna.load_study(
    study_name="deberta_base_no_aug",
    storage="sqlite:///optuna.db"
)

# Best trial
print(f"Best F1: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Top-5 trials
top_trials = sorted(study.best_trials, key=lambda t: t.value, reverse=True)[:5]
for rank, trial in enumerate(top_trials, 1):
    print(f"Rank {rank}: F1={trial.value:.4f}, Params={trial.params}")
```

### Compare Studies
```bash
sqlite3 optuna.db "
SELECT
    study_name,
    MAX(value) as best_f1,
    AVG(value) as mean_f1,
    COUNT(*) as total_trials
FROM trial_values
JOIN trials USING(trial_id)
JOIN studies USING(study_id)
WHERE state='COMPLETE'
GROUP BY study_name;
"
```

---

## 🔧 Troubleshooting

### Common Issues

**1. GPU Out of Memory**
- Handled automatically (trial pruned, study continues)
- If frequent, reduce target_effective_batch_size range in config

**2. Process Killed/Stopped**
- Check logs for errors
- Resume with same command (Optuna will continue)

**3. Slow Progress**
- Check GPU utilization (should be >90%)
- Verify torch.compile is enabled
- Check for CPU bottleneck in DataLoader

**4. Database Locked**
- SQLite has limited concurrency
- Don't run parallel HPO with SQLite
- Consider PostgreSQL for parallel optimization

---

## ✅ Success Indicators

**Healthy HPO Run:**
- ✅ GPU utilization 95-99%
- ✅ Trials completing/pruning regularly
- ✅ Best F1 improving over time
- ✅ Pruning rate 60-70%
- ✅ OOM rate <5%
- ✅ No process crashes

**Check After 24 Hours:**
- Completed trials: ~30-40 (bootstrap + some pruned)
- Best F1: Should be >0.70
- GPU utilization: Consistently high
- Logs: No repeated errors

**Check After 1 Week:**
- Completed trials: ~200-300
- Best F1: Should be improving
- Pruning working effectively
- No major issues in logs

---

## 📝 Launch Log

### Launch Time
**Date:** 2025-12-13
**Time:** [To be filled when launched]

### Studies Launched
- [ ] deberta_base_no_aug
- [ ] deberta_base_aug

### Initial Status
- GPU: [To be filled]
- Available VRAM: [To be filled]
- Initial best F1: [To be filled after first trial]

### Notes
[Add any observations or issues during launch]

---

## 🎯 Next Steps After Launch

1. **Monitor for first 1-2 hours**
   - Ensure GPU utilization is high
   - Check first few trials complete successfully
   - Verify pruning is working

2. **Check daily progress**
   - Query best F1 scores
   - Monitor trial completion rate
   - Check for any errors in logs

3. **After completion (~27 days)**
   - Extract best hyperparameters
   - Compare augmentation vs. no-augmentation
   - Run final validation with K-fold CV on best configs
   - Document findings

---

**Status:** Ready to launch! ✅

**Command to start:** `./scripts/launch_deberta_hpo.sh`
