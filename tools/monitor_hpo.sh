#!/bin/bash
# HPO Monitoring Script for DeBERTa base with augmentation
# Usage: ./monitor_hpo.sh

echo "════════════════════════════════════════════════════════════════"
echo "  DeBERTa-v3-base WITH Augmentation HPO - Status Monitor"
echo "════════════════════════════════════════════════════════════════"
echo ""

# GPU Status
echo "📊 GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
awk -F', ' '{printf "   Utilization: %s%% GPU, %s%% Memory\n   Memory: %s MB / %s MB (%.1f%%)\n   Temperature: %s°C\n", $1, $2, $3, $4, ($3/$4)*100, $5}'
echo ""

# Process Status
echo "🔄 Process Status:"
process_count=$(ps aux | grep "criteria_bge_hpo.cli" | grep -v grep | wc -l)
echo "   Active processes: $process_count"
if [ $process_count -eq 0 ]; then
    echo "   ⚠️  WARNING: No HPO processes running!"
fi
echo ""

# Optuna Study Progress
echo "📈 Study Progress:"
python3 << 'EOF'
import optuna
import datetime

storage = 'sqlite:///optuna.db'
try:
    study = optuna.load_study(study_name='pc_ce_debv3_base_aug', storage=storage)

    total = len(study.trials)
    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    running = len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
    failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    print(f"   Total trials: {total}")
    print(f"   ✅ Completed: {completed}")
    print(f"   🏃 Running: {running}")
    print(f"   ❌ Failed: {failed}")
    print(f"   Progress: {completed}/500 ({completed/500*100:.1f}%)")

    if completed > 0:
        print(f"\n🏆 Best Result:")
        best = study.best_trial
        print(f"   Trial #{best.number}: F1 = {best.value:.4f}")
        print(f"   LR: {best.params['learning_rate']:.2e}, BS: {best.params['batch_size']}")
        print(f"   Aug: {best.params.get('aug_method', 'N/A')} (prob={best.params.get('aug_prob', 0):.3f})")

        # Duration estimate
        durations = [t.duration.total_seconds() / 3600 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        avg_duration = sum(durations) / len(durations)
        remaining = 500 - total
        est_hours = remaining * avg_duration
        est_days = est_hours / 24

        print(f"\n⏱️  Timing:")
        print(f"   Avg trial duration: {avg_duration:.2f} hours")
        print(f"   Remaining trials: {remaining}")
        print(f"   Est. completion: {est_days:.1f} days")

    # Running trials detail
    if running > 0:
        print(f"\n🏃 Currently Running Trials:")
        running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
        for trial in running_trials:
            elapsed = datetime.datetime.now() - trial.datetime_start
            hours = int(elapsed.total_seconds() // 3600)
            mins = int((elapsed.total_seconds() % 3600) // 60)
            print(f"   Trial #{trial.number}: {hours}h {mins}m elapsed")
            print(f"      Aug: {trial.params.get('aug_method', 'N/A')} (prob={trial.params.get('aug_prob', 'N/A')})")

except Exception as e:
    print(f"   ❌ Error: {e}")
EOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Last updated: $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════════"
