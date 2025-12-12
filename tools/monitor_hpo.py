#!/usr/bin/env python3
"""Monitor HPO progress and estimate completion time."""

import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path


def get_study_stats(db_path: str, study_name: str):
    """Query Optuna database for study statistics."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get study info
    cursor.execute("""
        SELECT COUNT(*) as total_trials,
               SUM(CASE WHEN state = 'COMPLETE' THEN 1 ELSE 0 END) as completed,
               SUM(CASE WHEN state = 'PRUNED' THEN 1 ELSE 0 END) as pruned,
               SUM(CASE WHEN state = 'RUNNING' THEN 1 ELSE 0 END) as running,
               SUM(CASE WHEN state = 'FAIL' THEN 1 ELSE 0 END) as failed
        FROM trials
        WHERE study_id = (SELECT study_id FROM studies WHERE study_name = ?)
    """, (study_name,))

    total, completed, pruned, running, failed = cursor.fetchone()

    # Get trial durations for completed/pruned trials
    cursor.execute("""
        SELECT (julianday(t.datetime_complete) - julianday(t.datetime_start)) * 86400 as duration_seconds,
               t.state,
               tv.value
        FROM trials t
        LEFT JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = (SELECT study_id FROM studies WHERE study_name = ?)
        AND t.state IN ('COMPLETE', 'PRUNED')
        AND t.datetime_complete IS NOT NULL
        AND t.datetime_start IS NOT NULL
    """, (study_name,))

    trials_data = cursor.fetchall()

    # Get best value
    cursor.execute("""
        SELECT MAX(tv.value) as best_value
        FROM trials t
        JOIN trial_values tv ON t.trial_id = tv.trial_id
        WHERE t.study_id = (SELECT study_id FROM studies WHERE study_name = ?)
        AND t.state = 'COMPLETE'
    """, (study_name,))

    best_value_row = cursor.fetchone()
    best_value = best_value_row[0] if best_value_row and best_value_row[0] is not None else None

    conn.close()

    return {
        'total': total or 0,
        'completed': completed or 0,
        'pruned': pruned or 0,
        'running': running or 0,
        'failed': failed or 0,
        'trials_data': trials_data,
        'best_value': best_value
    }


def estimate_completion(stats, target_trials=2000, trial0_hours=10.9):
    """Estimate completion time based on current progress."""
    total = stats['total']

    if total == 0:
        return None

    # Calculate average duration
    durations = [d[0] for d in stats['trials_data'] if d[0] is not None]

    if not durations:
        # Still in first trial, use estimate
        remaining_trials = target_trials - total
        # Bootstrap phase (30 trials)
        bootstrap_trials = min(30, remaining_trials)
        bootstrap_time = bootstrap_trials * trial0_hours * 3600

        # Pruned phase
        pruned_trials = max(0, remaining_trials - bootstrap_trials)
        pruned_time = pruned_trials * 2.5 * 3600  # Average 2.5h with pruning

        total_seconds = bootstrap_time + pruned_time
    else:
        avg_duration = sum(durations) / len(durations)
        remaining_trials = target_trials - total
        total_seconds = remaining_trials * avg_duration

    return timedelta(seconds=total_seconds)


def main():
    """Main monitoring function."""
    db_path = Path("optuna.db")
    study_name = "pc_ce_debv3_base_aug_v2"
    target_trials = 2000

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    stats = get_study_stats(str(db_path), study_name)

    print("=" * 70)
    print(f" HPO PROGRESS: {study_name}")
    print("=" * 70)

    total = stats['total']
    completed = stats['completed']
    pruned = stats['pruned']
    running = stats['running']
    failed = stats['failed']

    progress_pct = (total / target_trials * 100) if target_trials > 0 else 0

    print(f"\nTrial Status:")
    print(f"  Total trials: {total}/{target_trials} ({progress_pct:.1f}%)")
    print(f"  Completed: {completed}")
    print(f"  Pruned: {pruned}")
    print(f"  Running: {running}")
    print(f"  Failed: {failed}")

    if stats['best_value'] is not None:
        print(f"\n  Best F1 Score: {stats['best_value']:.4f}")

    # Duration statistics
    if stats['trials_data']:
        durations = [d[0]/3600 for d in stats['trials_data'] if d[0] is not None]  # Convert to hours
        completed_durations = [d[0]/3600 for d in stats['trials_data'] if d[1] == 'COMPLETE' and d[0] is not None]
        pruned_durations = [d[0]/3600 for d in stats['trials_data'] if d[1] == 'PRUNED' and d[0] is not None]

        print(f"\nTrial Durations:")
        print(f"  Average (all): {sum(durations)/len(durations):.1f} hours")
        if completed_durations:
            print(f"  Average (completed): {sum(completed_durations)/len(completed_durations):.1f} hours")
        if pruned_durations:
            print(f"  Average (pruned): {sum(pruned_durations)/len(pruned_durations):.1f} hours")
        print(f"  Min: {min(durations):.1f} hours")
        print(f"  Max: {max(durations):.1f} hours")

    # Estimate completion
    estimated_time = estimate_completion(stats, target_trials)

    if estimated_time:
        eta = datetime.now() + estimated_time
        days = estimated_time.days
        hours = estimated_time.seconds // 3600

        print(f"\nEstimated Completion:")
        print(f"  Remaining time: {days} days, {hours} hours")
        print(f"  ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

        # Warning if too long
        if days > 60:
            print(f"\n⚠️  WARNING: Estimated completion is {days} days away!")
            print(f"   Consider reducing n_trials to 100-200 for faster completion.")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
