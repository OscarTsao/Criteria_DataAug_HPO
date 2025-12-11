
import optuna
import sys

storage = "sqlite:///optuna.db"
try:
    studies = optuna.study.get_all_study_summaries(storage=storage)
    if not studies:
        print("No studies found in optuna.db")
    else:
        for study in studies:
            print(f"Study Name: {study.study_name}")
            print(f"  Trials: {study.n_trials}")
            print(f"  Best Value: {study.best_trial.value if study.best_trial else 'None'}")
            print(f"  Direction: {study.direction}")
            print("-" * 20)
except Exception as e:
    print(f"Error accessing optuna.db: {e}")
