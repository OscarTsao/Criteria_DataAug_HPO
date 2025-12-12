import optuna
import pandas as pd
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)

storage = "sqlite:///optuna.db"
study_name = "transformer_hpo"

try:
    study = optuna.load_study(study_name=study_name, storage=storage)
except KeyError:
    print(f"Study '{study_name}' not found.")
    sys.exit(1)

print(f"Analysis for Study: {study_name}")
print("=" * 60)

# Get all trials
df = study.trials_dataframe()

# Clean up column names
df.columns = [c.replace("params_", "") for c in df.columns]

# Show trials sorted by value (descending for F1, assuming maximize)
if 'value' in df.columns:
    df_sorted = df.sort_values(by='value', ascending=False)
else:
    df_sorted = df

print("\nAll Trials (Sorted by Best Value):")
# Select relevant columns to display
cols_to_show = ['number', 'value', 'state', 'learning_rate', 'batch_size', 'weight_decay'] 
# Add other params if they exist
for c in ['classifier_head', 'classifier_dropout', 'aug_enable']:
    if c in df.columns:
        cols_to_show.append(c)

# Filter columns that actually exist
cols_to_show = [c for c in cols_to_show if c in df.columns]

print(df_sorted[cols_to_show].to_string(index=False))

print("\n" + "=" * 60)
print("Failure Analysis")

failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
if failed_trials:
    print(f"\nFound {len(failed_trials)} failed trials.")
    for t in failed_trials:
        print(f"\nTrial #{t.number} Failed.")
        # Try to find error message in system attributes
        if 'fail_reason' in t.system_attrs:
             print(f"Reason: {t.system_attrs['fail_reason']}")
        else:
             # Sometimes it's just stored as 'error' or not explicitly stored in standard attributes
             # depending on how the loop was written.
             # We can check system_attrs keys.
             print(f"System Attrs: {t.system_attrs}")
else:
    print("\nNo failed trials.")

print("\n" + "=" * 60)
print("Best Trial Details")
try:
    best = study.best_trial
    print(f"Trial #{best.number}")
    print(f"Value: {best.value}")
    print("Params:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
except ValueError:
    print("No successful trials yet.")
