#!/usr/bin/env python3
"""
Verify HPO Settings Before Launch
==================================
Confirms that all updated settings are correctly configured:
  - 100 epochs for training
  - Patience 20 for early stopping
  - torch.compile implementation fixed
  - Augmentation search space included in pc_ce.yaml
"""

import sys
from pathlib import Path
import yaml
from rich.console import Console
from rich.table import Table

console = Console()


def check_training_config():
    """Verify training configuration (epochs, patience)."""
    config_path = Path("configs/training/default.yaml")

    if not config_path.exists():
        return False, "configs/training/default.yaml not found"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    epochs = config.get("num_epochs")
    patience = config.get("early_stopping_patience")
    use_compile = config.get("optimization", {}).get("use_torch_compile")

    issues = []
    if epochs != 100:
        issues.append(f"num_epochs={epochs} (expected 100)")
    if patience != 20:
        issues.append(f"early_stopping_patience={patience} (expected 20)")
    if use_compile is not False:
        issues.append(f"use_torch_compile={use_compile} (expected False for HPO)")

    if issues:
        return False, "; ".join(issues)

    return True, "✓ 100 epochs, patience 20, torch.compile disabled"


def check_hpo_config():
    """Verify HPO configuration (pc_ce.yaml)."""
    config_path = Path("configs/hpo/pc_ce.yaml")

    if not config_path.exists():
        return False, "configs/hpo/pc_ce.yaml not found"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    n_trials = config.get("n_trials")
    pruner_type = config.get("pruner", {}).get("type")
    aug_enable = config.get("search_space", {}).get("aug_enable")

    issues = []
    if n_trials != 2000:
        issues.append(f"n_trials={n_trials} (expected 2000)")
    if pruner_type != "HyperbandPruner":
        issues.append(f"pruner.type={pruner_type} (expected HyperbandPruner)")
    if aug_enable is None:
        issues.append("aug_enable not in search_space")

    if issues:
        return False, "; ".join(issues)

    return True, f"✓ 2000 trials, {pruner_type}, augmentation search space"


def check_trainer_implementation():
    """Verify torch.compile is properly implemented in trainer."""
    trainer_path = Path("src/criteria_bge_hpo/training/trainer.py")

    if not trainer_path.exists():
        return False, "trainer.py not found"

    with open(trainer_path) as f:
        content = f.read()

    # Check for fixed implementation
    if "torch.compile(self.model" in content:
        return True, "✓ torch.compile properly implemented"
    elif "torch.compile requested but disabled" in content:
        return False, "torch.compile still hardcoded to disable"
    else:
        return False, "torch.compile implementation unclear"


def check_model_config():
    """Verify DeBERTa model configuration."""
    config_path = Path("configs/model/deberta_nli.yaml")

    if not config_path.exists():
        return False, "configs/model/deberta_nli.yaml not found"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config.get("model_name")
    num_labels = config.get("num_labels")

    if model_name != "microsoft/deberta-v3-base":
        return False, f"model_name={model_name} (expected microsoft/deberta-v3-base)"
    if num_labels != 2:
        return False, f"num_labels={num_labels} (expected 2)"

    return True, f"✓ {model_name}, binary classification"


def check_augmentation_config():
    """Verify augmentation configuration."""
    config_path = Path("configs/augmentation/default.yaml")

    if not config_path.exists():
        return False, "configs/augmentation/default.yaml not found"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    enable = config.get("enable")
    lib = config.get("lib")
    aug_type = config.get("type")

    if lib != "nlpaug":
        return False, f"lib={lib} (expected nlpaug)"

    return True, f"✓ nlpaug library, default: enable={enable}"


def main():
    """Run all verification checks."""
    console.print("\n[bold cyan]HPO Settings Verification[/bold cyan]\n")

    checks = [
        ("Training Config", check_training_config()),
        ("HPO Config (pc_ce)", check_hpo_config()),
        ("Trainer Implementation", check_trainer_implementation()),
        ("Model Config (DeBERTa)", check_model_config()),
        ("Augmentation Config", check_augmentation_config()),
    ]

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan", width=25)
    table.add_column("Status", width=10)
    table.add_column("Details", style="dim")

    all_passed = True
    for name, (passed, message) in checks:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, status, message)
        if not passed:
            all_passed = False

    console.print(table)
    console.print()

    if all_passed:
        console.print("[bold green]✓ All checks passed! Ready to launch HPO.[/bold green]\n")
        console.print("Launch with: [yellow]./launch_deberta_hpo.sh[/yellow]\n")
        return 0
    else:
        console.print("[bold red]✗ Some checks failed. Fix issues before launching.[/bold red]\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
