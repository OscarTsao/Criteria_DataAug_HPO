"""Data loading and preprocessing for DSM-5 NLI."""

from __future__ import annotations

import json
from typing import Dict, Iterable

import pandas as pd
from hydra.utils import to_absolute_path
from rich.console import Console

console = Console()


def _validate_required_columns(
    df: pd.DataFrame, required: Iterable[str], source: str
) -> None:
    """Ensure required columns exist in dataframe."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {source}: {sorted(missing)}. "
            "Verify the dataset matches the expected schema."
        )


def load_and_preprocess_data(config) -> pd.DataFrame:
    """Load and preprocess all data for DSM-5 NLI training.

    Args:
        config: Hydra configuration object

    Returns:
        DataFrame with columns: post_id, post, criterion_id, criterion, label, evidence_text
    """
    console.print("\n[cyan]═══════════════════════════════════════════════════════════[/cyan]")
    console.print(
        "[cyan bold]               DATA LOADING & PREPROCESSING                 [/cyan bold]"
    )
    console.print("[cyan]═══════════════════════════════════════════════════════════[/cyan]\n")

    # Load unified groundtruth data (already contains post/criterion pairs)
    groundtruth_path = to_absolute_path(config.data.groundtruth_csv)
    console.print(f"[yellow]Loading groundtruth from:[/yellow] {groundtruth_path}")
    groundtruth_df = pd.read_csv(groundtruth_path)
    console.print(f"[green]✓[/green] Loaded {len(groundtruth_df):,} rows")

    required_columns = {"post_id", "post", "DSM5_symptom", "groundtruth"}
    _validate_required_columns(groundtruth_df, required_columns, groundtruth_path)

    # Load DSM-5 criteria definitions
    criteria_path = to_absolute_path(config.data.criteria_json)
    console.print(f"[yellow]Loading DSM-5 criteria from:[/yellow] {criteria_path}")
    with open(criteria_path, "r", encoding="utf-8") as handle:
        criteria_data = json.load(handle)

    criteria_dict = {item["id"]: item["text"] for item in criteria_data["criteria"]}
    console.print(f"[green]✓[/green] Loaded {len(criteria_dict)} DSM-5 criteria")

    # Create NLI pairs
    console.print("\n[yellow]Preparing post-criterion pairs...[/yellow]")

    # Rename columns for consistency (groundtruth already has DSM5 A.1-A.10 ids)
    pairs_df = groundtruth_df.rename(
        columns={"DSM5_symptom": "criterion_id", "groundtruth": "label"}
    ).copy()

    # Map criterion IDs to full criterion text
    pairs_df["criterion"] = pairs_df["criterion_id"].map(criteria_dict)

    # Remove rows with missing values in required fields
    before_drop = len(pairs_df)
    pairs_df = pairs_df.dropna(subset=["post", "criterion", "criterion_id", "label"])
    dropped = before_drop - len(pairs_df)
    if dropped > 0:
        console.print(f"[yellow]• Dropped {dropped} rows with missing required fields[/yellow]")

    if pairs_df.empty:
        raise ValueError("Groundtruth dataset is empty after preprocessing.")

    # Ensure labels are integers (CSV may load as floats)
    pairs_df["label"] = pairs_df["label"].astype(int)

    # Load expert evidence annotations and map to criteria
    annotations_path = to_absolute_path(
        getattr(config.data, "annotations_csv", "data/redsm5/redsm5_annotations.csv")
    )
    console.print(f"[yellow]Loading annotations from:[/yellow] {annotations_path}")
    annotations_df = pd.read_csv(annotations_path)
    required_annotation_cols = {"post_id", "DSM5_symptom", "sentence_text", "status"}
    _validate_required_columns(annotations_df, required_annotation_cols, annotations_path)

    # Keep only positive evidence spans
    annotations_df = annotations_df[annotations_df["status"] == 1].copy()
    symptom_map: Dict[str, str] = {
        "A.1": "DEPRESSED_MOOD",
        "A.2": "ANHEDONIA",
        "A.3": "APPETITE_CHANGE",
        "A.4": "SLEEP_ISSUES",
        "A.5": "PSYCHOMOTOR",
        "A.6": "FATIGUE",
        "A.7": "WORTHLESSNESS",
        "A.8": "COGNITIVE_ISSUES",
        "A.9": "SUICIDAL_THOUGHTS",
    }

    # Map criterion IDs to symptom names for joining evidence
    pairs_df["symptom_name"] = pairs_df["criterion_id"].map(symptom_map)

    annotations_df = annotations_df.rename(columns={"DSM5_symptom": "symptom_name"})
    annotations_df = annotations_df[["post_id", "symptom_name", "sentence_text"]]

    # Merge evidence spans onto pairs
    pairs_df = pairs_df.merge(
        annotations_df,
        how="left",
        on=["post_id", "symptom_name"],
    )

    pairs_df = pairs_df.rename(columns={"sentence_text": "evidence_text"})
    pairs_df["evidence_text"] = pairs_df["evidence_text"].fillna("")

    # Select final columns in correct order
    pairs_df = pairs_df[
        ["post_id", "post", "criterion_id", "criterion", "label", "evidence_text"]
    ]

    # Optional subsampling for fast smoke tests
    sample_size = getattr(config.data, "sample_size", None)
    if sample_size:
        sample_size = min(int(sample_size), len(pairs_df))
        pairs_df = pairs_df.sample(n=sample_size, random_state=config.data.sample_seed)
        console.print(
            f"[yellow]• Using sample_size={sample_size} (seed={config.data.sample_seed}) for quick run[/yellow]"
        )

    evidence_count = int((pairs_df["evidence_text"].str.len() > 0).sum())

    positive = int((pairs_df["label"] == 1).sum())
    negative = int((pairs_df["label"] == 0).sum())

    console.print(f"[green]✓[/green] Prepared {len(pairs_df):,} NLI pairs")
    console.print(f"  • Unique posts: {pairs_df['post_id'].nunique():,}")
    console.print(f"  • Unique criteria: {pairs_df['criterion_id'].nunique():,}")
    console.print(f"  • Evidence spans linked: {evidence_count:,}")
    console.print(f"  • Positive samples: {positive:,}")
    console.print(f"  • Negative samples: {negative:,}")

    console.print("\n[cyan]═══════════════════════════════════════════════════════════[/cyan]\n")

    return pairs_df
