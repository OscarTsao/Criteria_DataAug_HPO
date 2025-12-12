"""Evaluator for model evaluation."""

import math
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm.rich import tqdm
from rich.console import Console
from rich.table import Table

console = Console()


def _format_metric(value):
    """Format metric for table display, handling NaN gracefully."""
    if value is None:
        return "N/A"

    try:
        if math.isnan(value):
            return "N/A"
    except TypeError:
        return "N/A"

    return f"{value:.4f}"


class Evaluator:
    """Evaluator for DSM-5 NLI model."""

    def __init__(self, model, device, use_bf16=False, positive_threshold: float = 0.5):
        self.model = model
        self.device = device
        self.use_bf16 = use_bf16 and torch.cuda.is_available()
        self.positive_threshold = positive_threshold
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, eval_loader, data):
        """Evaluate model."""
        all_preds, all_labels, all_probs = [], [], []

        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.amp.autocast("cuda", enabled=self.use_bf16, dtype=torch.bfloat16):
                outputs = self.model(**batch)

            logits = outputs["logits"].float()
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits).squeeze(-1)
                preds = (probs >= self.positive_threshold).long()
                prob_vector = probs
            else:
                probs = torch.softmax(logits, dim=-1).float()
                preds = torch.argmax(logits, dim=-1)
                prob_vector = probs[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
            all_probs.extend(prob_vector.cpu().numpy())

        # Compute aggregate metrics
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auc = float("nan")

        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="binary"),
            "precision": precision_score(all_labels, all_preds, average="binary"),
            "recall": recall_score(all_labels, all_preds, average="binary"),
            "auc": auc,
        }

        # Compute per-criterion metrics
        per_criterion = evaluate_per_criterion(
            all_preds, all_labels, all_probs, data["criterion_id"].values
        )

        return {
            "aggregate": metrics,
            "per_criterion": per_criterion,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
        }

    def save_predictions(self, data, predictions, probabilities, output_path):
        """Save predictions to CSV."""

        pred_df = data.copy()
        pred_df["prediction"] = predictions
        pred_df["probability"] = probabilities
        pred_df["groundtruth"] = data["label"]

        output_df = pred_df[
            [
                "post_id",
                "post",
                "criterion_id",
                "criterion",
                "prediction",
                "groundtruth",
                "probability",
            ]
        ]

        output_df.to_csv(output_path, index=False)
        console.print(f"[green]✓[/green] Saved predictions to {output_path}")


def evaluate_per_criterion(predictions, labels, probabilities, criterion_ids):
    """Compute per-criterion metrics."""
    import numpy as np

    predictions = np.array(predictions)
    labels = np.array(labels)
    probabilities = np.array(probabilities)
    criterion_ids = np.array(criterion_ids)

    per_criterion = {}

    for criterion_id in np.unique(criterion_ids):
        mask = criterion_ids == criterion_id

        criterion_preds = predictions[mask]
        criterion_labels = labels[mask]
        criterion_probs = probabilities[mask]

        try:
            auc = roc_auc_score(criterion_labels, criterion_probs)
        except ValueError:
            auc = float("nan")

        per_criterion[criterion_id] = {
            "f1": f1_score(criterion_labels, criterion_preds, average="binary"),
            "accuracy": accuracy_score(criterion_labels, criterion_preds),
            "precision": precision_score(criterion_labels, criterion_preds, average="binary"),
            "recall": recall_score(criterion_labels, criterion_preds, average="binary"),
            "auc": auc,
            "n_samples": mask.sum(),
        }

    return per_criterion


def display_per_criterion_results(per_criterion):
    """Display per-criterion results in table."""
    table = Table(title="Per-Criterion Metrics")

    table.add_column("Criterion", style="cyan")
    table.add_column("Samples", style="yellow")
    table.add_column("F1", style="green")
    table.add_column("Accuracy", style="green")
    table.add_column("Precision", style="blue")
    table.add_column("Recall", style="magenta")
    table.add_column("AUC", style="red")

    for criterion_id, metrics in sorted(per_criterion.items()):
        table.add_row(
            criterion_id,
            str(metrics["n_samples"]),
            f"{metrics['f1']:.4f}",
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            _format_metric(metrics.get("auc")),
        )

    console.print(table)
