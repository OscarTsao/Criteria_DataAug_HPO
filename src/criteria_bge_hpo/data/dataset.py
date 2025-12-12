"""PyTorch Dataset for DSM-5 NLI."""

import multiprocessing as mp
import random
import warnings
from typing import Dict, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from criteria_bge_hpo.data.augmentation import AugmentationFactory
from criteria_bge_hpo.data.augmentation_stats import AugmentationStats


class DSM5NLIDataset(Dataset):
    """Dataset for DSM-5 NLI binary classification.

    Each sample is a (post, criterion) pair with binary label.
    Input format: [CLS] post [SEP] criterion [SEP]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_length: int = 512,
        verify_format: bool = False,
        model_name: Optional[str] = None,
        augment_config=None,
    ):
        """Initialize dataset.

        Args:
            data: DataFrame with columns: post, criterion, label
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            verify_format: Whether to validate column presence (debug helper)
            model_name: Model name for detecting token_type_ids support (optional)
            augment_config: Optional augmentation config namespace
        """
        if verify_format:
            required_columns = {"post", "criterion", "label"}
            if augment_config is not None:
                required_columns.add("evidence_text")
            missing = required_columns - set(data.columns)
            if missing:
                raise ValueError(f"DSM5NLIDataset missing required columns: {sorted(missing)}")

        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment_config = augment_config
        self.augmenter = AugmentationFactory.get_augmenter(augment_config)
        self.augmentation_stats = AugmentationStats()
        
        # Detect if tokenizer produces token_type_ids
        test_encoding = self.tokenizer("test", "test", return_tensors="pt")
        self.has_token_type_ids = "token_type_ids" in test_encoding

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        row = self.data.iloc[idx]
        label_value = int(row["label"])
        post_text = str(row["post"])
        criterion_text = str(row["criterion"])

        if self.augmenter:
            evidence_text = str(row.get("evidence_text", "") or "").strip()
            prob = float(getattr(self.augment_config, "prob", 0.0) or 0.0)
            should_augment = (
                label_value == 1 and bool(evidence_text) and random.random() < prob
            )
            
            if not should_augment:
                self.augmentation_stats.record_skip()
            else:
                try:
                    augmented_span = self.augmenter(evidence_text)
                    if isinstance(augmented_span, list):
                        augmented_span = augmented_span[0]
                    if evidence_text and evidence_text in post_text:
                        post_text = post_text.replace(evidence_text, str(augmented_span), 1)
                    
                    # Record successful augmentation
                    method_name = getattr(self.augmenter, "name", "unknown")
                    self.augmentation_stats.record_augmentation(method_name)
                except Exception as e:
                    # Record failure
                    method_name = getattr(self.augmenter, "name", "unknown")
                    self.augmentation_stats.record_failure(method_name, str(e))

        encoding = self.tokenizer(
            post_text,
            criterion_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item: Dict[str, torch.Tensor] = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_value, dtype=torch.long),
        }
        if self.has_token_type_ids and "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        return item

    def get_augmentation_stats(self) -> Dict:
        """Return augmentation statistics as dictionary."""
        return self.augmentation_stats.to_dict()

    def reset_augmentation_stats(self) -> None:
        """Reset augmentation statistics."""
        self.augmentation_stats.reset()


def _supports_multiprocessing() -> bool:
    """Return True if Python multiprocessing primitives are usable."""
    try:
        ctx = mp.get_context()
        queue = ctx.Queue(maxsize=1)
        queue.put_nowait(None)
        queue.close()
        queue.join_thread()
        return True
    except (OSError, PermissionError):
        return False


def create_dataloaders(
    train_dataset, val_dataset, batch_size: int, num_workers: int = 4, pin_memory: bool = True
):
    """Create train and validation dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    use_pin_memory = pin_memory and torch.cuda.is_available()

    # Some sandboxes disable semaphores/shared memory; fall back to single-process loading
    worker_count = num_workers
    if worker_count > 0 and not _supports_multiprocessing():
        warnings.warn(
            "Multiprocessing dataloaders are not available; falling back to num_workers=0.",
            RuntimeWarning,
        )
        worker_count = 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=worker_count,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader
