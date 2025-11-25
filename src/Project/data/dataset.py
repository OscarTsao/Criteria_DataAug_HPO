"""PyTorch Dataset for DSM-5 NLI."""

import multiprocessing as mp
import warnings
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


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
        model_name: str = None,
    ):
        """Initialize dataset.

        Args:
            data: DataFrame with columns: post, criterion, label
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            verify_format: Whether to validate column presence (debug helper)
            model_name: Model name for detecting token_type_ids support (optional)
        """
        if verify_format:
            required_columns = {"post", "criterion", "label"}
            missing = required_columns - set(data.columns)
            if missing:
                raise ValueError(f"DSM5NLIDataset missing required columns: {sorted(missing)}")

        self.data = data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Detect if tokenizer produces token_type_ids
        test_encoding = self.tokenizer("test", "test", return_tensors="pt")
        self.has_token_type_ids = "token_type_ids" in test_encoding

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            str(row["post"]),
            str(row["criterion"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item: Dict[str, torch.Tensor] = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(row["label"], dtype=torch.long),
        }
        if self.has_token_type_ids and "token_type_ids" in encoding:
            item["token_type_ids"] = encoding["token_type_ids"].squeeze(0)
        return item


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
