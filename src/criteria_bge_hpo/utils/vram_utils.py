"""VRAM utilities for automatic batch size detection and gradient accumulation."""

import math
from typing import Dict, Tuple

import torch
from transformers import AutoModel, AutoModelForSequenceClassification


def get_gpu_vram_info(device: int = 0) -> Dict[str, float]:
    """
    Get GPU VRAM information.

    Args:
        device: CUDA device index

    Returns:
        Dictionary with keys:
            - total_gb: Total VRAM in GB
            - available_gb: Available VRAM in GB
            - used_gb: Used VRAM in GB
    """
    if not torch.cuda.is_available():
        return {"total_gb": 0.0, "available_gb": 0.0, "used_gb": 0.0}

    available, total = torch.cuda.mem_get_info(device)
    total_gb = total / (1024**3)
    available_gb = available / (1024**3)
    used_gb = total_gb - available_gb

    return {
        "total_gb": total_gb,
        "available_gb": available_gb,
        "used_gb": used_gb,
    }


def probe_max_batch_size(
    model_name: str,
    tokenizer,
    max_length: int = 512,
    vram_headroom: float = 0.10,
    use_bf16: bool = True,
    device: int = 0,
) -> int:
    """
    Probe maximum safe batch size using binary search.

    Loads the model, creates dummy batches, and tests different batch sizes
    to find the maximum that fits in VRAM with headroom.

    Args:
        model_name: HuggingFace model name
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        vram_headroom: Fraction of VRAM to reserve as headroom (0.10 = 10%)
        use_bf16: Use bfloat16 precision
        device: CUDA device index

    Returns:
        Maximum safe batch size
    """
    if not torch.cuda.is_available():
        return 4  # Default batch size for CPU

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    device_obj = torch.device(f"cuda:{device}")

    # Try to load classification model first, fallback to base model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, torch_dtype=dtype)
    except Exception:
        model = AutoModel.from_pretrained(model_name, torch_dtype=dtype)

    model = model.to(device_obj)
    model.eval()

    # Binary search for max batch size
    low, high = 1, 256
    max_safe_batch = 1

    # Create dummy input
    dummy_text = "This is a test sentence. " * 20  # Long enough to test memory
    dummy_inputs = tokenizer(
        dummy_text,
        dummy_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    while low <= high:
        mid = (low + high) // 2

        # Create batch
        batch = {
            "input_ids": dummy_inputs["input_ids"].repeat(mid, 1).to(device_obj),
            "attention_mask": dummy_inputs["attention_mask"].repeat(mid, 1).to(device_obj),
        }
        if "token_type_ids" in dummy_inputs:
            batch["token_type_ids"] = dummy_inputs["token_type_ids"].repeat(mid, 1).to(device_obj)

        try:
            with torch.no_grad():
                _ = model(**batch)
            torch.cuda.synchronize()

            # Success - try larger batch
            max_safe_batch = mid
            low = mid + 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM - try smaller batch
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise

    # Clean up
    del model
    torch.cuda.empty_cache()

    # Apply headroom safety factor
    safe_batch = int(max_safe_batch * (1 - vram_headroom))
    return max(1, safe_batch)


def calculate_gradient_accumulation(
    sampled_batch_size: int, max_safe_batch_size: int
) -> Tuple[int, int]:
    """
    Calculate physical batch size and gradient accumulation steps.

    If sampled batch size exceeds VRAM limit, splits it across multiple
    forward passes using gradient accumulation.

    Args:
        sampled_batch_size: Desired effective batch size (from HPO)
        max_safe_batch_size: Maximum batch size that fits in VRAM

    Returns:
        Tuple of (physical_batch_size, gradient_accumulation_steps)
    """
    if sampled_batch_size <= max_safe_batch_size:
        return (sampled_batch_size, 1)

    # Calculate gradient accumulation
    grad_accum_steps = math.ceil(sampled_batch_size / max_safe_batch_size)
    physical_batch = max_safe_batch_size

    return (physical_batch, grad_accum_steps)
