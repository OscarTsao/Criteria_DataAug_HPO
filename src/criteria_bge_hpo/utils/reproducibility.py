"""Reproducibility utilities."""

import random
import numpy as np
import torch
from rich.console import Console

console = Console()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    console.print(f"[green]✓[/green] Set random seed: {seed}")


def enable_deterministic(tf32: bool = True, enable_cudnn_benchmark: bool = True):
    """Configure TF32 and cuDNN settings for performance.

    Note: Deterministic algorithms are always DISABLED for performance.
    Use set_seed() for reproducibility instead.
    """
    # Always disable deterministic algorithms for performance
    torch.use_deterministic_algorithms(False)
    console.print(
        "[yellow]Deterministic algorithms: DISABLED (performance optimized)[/yellow]"
    )

    if torch.cuda.is_available():
        # Force cudnn benchmark for speed
        torch.backends.cudnn.benchmark = enable_cudnn_benchmark
        console.print(
            f"[yellow]cuDNN benchmark: {'ENABLED' if enable_cudnn_benchmark else 'DISABLED'} "
            "(performance optimized)[/yellow]"
        )

        # TF32 configuration
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            console.print(
                "[green]TF32 enabled for matmul and cuDNN (Ampere+ GPUs)[/green]"
            )
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            console.print("[yellow]TF32 disabled[/yellow]")
    else:
        console.print("[yellow]⚠[/yellow] CUDA not available; TF32 flag ignored")


def get_device():
    """Get device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        console.print(f"[green]✓[/green] Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        console.print("[yellow]⚠[/yellow] CUDA not available, using CPU")

    return device


def verify_cuda_setup():
    """Verify CUDA setup."""
    if not torch.cuda.is_available():
        console.print("[yellow]⚠[/yellow] CUDA not available")
        return

    console.print("\n[cyan]═══════════════════════════════════════════════════════════[/cyan]")
    console.print(
        "[cyan bold]                    CUDA INFORMATION                       [/cyan bold]"
    )
    console.print("[cyan]═══════════════════════════════════════════════════════════[/cyan]")
    console.print(f"  Device: {torch.cuda.get_device_name(0)}")
    console.print(f"  CUDA Version: {torch.version.cuda}")
    console.print(f"  Device Capability: {torch.cuda.get_device_capability(0)}")

    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    console.print(f"  Total Memory: {memory_total:.2f} GB")
    console.print(f"  BF16 Supported: {torch.cuda.is_bf16_supported()}")
    console.print("[cyan]═══════════════════════════════════════════════════════════[/cyan]\n")


def setup_reproducibility(
    seed: int,
    enable_tf32: bool = True,
    enable_cudnn_benchmark: bool = True,
) -> None:
    """
    Complete reproducibility setup: set seed and configure PyTorch.

    Args:
        seed: Random seed value
        enable_tf32: Enable TensorFloat-32 for Ampere+ GPUs
        enable_cudnn_benchmark: Enable cudnn benchmarking
    """
    set_seed(seed)
    enable_deterministic(
        tf32=enable_tf32,
        enable_cudnn_benchmark=enable_cudnn_benchmark,
    )
