"""Utilities for finding maximum batch size and calculating gradient accumulation."""

import gc
from typing import Tuple

import torch
from rich.console import Console
from transformers import PreTrainedModel, PreTrainedTokenizer

console = Console()


def find_max_physical_batch_size(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    device: torch.device,
    safety_margin: float = 0.9,
) -> int:
    """
    Binary search to find maximum physical batch size that fits in GPU memory.

    Args:
        model: The model to test
        tokenizer: Tokenizer for creating dummy inputs
        max_length: Maximum sequence length
        device: Device to run on (should be CUDA)
        safety_margin: Safety factor to apply to detected maximum (0.0-1.0)

    Returns:
        Maximum safe physical batch size
    """
    if not torch.cuda.is_available():
        console.print("[yellow]CUDA not available, returning batch_size=8[/yellow]")
        return 8

    model = model.to(device)
    model.train()

    # Binary search bounds
    min_batch = 1
    max_batch = 256
    best_batch = 1

    console.print(
        f"[cyan]Searching for maximum batch size (max_length={max_length})...[/cyan]"
    )

    while min_batch <= max_batch:
        candidate_batch = (min_batch + max_batch) // 2

        try:
            # Clear cache before test
            torch.cuda.empty_cache()
            gc.collect()

            # Create dummy inputs
            dummy_input_ids = torch.randint(
                0,
                tokenizer.vocab_size,
                (candidate_batch, max_length),
                device=device,
            )
            dummy_attention_mask = torch.ones(
                (candidate_batch, max_length),
                dtype=torch.long,
                device=device,
            )
            dummy_labels = torch.randint(
                0,
                model.config.num_labels,
                (candidate_batch,),
                device=device,
            )

            # Forward pass
            outputs = model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask,
                labels=dummy_labels,
            )

            # Backward pass (most memory intensive)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            loss.backward()

            # Success - try larger batch
            best_batch = candidate_batch
            min_batch = candidate_batch + 1

            console.print(
                f"[green]✓[/green] Batch size {candidate_batch} fits in memory"
            )

            # Clean up
            del dummy_input_ids, dummy_attention_mask, dummy_labels, outputs, loss
            model.zero_grad()
            torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # OOM - try smaller batch
                max_batch = candidate_batch - 1
                console.print(
                    f"[red]✗[/red] Batch size {candidate_batch} caused OOM"
                )

                # Clean up
                model.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

    # Apply safety margin
    safe_batch = max(1, int(best_batch * safety_margin))

    console.print(
        f"[bold green]Maximum physical batch size: {best_batch} "
        f"(safe: {safe_batch} with {safety_margin:.0%} margin)[/bold green]"
    )

    return safe_batch


def calculate_gradient_accumulation_steps(
    target_effective_batch: int,
    physical_batch: int,
) -> int:
    """
    Calculate gradient accumulation steps to achieve target effective batch size.

    Args:
        target_effective_batch: Desired effective batch size
        physical_batch: Physical batch size that fits in GPU memory

    Returns:
        Number of gradient accumulation steps (minimum 1)
    """
    steps = max(1, target_effective_batch // physical_batch)

    console.print(
        f"[cyan]Gradient accumulation: {steps} steps "
        f"({physical_batch} × {steps} = {physical_batch * steps} effective)[/cyan]"
    )

    return steps
