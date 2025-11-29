"""Factory for text augmentation utilities."""

from __future__ import annotations

from typing import Callable, Optional

import torch


class AugmentationFactory:
    """Create augmentation callables based on configuration."""

    @staticmethod
    def get_augmenter(config) -> Optional[Callable[[str], str]]:
        """Return a callable augmenter or None if disabled."""
        if config is None or not getattr(config, "enable", False):
            return None

        lib = str(getattr(config, "lib", "nlpaug")).lower()
        aug_type = str(getattr(config, "type", "")).lower()

        if lib == "nlpaug":
            try:
                import nlpaug.augmenter.word as naw
            except ImportError as exc:
                raise ImportError(
                    "nlpaug is required for augmentation. Install via `pip install nlpaug`."
                ) from exc

            if aug_type == "contextual":
                device = "cuda" if torch.cuda.is_available() else "cpu"
                augmenter = naw.ContextualWordEmbsAug(
                    model_path="bert-base-uncased",
                    action="substitute",
                    device=device,
                )
            elif aug_type == "synonym":
                augmenter = naw.SynonymAug(aug_src="wordnet")
            else:
                raise ValueError(f"Unsupported nlpaug augmentation type: {aug_type}")
        else:
            raise ValueError(f"Unsupported augmentation library: {lib}")

        return lambda text: augmenter.augment(text)
