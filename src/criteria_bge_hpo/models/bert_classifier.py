"""Sequence classification model wrapper for the BGE reranker."""

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification


class BERTClassifier(nn.Module):
    """Wrap AutoModelForSequenceClassification for the BGE reranker."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        num_labels: int = 1,
        freeze_backbone: bool = False,
        config: Optional[AutoConfig] = None,
        classifier_dropout: Optional[float] = None,
        hidden_dropout: Optional[float] = None,
        attention_dropout: Optional[float] = None,
        focal_gamma: float = 2.0,
    ):
        """Initialize the reranker model.

        Args:
            model_name: Pretrained model identifier or local path.
            num_labels: Number of output labels. Keep at 1 to align with the pretrained head.
            freeze_backbone: Whether to freeze the encoder parameters.
            config: Optional pre-loaded configuration to reuse.
            classifier_dropout: Dropout rate for classifier head (0.0 to 1.0).
            hidden_dropout: Dropout rate for hidden layers (0.0 to 1.0).
            attention_dropout: Dropout rate for attention weights (0.0 to 1.0).
            focal_gamma: Focusing parameter for focal loss (default 2.0).
        """
        super().__init__()

        self.model_name = model_name
        self.config = config or AutoConfig.from_pretrained(model_name)
        self.config.num_labels = num_labels or self.config.num_labels

        # Set dropout rates if provided
        if classifier_dropout is not None:
            self.config.classifier_dropout = classifier_dropout
        if hidden_dropout is not None:
            self.config.hidden_dropout_prob = hidden_dropout
        if attention_dropout is not None:
            self.config.attention_probs_dropout_prob = attention_dropout

        # Store focal loss parameter
        self.focal_gamma = focal_gamma

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=self.config,
            ignore_mismatched_sizes=True,
        )
        self.num_labels = self.model.config.num_labels

        # Detect model capabilities
        self.uses_token_type_ids = (
            hasattr(self.model.config, "type_vocab_size")
            and self.model.config.type_vocab_size is not None
            and self.model.config.type_vocab_size > 1
        )

        if freeze_backbone:
            backbone = getattr(self.model, getattr(self.model, "base_model_prefix", ""), None)
            if backbone is not None:
                for param in backbone.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """Forward pass."""
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.uses_token_type_ids and token_type_ids is not None:
            model_inputs["token_type_ids"] = token_type_ids

        outputs = self.model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]

        loss = None
        if labels is not None:
            if logits.shape[-1] == 1:
                # Binary classification with BCE or Focal Loss
                labels = labels.float()
                if self.focal_gamma != 2.0:
                    # Use focal loss with custom gamma
                    bce_loss = nn.functional.binary_cross_entropy_with_logits(
                        logits.view(-1), labels.view(-1), reduction="none"
                    )
                    probs = torch.sigmoid(logits.view(-1))
                    pt = torch.where(labels.view(-1) == 1, probs, 1 - probs)
                    focal_weight = (1 - pt) ** self.focal_gamma
                    loss = (focal_weight * bce_loss).mean()
                else:
                    # Standard BCE loss
                    loss_fct = nn.BCEWithLogitsLoss()
                    loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                # Multi-class classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return {"logits": logits, "loss": loss}

    def get_num_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_total_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def save_pretrained(self, save_directory: str):
        """Save model to directory."""
        import os

        os.makedirs(save_directory, exist_ok=True)
        self.model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model from directory."""
        config = AutoConfig.from_pretrained(load_directory)
        return cls(model_name=load_directory, num_labels=config.num_labels, config=config)
