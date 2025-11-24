"""BERT-based classifier for DSM-5 NLI."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class BERTClassifier(nn.Module):
    """BERT classifier with dropout and linear classification head."""

    def __init__(
        self,
        model_name: str = "google/bert-base-uncased",
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_bert: bool = False,
    ):
        """Initialize BERT classifier.

        Args:
            model_name: Pretrained model name
            num_labels: Number of output classes
            dropout: Dropout probability
            freeze_bert: Freeze BERT weights
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.dropout_prob = dropout

        # Load BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

        # Detect model capabilities for multi-model support
        # Some models (RoBERTa, DeBERTa, ModernBERT) don't use token_type_ids
        # RoBERTa has type_vocab_size=1 but ignores token_type_ids
        # BERT has type_vocab_size=2 and uses token_type_ids
        # DeBERTa has type_vocab_size=0 and doesn't use token_type_ids
        self.uses_token_type_ids = (
            hasattr(self.config, "type_vocab_size") and self.config.type_vocab_size > 1
        )

        # Some models don't have pooler_output, need to use CLS token manually
        # This will be checked dynamically in forward pass
        self.has_pooler = None  # Determined on first forward pass

        # Freeze BERT if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)

        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Segment IDs (optional, not used by RoBERTa/DeBERTa)
            labels: Labels for loss computation (optional)

        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        # Only pass token_type_ids if the model supports them
        if self.uses_token_type_ids and token_type_ids is not None:
            outputs = self.bert(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use pooler_output if available, otherwise extract CLS token manually
        # Detect pooler availability on first forward pass
        if self.has_pooler is None:
            self.has_pooler = (
                hasattr(outputs, "pooler_output") and outputs.pooler_output is not None
            )

        pooled_output = (
            outputs.pooler_output if self.has_pooler else outputs.last_hidden_state[:, 0, :]
        )

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

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

        # Save BERT
        self.bert.save_pretrained(save_directory)

        # Save classifier
        torch.save(
            {
                "classifier": self.classifier.state_dict(),
                "dropout_prob": self.dropout_prob,
                "num_labels": self.num_labels,
            },
            os.path.join(save_directory, "classifier.pt"),
        )

    @classmethod
    def from_pretrained(cls, load_directory: str):
        """Load model from directory."""
        import os

        # Load classifier config
        classifier_path = os.path.join(load_directory, "classifier.pt")
        classifier_state = torch.load(classifier_path, map_location="cpu")

        # Create model
        model = cls(
            model_name=load_directory,
            num_labels=classifier_state["num_labels"],
            dropout=classifier_state["dropout_prob"],
        )

        # Load classifier weights
        model.classifier.load_state_dict(classifier_state["classifier"])

        return model
