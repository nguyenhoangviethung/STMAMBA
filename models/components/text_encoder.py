"""Text encoder wrapper using Hugging Face BERT.

Provides a small adapter around `transformers` to produce token
embeddings and pooled representations suitable for downstream fusion.
"""
from typing import Tuple, List
import torch
import torch.nn as nn

from transformers import AutoModel, AutoTokenizer


class TextEncoder(nn.Module):
    """BERT-based text encoder wrapper.

    Loads a Hugging Face encoder and tokenizer. On forward it returns the
    last hidden states and a pooled vector. Inputs are raw strings.

    Args:
        model_name: HF model id (default: bert-base-uncased).
    """

    def __init__(self, model_name: str = "bert-base-uncased") -> None:
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of texts.

        Args:
            texts: list of raw strings length B.

        Returns:
            last_hidden: (B, L, D)
            pooled: (B, D)
        """
        device = next(self.model.parameters()).device
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = self.model(**enc)
        last_hidden = out.last_hidden_state
        pooled = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else last_hidden[:, 0]
        return last_hidden, pooled
