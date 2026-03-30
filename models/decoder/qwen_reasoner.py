"""SLM Reasoner wrapper (Qwen-like) for causal decoding.

This module wraps a Hugging Face causal LM to provide a small
interface for generation and for integrating KV caches.
"""
from typing import Optional, List
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer


class QwenReasoner(nn.Module):
    """Causal LM wrapper.

    Lightweight wrapper around a HF causal LM used for generation. The
    wrapper accepts a device string and ensures tokenization and model
    execution occur on the same device.

    Args:
        model_name: HF model id for the SLM (e.g., a Qwen checkpoint).
        device: optional device string (e.g., 'cuda' or 'cpu').
    """

    def __init__(self, model_name: str = "facebook/opt-125m", device: Optional[str] = None) -> None:
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        if device is not None:
            self.device = torch.device(device)
            self.model.to(self.device)
        else:
            self.device = next(self.model.parameters()).device
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompts: List[str], max_new_tokens: int = 64, temperature: float = 0.0) -> List[str]:
        """Generate outputs for a list of prompts.

        Args:
            prompts: list of input strings (B,)
            max_new_tokens: number of new tokens to generate.
            temperature: generation temperature.

        Returns:
            List of decoded output strings.
        """
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        gen_ids = self.model.generate(**enc, max_new_tokens=max_new_tokens, temperature=temperature)
        out = [self.tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in gen_ids]
        return out
