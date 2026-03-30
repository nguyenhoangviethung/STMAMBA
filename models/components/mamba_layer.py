"""MambaLayer: memory-efficient transformer block.

This layer uses PyTorch's MultiheadAttention and supports gradient
checkpointing optionally by wrapping the core attention+ffn block.
"""
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Tuple


class MambaLayer(nn.Module):
    """Single Mamba transformer-like layer.

    Args:
        dim: hidden dimensionality
        n_heads: attention heads
        mlp_dim: feed-forward hidden dim
        dropout: dropout rate
        use_checkpoint: whether to use gradient checkpointing
    """

    def __init__(self, dim: int, n_heads: int = 8, mlp_dim: int = 2048, dropout: float = 0.1, use_checkpoint: bool = False) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        # Prefer FlashAttention if available for speed/memory
        self._use_flash = False
        try:
            from flash_attn import FlashAttention

            self.flash_attn = FlashAttention(dropout=dropout)
            self._use_flash = True
            self._use_torch_sdp = False
        except Exception:
            # Try to use PyTorch scaled_dot_product_attention if available (PyTorch 2.0+)
            try:
                from torch.nn.functional import scaled_dot_product_attention  # type: ignore

                self._use_flash = False
                self._use_torch_sdp = True
                self.attn = None
            except Exception:
                self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True, dropout=dropout)
                self._use_flash = False
                self._use_torch_sdp = False
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_dim, dim), nn.Dropout(dropout))
        self.use_checkpoint = use_checkpoint

    def _forward_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Attention (self)
        x_ln = self.norm1(x)
        if self._use_flash:
            # FlashAttention expects (B, L, D)
            attn_out = self.flash_attn(x_ln, x_ln, x_ln)
        elif getattr(self, "_use_torch_sdp", False):
            # PyTorch's scaled_dot_product_attention expects (B, L, D)
            # Need to reshape to (B, L, D) and provide attn_mask if present
            # We implement simple self-attention without causal mask here.
            q = x_ln
            k = x_ln
            v = x_ln
            # scaled_dot_product_attention signature: (q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            try:
                from torch.nn.functional import scaled_dot_product_attention

                attn_out = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
            except Exception:
                # fallback to multihead if something unexpected happens
                attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=attn_mask)  # type: ignore
        else:
            attn_out, _ = self.attn(x_ln, x_ln, x_ln, attn_mask=attn_mask)  # type: ignore
        x = x + attn_out
        # FFN
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for sequence x of shape (B, L, D).

        Args:
            x: input tensor (B, L, D)
            attn_mask: optional attention mask compatible with MultiheadAttention

        Returns:
            Tensor of same shape (B, L, D)
        """
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_block, x, attn_mask)
        return self._forward_block(x, attn_mask)
