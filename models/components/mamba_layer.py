"""MambaLayer: Selective State Space Model (Mamba) block.

Implements the official Mamba layer from mamba-ssm package.
Memory-efficient, linear complexity O(L), used as backbone.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from mamba_ssm import Mamba   


class MambaLayer(nn.Module):
    """Mamba SSM layer (replaces the old Transformer block).

    Args:
        dim: hidden dimensionality (d_model)
        d_state: SSM state dimension (default 16)
        d_conv: local convolution width (default 4)
        expand: expansion factor for inner dim (default 2)
        use_checkpoint: whether to use gradient checkpointing (saves VRAM)
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            # Các tham số mặc định khác của mamba-ssm đã tối ưu
        )
        self.use_checkpoint = use_checkpoint

    def _forward_block(self, x: torch.Tensor) -> torch.Tensor:
        """Core forward without checkpointing."""
        x_norm = self.norm(x)           # pre-norm
        x_mamba = self.mamba(x_norm)    # Mamba SSM
        return x + x_mamba              # residual connection

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: input tensor of shape (B, L, D)
            attn_mask: ignored (Mamba không dùng attention mask)
        """
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward_block, x)
        return self._forward_block(x)