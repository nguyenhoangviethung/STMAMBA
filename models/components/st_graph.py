"""Spatio-Temporal Graph (ST-Graph) module.

Provides simple temporal and spatial message passing for chunked
video features. Designed to be modular and lightweight for research.
"""
from typing import Optional
import torch
import torch.nn as nn


class STGraph(nn.Module):
    """A compact spatio-temporal graph network.

    Expects input of shape (B, T, N, D) where T is temporal chunks
    and N is spatial nodes (objects / patches). The module performs
    temporal message passing followed by spatial attention per timestep.
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1, temporal_kernel: int = 3, max_T: int = 128) -> None:
        super().__init__()
        self.dim = dim
        self.temporal_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))
        # learnable temporal positional encoding (for up to max_T timesteps)
        self.max_T = int(max_T)
        self.temporal_pos = nn.Parameter(torch.randn(self.max_T, dim))
        # small temporal conv applied per-node across time
        self.temporal_conv = nn.Conv1d(dim, dim, kernel_size=temporal_kernel, padding=temporal_kernel // 2, groups=1)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.update = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        # residual scaling parameter
        self.res_alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, N, D)
            mask: optional boolean mask for attention (B, T, N) or (B, N)
        Returns:
            out: (B, T, N, D)
        """
        B, T, N, D = x.shape

        # Temporal message: apply temporal conv per-node
        # x: (B, T, N, D) -> process nodes independently using conv1d over T
        # add learnable positional encodings (clip to max_T)
        if T > self.max_T:
            # repeat or truncate pos encodings to match T
            pos = self.temporal_pos.repeat(int(np.ceil(T / self.max_T)), 1)[:T]
        else:
            pos = self.temporal_pos[:T]
        pos = pos.to(x.device)
        x = x + pos.unsqueeze(1).unsqueeze(0)  # (1,1,T,D) -> broadcast to (B,T,N,D)

        x_reshaped = x.permute(0, 2, 3, 1).contiguous()  # (B, N, D, T)
        Bn, Nn, Dd, Tt = x_reshaped.shape
        x_conv = x_reshaped.view(B * N, Dd, Tt)
        x_conv = self.temporal_conv(x_conv)  # (B*N, D, T)
        x_conv = x_conv.view(B, N, Dd, Tt).permute(0, 3, 1, 2).contiguous()  # (B, T, N, D)

        # linear projection residual
        x_tem = self.temporal_proj(x_conv)

        out = torch.zeros_like(x)
        # Spatial attention per timestep (nodes attend to nodes)
        for t in range(T):
            q = x_tem[:, t]  # (B, N, D)
            k = x_tem[:, t]
            v = x_tem[:, t]
            attn_mask = None
            if mask is not None:
                if mask.dim() == 3:
                    attn_mask = ~mask[:, t].bool()
                elif mask.dim() == 2:
                    attn_mask = ~mask.bool()
            attn_out, _ = self.spatial_attn(q, k, v, key_padding_mask=attn_mask)
            out[:, t] = self.update(attn_out) + self.res_alpha * x[:, t]

        return out
