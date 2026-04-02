"""ShaRP: adaptive token pruning module.

Implements a lightweight learned scorer to rank visual tokens and
keep only the top-K per sample. Returns pruned tokens and indices
so the caller can maintain mapping for attention keys/values.
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn


def _to_int(value: float) -> int:
    return int(max(1, value))


class ShaRPPruner(nn.Module):
    """Adaptive token pruner.

    This module computes per-token scores and keeps the top-K tokens per
    batch element. It returns the pruned tokens, the indices of kept tokens,
    and a soft weight per kept token (useful for soft-attention weighting).

    Args:
        dim: token feature dimensionality.
        keep_ratio: fraction of tokens to keep (0,1].
        ema_alpha: smoothing factor for running mean of scores.
        min_keep: minimum tokens to keep.
    """

    def __init__(self, dim: int, keep_ratio: float = 0.7, ema_alpha: float = 0.9, min_keep: int = 1, threshold: Optional[float] = None) -> None:
        super().__init__()
        self.score_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.keep_ratio: float = float(keep_ratio)
        self.ema_alpha: float = float(ema_alpha)
        self.min_keep: int = int(min_keep)
        # If set, absolute score threshold used to keep tokens (per-sample)
        self.threshold: Optional[float] = float(threshold) if threshold is not None else None
        # scalar running average for stability across varying batch sizes
        self.register_buffer("running_mean", torch.tensor(0.0))

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Score and prune tokens.

        Args:
            tokens: tensor of shape (B, N, D)

        Returns:
            pruned: (B, K, D) pruned token tensor
            keep_idx: (B, K) indices of kept tokens along N
            weights: (B, K) soft weights in (0,1)
        """
        if tokens.dim() != 3:
            raise ValueError("tokens must be shape (B, N, D)")

        B, N, D = tokens.shape
        scores = self.score_proj(tokens).squeeze(-1)  # (B, N)

        # update running mean for stability (use mean of absolute scores)
        batch_mean = scores.abs().mean().detach()
        self.running_mean = self.running_mean * self.ema_alpha + batch_mean * (1.0 - self.ema_alpha)

        # normalize scores per-sample for robust top-k selection
        scores_norm = (scores - scores.mean(dim=1, keepdim=True)) / (scores.std(dim=1, keepdim=True) + 1e-6)

        k = max(self.min_keep, int(N * self.keep_ratio))

        if self.threshold is not None:
            # per-sample boolean mask of tokens to keep by absolute raw score
            keep_mask = (scores.abs() >= float(self.threshold))
            # ensure at least k tokens per sample: if fewer, fall back to topk
            counts = keep_mask.sum(dim=1)
            topk = torch.topk(scores_norm, k=k, dim=1).indices  # (B, k)
            # build final indices per batch element
            final_idx = []
            for b in range(B):
                if counts[b] >= k:
                    idxs = torch.nonzero(keep_mask[b], as_tuple=False).squeeze(1)
                    # if more than k, take highest scored among them
                    if idxs.numel() > k:
                        sub_scores = scores_norm[b, idxs]
                        sub_top = torch.topk(sub_scores, k=k).indices
                        chosen = idxs[sub_top]
                    else:
                        chosen = idxs
                    # pad/truncate to exactly k
                    if chosen.numel() < k:
                        pad_needed = k - chosen.numel()
                        pad = topk[b, :pad_needed]
                        chosen = torch.cat([chosen, pad], dim=0)
                    final_idx.append(chosen[:k])
                else:
                    final_idx.append(topk[b])
            topk = torch.stack(final_idx, dim=0)
        else:
            topk = torch.topk(scores_norm, k=k, dim=1).indices  # (B, k)

        # Gather pruned tokens efficiently
        idx_exp = topk.unsqueeze(-1).expand(-1, -1, D)  # (B, k, D)
        pruned = torch.gather(tokens, dim=1, index=idx_exp)

        # Also return soft weights for kept tokens (sigmoid of raw scores)
        kept_scores = torch.gather(scores, dim=1, index=topk)
        weights = torch.sigmoid(kept_scores)
        return pruned, topk, weights
