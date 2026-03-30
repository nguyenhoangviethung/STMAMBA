"""GRPO trainer utilities.

Implements a lightweight Group Relative Policy Optimization (GRPO)
surrogate that reweights per-sample supervised losses by group-relative
advantages. This avoids PPO-style rollout storage and is memory-friendly.
"""
from typing import Optional
import torch


class GRPO:
    """Simple GRPO helper to compute advantage-normalized weights and
    apply reweighted losses.

    Usage:
        grpo = GRPO(clip_ratio=0.2)
        final_loss, weights = grpo.compute_surrogate(per_sample_losses, rewards)

    Notes:
    - `per_sample_losses` is expected to be a tensor shape (B,) of
      supervised losses (e.g., cross-entropy per sample).
    - `rewards` is a tensor shape (B,) with scalar rewards (higher=better).
    - The method returns a scalar final loss (to backprop) and the
      normalized weights used for debugging/monitoring.
    """

    def __init__(self, clip_ratio: float = 0.2, eps: float = 1e-8) -> None:
        self.clip_ratio = float(clip_ratio)
        self.eps = float(eps)

    def compute_surrogate(self, per_sample_losses: torch.Tensor, rewards: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """Compute reweighted loss using group-relative advantages.

        Args:
            per_sample_losses: (B,) tensor of losses (higher worse).
            rewards: (B,) tensor of rewards (higher better).

        Returns:
            final_loss: scalar tensor
            weights: (B,) normalized weights applied to per-sample losses
        """
        if per_sample_losses.dim() != 1 or rewards.dim() != 1:
            raise ValueError("per_sample_losses and rewards must be 1-D tensors")

        # Convert supervised loss into a proxy 'value' (higher better -> negative loss)
        values = -per_sample_losses.detach()

        # If explicit rewards are provided, combine them with values
        adv = rewards.detach() + values

        # group-relative advantage: subtract group mean and normalize
        adv_mean = adv.mean()
        adv_std = adv.std(unbiased=False) + self.eps
        advantages = (adv - adv_mean) / adv_std

        # now convert advantages into positive weights in (0, +inf); shift to positive
        weights = advantages - advantages.min()
        weights = weights / (weights.mean() + self.eps)

        # optional clipping to avoid extreme weights
        clip_low = 1.0 - self.clip_ratio
        clip_high = 1.0 + self.clip_ratio
        weights = torch.clamp(weights, clip_low, clip_high)

        # final loss: weighted mean of per-sample losses
        final_loss = torch.mean(per_sample_losses * weights)
        return final_loss, weights
