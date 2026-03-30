"""Reward functions for GRPO training.

Provides causal and temporal reward functions used to compute scalar
rewards for generated answers. Implementations are lightweight and
interpretable; you can replace them with more advanced metrics later.
"""
from typing import Optional
import math


def causal_reward(generated: str, reference: str) -> float:
    """Simple causal reward based on token overlap (Jaccard).

    Returns a float in [0,1] where 1.0 indicates perfect overlap.
    """
    if not generated or not reference:
        return 0.0
    gen_tokens = set(generated.lower().split())
    ref_tokens = set(reference.lower().split())
    if not ref_tokens:
        return 0.0
    inter = gen_tokens.intersection(ref_tokens)
    union = gen_tokens.union(ref_tokens)
    return float(len(inter)) / max(1, len(union))


def temporal_accuracy_reward(pred_focus: Optional[int], target_focus: Optional[int], tolerance: int = 2) -> float:
    """Reward how close a predicted focus frame is to the target frame.

    If either value is None, returns 0.0. Otherwise returns a value in
    (0,1] that decays with distance; within `tolerance` frames returns 1.0.
    """
    if pred_focus is None or target_focus is None:
        return 0.0
    d = abs(int(pred_focus) - int(target_focus))
    if d <= tolerance:
        return 1.0
    # decay with distance
    return math.exp(-0.25 * (d - tolerance))
