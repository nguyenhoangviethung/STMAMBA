"""Basic metrics utilities."""
from typing import Sequence


def accuracy(preds: Sequence[int], labels: Sequence[int]) -> float:
    correct = sum(p == t for p, t in zip(preds, labels))
    return float(correct) / max(1, len(labels))
