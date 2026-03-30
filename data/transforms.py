"""Transforms and ActionChunk utilities (Phase 1 scaffold).

Phase 4 will implement efficient HDF5-backed feature transforms.
"""
from typing import List, Tuple
import random
import numpy as np


def make_action_chunks(features: np.ndarray, chunk_size: int, stride: int) -> List[np.ndarray]:
    """Split frame-level features into action chunks.

    Args:
        features: [T, D] frame features
        chunk_size: number of frames per chunk
        stride: sliding window stride
    Returns:
        list of chunks each shape [chunk_size, D]
    """
    chunks = []
    T = features.shape[0]
    for start in range(0, max(1, T - chunk_size + 1), stride):
        chunks.append(features[start:start + chunk_size])
    if not chunks and T > 0:
        # pad/truncate to at least one chunk
        if T >= chunk_size:
            chunks.append(features[:chunk_size])
        else:
            pad = np.zeros((chunk_size - T, features.shape[1]), dtype=features.dtype)
            chunks.append(np.concatenate([features, pad], axis=0))
    return chunks


def frame_shuffle(chunks: List[np.ndarray], shuffle_prob: float = 0.1) -> List[np.ndarray]:
    """Randomly shuffle frames inside chunks with given probability."""
    out = []
    for c in chunks:
        if random.random() < shuffle_prob:
            idx = np.arange(c.shape[0])
            np.random.shuffle(idx)
            out.append(c[idx])
        else:
            out.append(c)
    return out
