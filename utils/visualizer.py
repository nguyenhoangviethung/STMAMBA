"""Minimal visualization helpers."""
from typing import Any
import matplotlib.pyplot as plt
import numpy as np


def plot_action_chunk(chunk: Any, title: str = "ActionChunk") -> None:
    """Plot a single ActionChunk (frames x features) as an image.

    `chunk` can be a numpy array or torch tensor of shape (chunk_size, F).
    """
    if hasattr(chunk, "cpu"):
        chunk = chunk.cpu().numpy()
    arr = np.array(chunk)
    plt.imshow(arr.T, aspect="auto", cmap="viridis")
    plt.title(title)
    plt.xlabel("frame")
    plt.ylabel("feature")
    plt.colorbar()
    plt.show()
