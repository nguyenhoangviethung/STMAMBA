
"""NextQA dataset implementation using HDF5-backed features.

This dataset opens the HDF5 feature file lazily (once per worker) to
avoid repeated open/close in `__getitem__`. It supports mapping between
CSV `video_id` and HDF5 keys via `map_vid_file` and creates ActionChunks
using `transforms.make_action_chunks`. Frame shuffling and chunk masking
are applied during training for anti-shortcut robustness.
"""
from typing import Optional, Dict, Any, List
import pandas as pd
import os
import json
import h5py
import numpy as np
import torch
from .transforms import make_action_chunks, frame_shuffle
import random


class NextQADataset:
    """HDF5-backed dataset for NExT-QA.

    Args:
        csv_path: path to CSV with fields including `video_id` and `question`.
        h5_path: path to HDF5 file containing per-video features.
        map_vid_file: optional JSON mapping from CSV video_id -> HDF5 key.
        chunk_size: frames per ActionChunk.
        stride: sliding stride for chunking.
        shuffle_prob: probability to shuffle frames inside chunks (train-only).
        mask_prob: probability to mask an ActionChunk (train-only).
    """

    def __init__(
        self,
        csv_path: str,
        h5_path: str,
        map_vid_file: Optional[str] = None,
        is_train: bool = True,
        chunk_size: int = 8,
        stride: int = 4,
        shuffle_prob: float = 0.1,
        mask_prob: float = 0.0,
    ) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 features not found: {h5_path}")

        self.csv_path = csv_path
        self.h5_path = h5_path
        self.df = pd.read_csv(csv_path)
        self.is_train = is_train
        self.chunk_size = int(chunk_size)
        self.stride = int(stride)
        self.shuffle_prob = float(shuffle_prob)
        self.mask_prob = float(mask_prob)

        # mapping from csv video_id to h5 key
        self.map: Optional[Dict[str, str]] = None
        if map_vid_file and os.path.exists(map_vid_file):
            with open(map_vid_file, "r", encoding="utf-8") as f:
                self.map = json.load(f)

        # h5 file handle (opened lazily per worker)
        self.h5: Optional[h5py.File] = None

    def __len__(self) -> int:
        return len(self.df)

    def _ensure_h5(self):
        """Open HDF5 file if not already opened (worker-safe lazy open)."""
        if self.h5 is None:
            # open in read-only mode
            self.h5 = h5py.File(self.h5_path, "r")

    def _get_h5_key(self, video_id: str) -> str:
        if self.map is not None:
            return self.map.get(str(video_id), str(video_id))
        return str(video_id)

    def _load_features_for_video(self, key: str) -> np.ndarray:
        self._ensure_h5()
        if key not in self.h5:
            raise KeyError(f"Video key {key} not found in HDF5 file")
        arr = self.h5[key]
        # read to memory (small per-video arrays); caller will chunk further
        data = np.array(arr)
        return data

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        question = row.get("question", "")
        video_id = row.get("video_id", "")
        choices = row.get("choices", "")
        label = row.get("label", None)

        h5_key = self._get_h5_key(video_id)
        feats = self._load_features_for_video(h5_key)  # numpy array

        # Expected shapes: (T, F) or (T, N, F)
        if feats.ndim == 2:
            # single spatial token per frame -> treat N=1
            T, F = feats.shape
            feats = feats.reshape(T, 1, F)
        elif feats.ndim == 3:
            T, N, F = feats.shape
        else:
            raise ValueError(f"Unsupported feature shape for video {h5_key}: {feats.shape}")

        # Create ActionChunks per node by flattening spatial nodes into token dimension
        # We'll produce chunks over the frame axis for each spatial node and then
        # stack them so each chunk preserves motion cues.
        # Merge N into the frame dimension to form (T*N, F)
        merged = feats.reshape(T * feats.shape[1], feats.shape[2])  # (T*N, F)

        chunks = make_action_chunks(merged, chunk_size=self.chunk_size, stride=self.stride)

        # apply frame shuffling (train only)
        if self.is_train and self.shuffle_prob > 0.0:
            chunks = frame_shuffle(chunks, shuffle_prob=self.shuffle_prob)

        # apply random chunk masking
        if self.is_train and self.mask_prob > 0.0:
            masked_chunks: List[np.ndarray] = []
            for c in chunks:
                if random.random() < self.mask_prob:
                    masked_chunks.append(np.zeros_like(c))
                else:
                    masked_chunks.append(c)
            chunks = masked_chunks

        # convert to torch tensor: list of (chunk_size, F) -> (K, chunk_size, F)
        if len(chunks) == 0:
            # fallback: return a zero chunk
            chunks = [np.zeros((self.chunk_size, feats.shape[2]), dtype=feats.dtype)]

        chunks_arr = np.stack(chunks, axis=0)
        chunks_tensor = torch.from_numpy(chunks_arr).float()

        return {
            "question": question,
            "video_id": video_id,
            "h5_key": h5_key,
            "chunks": chunks_tensor,
            "choices": choices,
            "label": label,
        }

    def __del__(self):
        try:
            if self.h5 is not None:
                self.h5.close()
        except Exception:
            pass
