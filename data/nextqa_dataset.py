"""NextQA dataset implementation using HDF5-backed features.

This dataset opens the HDF5 feature file lazily inside __getitem__ to
avoid multiprocessing issues with h5py objects. It supports mapping between
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
import logging

logger = logging.getLogger(__name__)

class NextQADataset(torch.utils.data.Dataset):
    """HDF5-backed dataset for NExT-QA."""

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
        self.missing_keys_logged = 0

        # VÁ LỖI CẤU TRÚC HDF5: Đọc mảng 'ids' một lần vào RAM để làm từ điển tra cứu (Tra cứu index)
        self.vid_to_idx = {}
        self._build_vid_index()

    def _build_vid_index(self):
        try:
            with h5py.File(self.h5_path, "r") as f:
                if "ids" in f:
                    ids_array = np.array(f["ids"])
                    for idx, vid in enumerate(ids_array):
                        self.vid_to_idx[str(vid)] = idx
        except Exception as e:
            logger.warning(f"Could not build video index from {self.h5_path}: {e}")

    def __len__(self) -> int:
        return len(self.df)

    def _load_features_for_video(self, video_id: str) -> Optional[np.ndarray]:
        # Kiểm tra xem video_id có trong từ điển tra cứu không
        if video_id not in self.vid_to_idx:
            if self.missing_keys_logged < 5:
                print(f"WARNING: Video ID '{video_id}' not found in HDF5 'ids' array!")
                self.missing_keys_logged += 1
            return None
            
        idx = self.vid_to_idx[video_id]
        try:
            with h5py.File(self.h5_path, "r") as h5_file:
                # Trích xuất dòng feature tương ứng từ mảng 'feat'
                data = np.array(h5_file["feat"][idx])
            return data
        except Exception as e:
            if self.missing_keys_logged < 5:
                print(f"ERROR reading HDF5 for video '{video_id}': {e}")
                self.missing_keys_logged += 1
            return None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        question = row.get("question", "")
        
        video_id = str(row.get("video", "")).strip()
        if video_id.lower() == "nan":
            video_id = ""

        choices = [row.get(f"a{i}", "") for i in range(5)]
        label = row.get("answer", None)

        feats = self._load_features_for_video(video_id)

        # Fallback khi không tìm thấy feature
        if feats is None:
            if self.is_train:
                return self.__getitem__(random.randint(0, len(self) - 1))
            else:
                # Dựa vào log, feature dimension là 4096
                feats = np.zeros((self.chunk_size, 1, 4096), dtype=np.float32)

        if feats.ndim == 2:
            T, F = feats.shape
            feats = feats.reshape(T, 1, F)
        elif feats.ndim == 3:
            T, N, F = feats.shape
        else:
            raise ValueError(f"Unsupported feature shape for video {video_id}: {feats.shape}")

        merged = feats.reshape(T * feats.shape[1], feats.shape[2])

        chunks = make_action_chunks(merged, chunk_size=self.chunk_size, stride=self.stride)

        if self.is_train and self.shuffle_prob > 0.0:
            chunks = frame_shuffle(chunks, shuffle_prob=self.shuffle_prob)

        if self.is_train and self.mask_prob > 0.0:
            masked_chunks: List[np.ndarray] = []
            for c in chunks:
                if random.random() < self.mask_prob:
                    masked_chunks.append(np.zeros_like(c))
                else:
                    masked_chunks.append(c)
            chunks = masked_chunks

        if len(chunks) == 0:
            chunks = [np.zeros((self.chunk_size, feats.shape[2]), dtype=feats.dtype)]

        chunks_arr = np.stack(chunks, axis=0)
        chunks_tensor = torch.from_numpy(chunks_arr).float()

        return {
            "question": question,
            "video_id": video_id,
            "chunks": chunks_tensor,
            "choices": choices,
            "label": label,
        }