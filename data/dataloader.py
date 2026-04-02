# data/dataloader.py
"""PyTorch Lightning DataModule scaffold for NExT-QA."""
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch

from .nextqa_dataset import NextQADataset

def nextqa_collate_fn(batch):
    """
    Collate function to pad sequence of chunks so they have the same size.
    Handles the case where different videos yield different numbers of ActionChunks (K).
    """
    questions = [item["question"] for item in batch]
    video_ids = [item["video_id"] for item in batch]
    h5_keys = [item["h5_key"] for item in batch]
    choices = [item["choices"] for item in batch]
    
    # Extract the chunks and pad them
    chunks_list = [item["chunks"] for item in batch]
    # chunks is a list of tensors of shape (K, chunk_size, F). K is variable.
    # pad_sequence expects sequences of shape (L, *). In our case, L is K.
    padded_chunks = pad_sequence(chunks_list, batch_first=True) # Output shape: (B, max_K, chunk_size, F)

    out = {
        "question": questions,
        "video_id": video_ids,
        "h5_key": h5_keys,
        "chunks": padded_chunks,
        "choices": choices
    }
    
    # Process labels if they exist
    if "label" in batch[0] and batch[0]["label"] is not None:
        labels = [item["label"] for item in batch]
        out["label"] = torch.tensor(labels, dtype=torch.long)
        
    return out


class NextQADataModule(pl.LightningDataModule):
    """DataModule that constructs NextQADataset with HDF5 feature access.

    Args:
        train_csv: path to training CSV
        val_csv: path to validation CSV
        feat_h5_train: HDF5 file path for training features
        feat_h5_val: HDF5 file path for validation features
        map_vid_file: optional mapping JSON for video IDs
        batch_size: batch size
        num_workers: DataLoader workers
        chunk_size/stride/shuffle_prob/mask_prob: forwarded to dataset transforms
    """

    def __init__(
        self,
        train_csv: str,
        val_csv: str,
        feat_h5_train: str,
        feat_h5_val: str,
        map_vid_file: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 4,
        chunk_size: int = 8,
        stride: int = 4,
        shuffle_prob: float = 0.1,
        mask_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.feat_h5_train = feat_h5_train
        self.feat_h5_val = feat_h5_val
        self.map_vid_file = map_vid_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.stride = stride
        self.shuffle_prob = shuffle_prob
        self.mask_prob = mask_prob

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = NextQADataset(
            self.train_csv,
            self.feat_h5_train,
            map_vid_file=self.map_vid_file,
            is_train=True,
            chunk_size=self.chunk_size,
            stride=self.stride,
            shuffle_prob=self.shuffle_prob,
            mask_prob=self.mask_prob,
        )
        self.val_dataset = NextQADataset(
            self.val_csv,
            self.feat_h5_val,
            map_vid_file=self.map_vid_file,
            is_train=False,
            chunk_size=self.chunk_size,
            stride=self.stride,
            shuffle_prob=0.0,
            mask_prob=0.0,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            collate_fn=nextqa_collate_fn # VÁ LỖI: Thêm collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            collate_fn=nextqa_collate_fn # VÁ LỖI: Thêm collate_fn
        )