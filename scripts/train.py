"""Training entry point for NExT-ST-Mamba.

This script builds the data pipeline and Lightning module and runs
training using PyTorch Lightning. By default it avoids loading heavy
SLM reasoners; pass `--load-reasoner` to enable (requires large downloads).
"""
import argparse
import os
import torch
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence

from data.dataloader import NextQADataModule
from training.lightning_module import NextSTLightning

# --- HÀM XỬ LÝ PADDING ĐỂ SỬA LỖI RESIZE STORAGE ---
def nextqa_collate_fn(batch):
    questions = [item["question"] for item in batch]
    chunks = [item["chunks"] for item in batch]
    
    # Pad temporal dimension (chiều thời gian) bằng 0 để các tensor bằng nhau
    padded_chunks = pad_sequence(chunks, batch_first=True)
    
    out = {
        "question": questions,
        "chunks": padded_chunks
    }
    
    if "label" in batch[0] and batch[0]["label"] is not None:
        labels = [item["label"] for item in batch]
        out["label"] = torch.tensor(labels, dtype=torch.long)
        
    return out

class SafeNextQADataModule(NextQADataModule):
    def train_dataloader(self):
        dl = super().train_dataloader()
        dl.collate_fn = nextqa_collate_fn
        return dl

    def val_dataloader(self):
        dl = super().val_dataloader()
        if isinstance(dl, list):
            for d in dl: d.collate_fn = nextqa_collate_fn
        else:
            dl.collate_fn = nextqa_collate_fn
        return dl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-csv", required=True)
    p.add_argument("--val-csv", required=True)
    p.add_argument("--feat-h5-train", required=True)
    p.add_argument("--feat-h5-val", required=True)
    p.add_argument("--map-vid", default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--save-path", type=str, default="checkpoints/last.ckpt")
    p.add_argument("--load-reasoner", action="store_true")
    p.add_argument("--limit-train-batches", type=int, default=None, help="Limit number of training batches (for debugging)")
    p.add_argument("--limit-val-batches", type=int, default=None, help="Limit number of validation batches (for debugging)")
    p.add_argument("--fast-dev-run", type=bool, default=False, help="Run a single batch for train/val to quickly check the pipeline")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Sử dụng module đã được vá lỗi
    dm = SafeNextQADataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        feat_h5_train=args.feat_h5_train,
        feat_h5_val=args.feat_h5_val,
        map_vid_file=args.map_vid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dm.setup()

    model = NextSTLightning(
        model=None, 
        input_feat_dim=4096, 
        hidden_dim=1024, 
        num_labels=5, 
        lr=1e-4
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs, 
        devices=args.gpus if args.gpus > 0 else None, 
        accelerator="gpu" if args.gpus > 0 else None,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        fast_dev_run = args.fast_dev_run
    )
    trainer.fit(model, datamodule=dm)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    trainer.save_checkpoint(args.save_path)
    print("Saved checkpoint to", args.save_path)


if __name__ == "__main__":
    main()