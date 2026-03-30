"""Training entry point for NExT-ST-Mamba.

This script builds the data pipeline and Lightning module and runs
training using PyTorch Lightning. By default it avoids loading heavy
SLM reasoners; pass `--load-reasoner` to enable (requires large downloads).
"""
import argparse
import os
import pytorch_lightning as pl

from data.dataloader import NextQADataModule
from training.lightning_module import NextSTLightning


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
    return p.parse_args()


def main():
    args = parse_args()
    dm = NextQADataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        feat_h5_train=args.feat_h5_train,
        feat_h5_val=args.feat_h5_val,
        map_vid_file=args.map_vid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    dm.setup()

    model = NextSTLightning(model=None, hidden_dim=1024, num_labels=5, lr=1e-4)

    trainer = pl.Trainer(max_epochs=args.epochs, devices=args.gpus if args.gpus > 0 else None, accelerator="gpu" if args.gpus > 0 else None)
    trainer.fit(model, datamodule=dm)

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    trainer.save_checkpoint(args.save_path)
    print("Saved checkpoint to", args.save_path)


if __name__ == "__main__":
    main()
