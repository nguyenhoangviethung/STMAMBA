"""Evaluation script for NExT-ST-Mamba.

Loads a Lightning checkpoint (saved by `scripts/train.py`) and runs
evaluation on a validation split. This is a lightweight evaluator that
prints accuracy when `label` exists in the dataset.
"""
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence

from training.lightning_module import NextSTLightning
from data.dataloader import NextQADataModule
from utils.metrics import accuracy

# --- HÀM XỬ LÝ PADDING ĐỂ SỬA LỖI RESIZE STORAGE ---
def nextqa_collate_fn(batch):
    questions = [item["question"] for item in batch]
    chunks = [item["chunks"] for item in batch]
    
    padded_chunks = pad_sequence(chunks, batch_first=True)
    
    out = {
        "question": questions,
        "chunks": padded_chunks
    }
    
    if "label" in batch[0] and batch[0]["label"] is not None:
        labels = [item["label"] for item in batch]
        out["label"] = torch.tensor(labels, dtype=torch.long)
        
    return out

# --- BỌC DATAMODULE LẠI ĐỂ INJECT HÀM COLLATE ---
class SafeNextQADataModule(NextQADataModule):
    def val_dataloader(self):
        dl = super().val_dataloader()
        if isinstance(dl, list):
            for d in dl: d.collate_fn = nextqa_collate_fn
        else:
            dl.collate_fn = nextqa_collate_fn
        return dl


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--val-csv", type=str, required=True)
    p.add_argument("--feat-h5-val", type=str, required=True)
    p.add_argument("--map-vid", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    module = NextSTLightning.load_from_checkpoint(args.ckpt, map_location=torch.device("cpu"))
    module.eval()

    # Sử dụng module đã được vá lỗi
    dm = SafeNextQADataModule(
        train_csv="", 
        val_csv=args.val_csv, 
        feat_h5_train="", 
        feat_h5_val=args.feat_h5_val, 
        map_vid_file=args.map_vid, 
        batch_size=args.batch_size
    )
    dm.setup()
    dl = dm.val_dataloader()

    preds = []
    labels = []
    for batch in dl:
        # Lấy trực tiếp từ logic gộp batch mới
        texts = batch["question"]
        chunks = batch["chunks"]
        
        if chunks.dim() == 3:
            visual_feats = chunks.unsqueeze(2)
        else:
            B, K, C, F = chunks.shape
            visual_feats = chunks.view(B, K * C, 1, F)

        with torch.no_grad():
            out = module.model(texts, visual_feats)
            feat = torch.cat([out["visual_repr"], out["text_repr"]], dim=1)
            logits = module.classifier(feat)
            pred = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            
        lbl = batch.get("label", None)
        if lbl is not None:
            if torch.is_tensor(lbl):
                lbl = lbl.cpu().numpy().tolist()
            labels.extend(lbl)
        preds.extend(pred)

    if labels:
        acc = accuracy(preds, labels)
        print(f"Validation accuracy: {acc:.4f}")
    else:
        print("No labels found in validation data; printed predictions:")
        print(preds[:20])


if __name__ == "__main__":
    main()