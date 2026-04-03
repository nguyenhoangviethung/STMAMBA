"""Evaluation script for NExT-ST-Mamba.

Loads a Lightning checkpoint (saved by `scripts/train.py`) and runs
evaluation on a validation split. This is a lightweight evaluator that
prints accuracy when `label` exists in the dataset.
"""
import argparse
import torch

from training.lightning_module import NextSTLightning
from data.dataloader import NextQADataModule
from utils.metrics import accuracy

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
    
    # VÁ LỖI DEVICE: Tự động nhận diện GPU nếu có để tăng tốc evaluate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")

    # Load checkpoint. Vì chúng ta đã có self.save_hyperparameters() trong file train
    # nên checkpoint này đã tự nhớ input_feat_dim=64 và text_dim=768.
    module = NextSTLightning.load_from_checkpoint(args.ckpt, map_location=device)
    module.eval()
    module.to(device)

    # Sử dụng DataModule gốc (vì ta đã vá lỗi collate_fn trực tiếp trong data/dataloader.py)
    dm = NextQADataModule(
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
        texts = batch["question"] if isinstance(batch["question"], list) else [batch["question"]]
        chunks = batch["chunks"]
        
        if chunks.dim() == 3:
            visual_feats = chunks.unsqueeze(2)
        else:
            B, K, C, F = chunks.shape
            visual_feats = chunks.view(B, K * C, 1, F)

        # Chuyển visual_feats sang cùng device với model
        visual_feats = visual_feats.to(device)

        with torch.no_grad():
            out = module.model(texts, visual_feats)
            feat = torch.cat([out["visual_repr"], out["text_repr"]], dim=1)
            
            # VÁ LỖI DTYPE: Đồng bộ kiểu dữ liệu trước khi đưa vào classifier
            feat = feat.to(module.classifier.weight.dtype)
            
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
        print("==============================")
        print(f"Validation Accuracy: {acc * 100:.2f}%")
        print("==============================")
    else:
        print("No labels found in validation data; printed predictions:")
        print(preds[:20])


if __name__ == "__main__":
    main()