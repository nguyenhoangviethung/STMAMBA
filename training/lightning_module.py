"""PyTorch Lightning module wrapping NextSTMamba and GRPO training.

This LightningModule trains a small classifier head on top of the fused
visual+text representations produced by `NextSTMamba`. It uses a GRPO
surrogate (implemented in `training/grpo_trainer.py`) to reweight
per-sample supervised losses by group-relative advantages computed from
reward functions.
"""
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.next_st_mamba import NextSTMamba
from training.grpo_trainer import GRPO
from training import reward_funcs


class NextSTLightning(pl.LightningModule):
    """LightningModule implementing a GRPO-style training loop.

    Args:
        model: optional pre-built NextSTMamba model; if None a default is built.
        input_feat_dim: dimension of input visual features (added to fix shape error).
        hidden_dim: dimension of model pooled vectors (used to size classifier).
        num_labels: number of labels for classification head (if dataset uses labels).
        lr: learning rate.
    """

    def __init__(
        self, 
        model: Optional[NextSTMamba] = None, 
        input_feat_dim: int = 1024, # <--- Thêm tham số input_feat_dim vào
        hidden_dim: int = 1024, 
        num_labels: int = 5, 
        lr: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # VÁ LỖI SHAPE: Truyền input_feat_dim xuống NextSTMamba
        self.model = model if model is not None else NextSTMamba(
            input_feat_dim=input_feat_dim, # <--- Truyền tham số này
            hidden_dim=hidden_dim, 
            load_reasoner=False
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.grpo = GRPO(clip_ratio=0.2)
        self.lr = lr

    def forward(self, texts, visual_feats):
        return self.model(texts, visual_feats)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        # batch expected to contain: question, chunks (K, chunk_size, F), label (optional), answer (optional)
        texts = batch["question"] if isinstance(batch["question"], list) else [batch["question"]]

        # prepare a minimal visual_feats tensor from chunks: take first chunk per sample
        chunks = batch["chunks"]  # shape: (B, K, chunk_size, F) or (B, K, chunk_size, F)
        if chunks.dim() == 3:
            # (B, chunk_size, F)
            visual_feats = chunks.unsqueeze(2)  # (B, T, N=1, F)
        elif chunks.dim() == 4:
            # (B, K, chunk_size, F) -> collapse K into temporal dimension
            B, K, C, F = chunks.shape
            visual_feats = chunks.view(B, K * C, 1, F)
        else:
            raise ValueError("Unsupported chunks shape")

        out = self.model(texts, visual_feats)
        visual_repr = out["visual_repr"]
        text_repr = out["text_repr"]

        # classifier input: concat visual + text pooled vectors
        feat = torch.cat([visual_repr, text_repr], dim=1)
        
        # VÁ LỖI DTYPE: Đồng bộ kiểu dữ liệu của feat với trọng số của classifier
        feat = feat.to(self.classifier.weight.dtype)
        
        logits = self.classifier(feat)

        # extract label (if present) else create dummy zeros
        labels = batch.get("label", None)
        if labels is None:
            # if no label provided, create zero labels to allow loss computation
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        # ensure tensor
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long, device=logits.device)

        per_sample_losses = self.criterion(logits, labels)  # (B,)

        # compute rewards: try to use generated text vs any reference 'answer'
        rewards = torch.zeros(per_sample_losses.size(0), device=per_sample_losses.device)
        answers = batch.get("answer", None)
        # attempt generation if reasoner available and answers exist
        if answers is not None and self.model.reasoner is not None:
            prompts = []
            for q in texts:
                prompts.append(q)
            gens = self.model.generate_answer(prompts, visual_feats)
            for i, (g, ref) in enumerate(zip(gens, answers)):
                rewards[i] = float(reward_funcs.causal_reward(g, ref))
        else:
            # fallback reward: negative supervised loss (higher reward if lower loss)
            rewards = (-per_sample_losses.detach()).cpu()
            rewards = rewards.to(per_sample_losses.device)

        final_loss, weights = self.grpo.compute_surrogate(per_sample_losses, rewards)

        self.log("train/loss", final_loss, on_step=True, on_epoch=True)
        self.log("train/weights_mean", weights.mean(), on_step=True, on_epoch=True)
        return final_loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(list(self.model.parameters()) + list(self.classifier.parameters()), lr=self.lr)
        return opt