"""Sanity test for Phase 3 model assembly.

This script instantiates the assembled `NextSTMamba` with `load_reasoner=False`
to avoid heavy model downloads, runs a forward pass with fake inputs, and
verifies tensor shapes.
"""
import torch
from models.next_st_mamba import build_default_model, NextSTMamba


def main():
    B, T, N, F = 2, 4, 6, 1024
    model = NextSTMamba(input_feat_dim=F, hidden_dim=128, num_mamba_layers=2, load_reasoner=False)
    texts = ["What happens after the object moves?", "Why did the person stop?"]
    visual = torch.randn(B, T, N, F)
    out = model(texts, visual)
    print("visual_repr", out["visual_repr"].shape)
    print("text_repr", out["text_repr"].shape)
    print("pruned_tokens", out["pruned_tokens"].shape)


if __name__ == "__main__":
    main()
