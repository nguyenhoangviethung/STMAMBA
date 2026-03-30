"""Sanity test for Phase 2 components.

Runs minimal forward passes through ShaRPPruner, STGraph and MambaLayer
to ensure shapes and basic ops function as expected.
"""
import torch
from models.components.sharp_pruner import ShaRPPruner
from models.components.st_graph import STGraph
from models.components.mamba_layer import MambaLayer


def test_sharp():
    B, T, N, D = 2, 4, 8, 64
    tokens = torch.randn(B, T * N, D)
    pruner = ShaRPPruner(dim=D, keep_ratio=0.5)
    pruned, idx, weights = pruner(tokens)
    print("ShaRP pruned shape:", pruned.shape, "idx shape:", idx.shape, "weights:", weights.shape)
    # test threshold mode
    pruner_t = ShaRPPruner(dim=D, keep_ratio=0.5, threshold=0.1)
    pruned_t, idx_t, weights_t = pruner_t(tokens)
    print("ShaRP (threshold) pruned shape:", pruned_t.shape)


def test_stgraph():
    B, T, N, D = 2, 6, 5, 64
    x = torch.randn(B, T, N, D)
    st = STGraph(dim=D, max_T=16)
    out = st(x)
    print("STGraph out shape:", out.shape)


def test_mamba():
    B, L, D = 2, 10, 64
    x = torch.randn(B, L, D)
    layer = MambaLayer(dim=D, n_heads=4, use_checkpoint=False)
    out = layer(x)
    print("MambaLayer out shape:", out.shape)


if __name__ == "__main__":
    test_sharp()
    test_stgraph()
    test_mamba()
