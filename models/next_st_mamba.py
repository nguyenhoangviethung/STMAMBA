"""Next-ST-Mamba model assembly.

This module composes the core components implemented in Phase 2:
- `TextEncoder` (BERT)
- `ShaRPPruner` (adaptive token pruning)
- `STGraph` (spatio-temporal message passing)
- `MambaLayer` stack (memory-efficient transformer blocks)
- `QwenReasoner` (SLM wrapper used for generation)

The forward pass produces fused representations and a `generate_answer`
helper converts the fused visual+text representations into prompts for
the causal LM.
"""
from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn

from .components.text_encoder import TextEncoder
from .components.sharp_pruner import ShaRPPruner
from .components.st_graph import STGraph
from .components.mamba_layer import MambaLayer
from .decoder.qwen_reasoner import QwenReasoner


class NextSTMamba(nn.Module):
    """Assembled NExT-ST-Mamba model.

    Args:
        input_feat_dim: dimensionality of input appearance/motion features
        hidden_dim: internal model dimension
        num_mamba_layers: number of transformer (Mamba) layers
        keep_ratio: token keep ratio for ShaRP
        dtype: tensor dtype (default: torch.bfloat16)
    """

    def __init__(
        self,
        input_feat_dim: int = 1024,
        hidden_dim: int = 1024,
        num_mamba_layers: int = 6,
        n_heads: int = 8,
        mlp_dim: int = 2048,
        keep_ratio: float = 0.7,
        text_model_name: str = "bert-base-uncased",
        reasoning_model_name: str = "facebook/opt-125m",
        dtype: torch.dtype = torch.bfloat16,
        use_checkpoint: bool = True,
        device: Optional[torch.device] = None,
        load_reasoner: bool = False,
    ) -> None:
        super().__init__()
        self.dtype = dtype
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Encoders / modules
        self.text_encoder = TextEncoder(text_model_name)
        self.input_proj = nn.Linear(input_feat_dim, hidden_dim)
        self.st_graph = STGraph(hidden_dim, num_heads=n_heads)
        self.pruner = ShaRPPruner(hidden_dim, keep_ratio=keep_ratio)
        self.mamba_layers = nn.ModuleList([
            MambaLayer(hidden_dim, n_heads=n_heads, mlp_dim=mlp_dim, use_checkpoint=use_checkpoint)
            for _ in range(num_mamba_layers)
        ])

        # small projection to embed visual summary into short textual tokens
        self.visual_to_text_proj = nn.Linear(hidden_dim, 32)

        # Reasoner (causal LM wrapper)
        # Optionally load the heavy SLM. Set `load_reasoner=False` in tests to
        # avoid downloading/initializing large models.
        self.reasoner: Optional[QwenReasoner]
        if load_reasoner:
            self.reasoner = QwenReasoner(reasoning_model_name, device=str(self.device))
        else:
            self.reasoner = None

        # Move modules to device
        self.to(self.device)

        # Attempt to cast parameters to requested dtype when supported
        if self.device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
            try:
                self.half() if dtype == torch.float16 else self.to(dtype)
            except Exception:
                # ignore if device/dtype not supported
                pass

    def forward(self, texts: List[str], visual_feats: torch.Tensor) -> Dict[str, Any]:
        """Forward fusion pass.

        Args:
            texts: list of B question strings
            visual_feats: tensor (B, T, N, F) where F == input_feat_dim

        Returns:
            dict with keys: `visual_repr` (B, D), `text_repr` (B, D), `pruned_tokens` (B, K, D)
        """
        # visual_feats -> project to hidden dim
        assert visual_feats.dim() == 4, "visual_feats must be (B, T, N, F)"
        B, T, N, F = visual_feats.shape
        # ensure device and dtype
        visual_feats = visual_feats.to(self.device)
        if hasattr(self, "dtype") and self.dtype is not None:
            try:
                visual_feats = visual_feats.to(self.dtype)
            except Exception:
                # some cpu devices may not support bfloat16
                pass

        x = self.input_proj(visual_feats)  # (B, T, N, D)

        # ST-Graph message passing
        x = self.st_graph(x)  # (B, T, N, D)

        # flatten temporal+spatial into token sequence
        tokens = x.view(B, T * N, -1)  # (B, L, D)

        # ShaRP pruning -> (B, K, D), weights
        pruned_tokens, keep_idx, weights = self.pruner(tokens)

        # apply soft weighting to pruned tokens
        # weights: (B, K) -> expand to (B, K, D)
        h = pruned_tokens * weights.unsqueeze(-1)
        for layer in self.mamba_layers:
            h = layer(h)

        # pooled visual representation
        visual_repr = h.mean(dim=1)  # (B, D)

        # text encoding
        last_hidden, text_pooled = self.text_encoder(texts)
        # ensure same device/dtype
        text_pooled = text_pooled.to(visual_repr.dtype).to(visual_repr.device)

        return {"visual_repr": visual_repr, "text_repr": text_pooled, "pruned_tokens": h, "keep_idx": keep_idx, "pruner_weights": weights}

    def generate_answer(self, texts: List[str], visual_feats: torch.Tensor, max_new_tokens: int = 64) -> List[str]:
        """Create a succinct prompt from fused representations and call the SLM.

        This method projects the visual representation into a short numeric
        summary and appends it to the textual question to provide context
        to the causal LM. This keeps the bridge compact and reproducible.
        """
        if self.reasoner is None:
            raise RuntimeError("Reasoner not loaded. Initialize NextSTMamba with load_reasoner=True to enable generation.")

        self.eval()
        with torch.no_grad():
            out = self.forward(texts, visual_feats)
            visual_repr = out["visual_repr"]  # (B, D)
            # small numeric summary
            vis_summary = self.visual_to_text_proj(visual_repr)  # (B, 32)
            vis_summary = vis_summary.to("cpu").numpy()

            prompts: List[str] = []
            for q, vs in zip(texts, vis_summary):
                nums = ",".join([f"{float(x):.4f}" for x in vs.tolist()])
                prompt = f"Question: {q}\nVisualSummary: {nums}\nAnswer:"
                prompts.append(prompt)

            # call the causal reasoner to generate answers
            answers = self.reasoner.generate(prompts, max_new_tokens=max_new_tokens)
            return answers

    def save_checkpoint(self, path: str) -> None:
        """Save model state dict to `path` (CPU-friendly)."""
        sd = {k: v.cpu() for k, v in self.state_dict().items()}
        torch.save(sd, path)

    def load_checkpoint(self, path: str, map_location: Optional[str] = None) -> None:
        """Load state dict from `path`.

        Args:
            path: file path to load
            map_location: optional map_location for torch.load
        """
        map_loc = map_location or ("cpu" if not torch.cuda.is_available() else None)
        sd = torch.load(path, map_location=map_loc)
        self.load_state_dict(sd)


def build_default_model() -> NextSTMamba:
    """Helper to instantiate a default NextSTMamba for experiments."""
    return NextSTMamba()
