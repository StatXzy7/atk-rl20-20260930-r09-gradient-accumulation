"""Tiny deterministic model + synthetic token-level data.

No dropout and no batch statistics (BatchNorm/LayerNorm), so a logical
batch and any of its microbatch partitions compute exactly the same
function of the same tokens -- the only permitted source of divergence
between partitions is floating-point rounding.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

D_IN = 8
D_HIDDEN = 16
N_CLASSES = 5
N_SEQ = 10
N_TOK = 6


class TinyTokenClassifier(nn.Module):
    """Two-layer MLP applied independently to every token."""

    def __init__(self, d_in: int = D_IN, d_hidden: int = D_HIDDEN,
                 n_classes: int = N_CLASSES) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_hidden)
        self.out_proj = nn.Linear(d_hidden, n_classes)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> logits: (B, T, C)
        return self.out_proj(self.act(self.in_proj(x)))


def make_model(seed: int = 0, dtype: torch.dtype = torch.float64
               ) -> TinyTokenClassifier:
    torch.manual_seed(seed)
    model = TinyTokenClassifier().to(dtype)
    return model


def token_cross_entropy_sum(model: nn.Module, microbatch: Dict[str, torch.Tensor]):
    """Sum of per-token cross-entropies over valid tokens, and their count.

    This is the loss_sum_fn expected by MicrobatchTrainer: it must NOT
    divide by the microbatch token count.
    """
    x, y, mask = microbatch["inputs"], microbatch["targets"], microbatch["mask"]
    n = int(mask.sum().item())
    if n == 0:
        return torch.zeros((), dtype=x.dtype), 0
    logits = model(x)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="none"
    )
    return (losses * mask.reshape(-1).to(losses.dtype)).sum(), n


def make_logical_batch(seed: int = 123, dtype: torch.dtype = torch.float64,
                       n_seq: int = N_SEQ, n_tok: int = N_TOK
                       ) -> Dict[str, torch.Tensor]:
    """One fixed logical batch of token sequences with a valid-token mask.

    Sequences 2, 3 and 4 are fully masked out so that some partitions
    contain microbatches with zero valid tokens.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n_seq, n_tok, D_IN, generator=g).to(dtype)
    y = torch.randint(0, N_CLASSES, (n_seq, n_tok), generator=g)
    mask = torch.rand(n_seq, n_tok, generator=g) > 0.25
    mask[2] = False
    mask[3] = False
    mask[4] = False
    return {"inputs": x, "targets": y, "mask": mask}


def split_batch(batch: Dict[str, torch.Tensor],
                sizes: List[int]) -> List[Dict[str, torch.Tensor]]:
    """Split a logical batch along the sequence dimension into microbatches."""
    assert sum(sizes) == batch["inputs"].shape[0], (sizes, batch["inputs"].shape)
    out, start = [], 0
    for s in sizes:
        out.append({k: v[start:start + s] for k, v in batch.items()})
        start += s
    return out
