"""小型逐 token 语言模型与数据结构。

刻意只使用 Linear / Embedding / GELU：无 dropout、无 BatchNorm，
因此不存在批间统计差异，且不开启任何随机源。

loss 对有效 token（mask == 1）做**求和**，不做平均；归一化由训练步
按整个逻辑 batch 的有效 token 总数统一完成。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Batch:
    """token id 形状 [N, S]，mask 形状 [N, S]，1 表示有效 token。"""

    token_ids: torch.Tensor
    mask: torch.Tensor
    labels: torch.Tensor

    @property
    def num_valid_tokens(self) -> int:
        return int(self.mask.sum().item())

    def slice(self, start: int, end: int) -> "Batch":
        return Batch(
            token_ids=self.token_ids[start:end],
            mask=self.mask[start:end],
            labels=self.labels[start:end],
        )


class TinyTokenModel(nn.Module):
    """embedding -> MLP -> 逐位置 logits。

    对位置不做任何池化，每条序列的每个有效位置都贡献一个交叉熵项。
    """

    def __init__(self, vocab_size: int = 37, hidden_dim: int = 24, seed: int = 0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(
                    torch.randn(
                        p.shape,
                        generator=gen,
                        dtype=p.dtype,
                    )
                    * 0.2
                )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(token_ids)
        h = F.gelu(self.fc1(h))
        h = F.gelu(self.fc2(h))
        return self.head(h)

    def summed_loss(
        self,
        token_ids: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        overflow_bias: float = 0.0,
    ) -> tuple[torch.Tensor, int]:
        """返回 (有效 token 损失之和, 有效 token 数)。

        忽略全部 pad 位置；对有效 token 不归一化。``overflow_bias`` 仅用于
        测试中确定性注入溢出（加到每个有效 token 的 logits 上）。
        """
        logits = self(token_ids) + overflow_bias
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        mask_f = mask.to(ce.dtype)
        n_valid = int(mask_f.sum().item())
        return (ce * mask_f).sum(), n_valid
