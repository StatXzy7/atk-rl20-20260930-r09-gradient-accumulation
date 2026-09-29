"""从实现的最小 AdamW（解耦 weight decay）。

不使用 torch.optim.AdamW，以便：
1. 明确 ``step`` 只有一次时 weight decay、一阶/二阶矩、步数才推进；
2. 跳过/失败时完全不触碰参数与状态。
"""
from __future__ import annotations

from typing import Iterable

import torch


class AdamW:
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self, set_to_none: bool = False) -> None:
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    p.grad.detach_().zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.beta1**self.t
        bc2 = 1.0 - self.beta2**self.t
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            if g is None:
                continue
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
            m_hat = m / bc1
            v_hat = v / bc2
            if self.weight_decay != 0.0:
                p.mul_(1.0 - self.lr * self.weight_decay)
            p.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def state_dict(self) -> dict:
        return {
            "t": self.t,
            "lr": self.lr,
            "m": [m.clone() for m in self.m],
            "v": [v.clone() for v in self.v],
        }
