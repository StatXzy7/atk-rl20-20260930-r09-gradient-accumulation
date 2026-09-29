"""线性 warmup + 余弦衰减；仅在被显式调用 step 时前进。"""
from __future__ import annotations

import math


class WarmupCosineScheduler:
    """线性 warmup + 余弦衰减。

    约定：优化器与调度器的步数一一对应；``step_count`` 等于**已完成**的
    逻辑更新数，当前更新使用 ``lr_at(step_count + 1)``。因此第一次更新
    学习率为 ``base_lr / warmup_steps``（非零），warmup 结束当步达到
    ``base_lr``，之后余弦衰减到 ``min_lr_ratio * base_lr``。
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps, total_steps)
        self.min_lr_ratio = min_lr_ratio
        self.base_lr = optimizer.lr
        self.step_count = 0
        self.set_lr()

    def lr_at(self, update_index: int) -> float:
        """``update_index`` 为 1 起的逻辑更新序号。"""
        if update_index <= self.warmup_steps:
            frac = update_index / float(self.warmup_steps)
            return self.base_lr * frac
        progress = (update_index - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        progress = min(1.0, max(0.0, progress))
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.base_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cos)

    def set_lr(self) -> None:
        # step_count 个更新已完成；下一次更新使用第 step_count + 1 个序号的 lr。
        self.optimizer.lr = self.lr_at(self.step_count + 1)

    def step(self) -> None:
        self.step_count += 1
        self.set_lr()
