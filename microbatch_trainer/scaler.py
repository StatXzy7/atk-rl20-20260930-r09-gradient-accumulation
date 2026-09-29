"""训练用 loss scale 协议（AMP 风格，但用普通 float 张量模拟）。

- ``scale``：当前放大倍数。
- 累积中一旦发现非有限梯度，本次逻辑更新失败：
  不反缩放、不裁剪、不 step、不调度；随后调用 :meth:`record_overflow`
  调整 scale。
- 动态模式成功 ``growth_interval`` 次后放大一次，失败立即按
  ``backoff_factor`` 缩小。
"""
from __future__ import annotations


class LossScaler:
    def __init__(
        self,
        init_scale: float = 2.0**16,
        backoff_factor: float = 0.5,
        growth_factor: float = 2.0,
        growth_interval: int = 128,
        enabled: bool = True,
    ):
        self.scale = float(init_scale)
        self.backoff_factor = float(backoff_factor)
        self.growth_factor = float(growth_factor)
        self.growth_interval = int(growth_interval)
        self.enabled = enabled
        self._successes = 0

    def get_scale(self) -> float:
        return self.scale if self.enabled else 1.0

    def record_overflow(self) -> float:
        self._successes = 0
        self.scale = max(1.0, self.scale * self.backoff_factor)
        return self.scale

    def record_success(self) -> float:
        self._successes += 1
        if self._successes >= self.growth_interval:
            self._successes = 0
            self.scale *= self.growth_factor
        return self.scale
