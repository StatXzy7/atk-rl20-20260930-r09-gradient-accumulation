"""任意 microbatch 切分的等价梯度累积训练步。

公共 API 见 :mod:`microbatch_trainer.trainer` 中的 :func:`logical_train_step`
以及 :func:`naive_mean_microbatch_train_step`（反例）。
"""
from .trainer import (
    StepResult,
    logical_train_step,
    naive_mean_microbatch_train_step,
    per_microbatch_update_train_step,
)
from .model import TinyTokenModel, Batch
from .optimizer import AdamW
from .scheduler import WarmupCosineScheduler
from .scaler import LossScaler

__all__ = [
    "StepResult",
    "logical_train_step",
    "naive_mean_microbatch_train_step",
    "per_microbatch_update_train_step",
    "TinyTokenModel",
    "Batch",
    "AdamW",
    "WarmupCosineScheduler",
    "LossScaler",
]
