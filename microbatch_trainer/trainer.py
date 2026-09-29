"""核心：一个逻辑 batch 的训练步，支持任意 microbatch 切分。

数学目标（见 ``DERIVATION.md``）::

    目标 = (1 / T) * 对逻辑 batch 内全部有效 token 的损失之和
    梯度 = (1 / T) * sum_j grad(l_j)

其中 T 是**整个逻辑 batch** 的有效 token 总数。每个 microbatch 前向时
对其有效 token 损失只做**求和**，然后统一乘 ``scale / T`` 反传。
切分方式只改变浮点加法的次序，不改变数学目标。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .model import Batch, TinyTokenModel
from .optimizer import AdamW
from .scheduler import WarmupCosineScheduler
from .scaler import LossScaler

STATUS_UPDATED = "updated"
STATUS_SKIPPED = "skipped"
STATUS_OVERFLOW = "overflow"


@dataclass
class StepResult:
    status: str
    n_valid_tokens: int
    n_microbatches: int
    scale: float
    grad_norm: Optional[float] = None
    clipped: bool = False
    clip_coef: Optional[float] = None


def _grads_finite(model: TinyTokenModel) -> bool:
    for p in model.parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all().item():
            return False
    return True


def _clear_grads(optimizer: AdamW) -> None:
    optimizer.zero_grad(set_to_none=True)


@dataclass
class _AccumResult:
    status: str  # "ok" | "skipped" | "overflow"
    n_valid_tokens: int
    n_microbatches: int
    scale: float
    grad_norm: Optional[float] = None
    clipped: bool = False
    clip_coef: Optional[float] = None


def accumulate_gradients(
    model: TinyTokenModel,
    optimizer: AdamW,
    scaler: LossScaler,
    batch: Batch,
    microbatch_size: int,
    max_norm: Optional[float] = None,
    overflow_in_microbatch: Optional[int] = None,
) -> _AccumResult:
    """完成清零、分 microbatch 累积、反缩放、全局裁剪。成功后梯度留在
    ``param.grad`` 上（已归一化、已裁剪），尚未更新参数。

    失败/跳过时保证 ``param.grad`` 全部为 None，且除 scaler 外无任何状态变化。
    """
    if microbatch_size <= 0:
        raise ValueError("microbatch_size must be positive")

    _clear_grads(optimizer)
    n_rows = batch.token_ids.shape[0]
    total_valid = batch.num_valid_tokens

    # 没有有效 token：定义为跳过更新（不反传、不裁剪、不 step、无 weight decay）。
    if total_valid == 0:
        return _AccumResult(STATUS_SKIPPED, 0, 0, scaler.get_scale())

    scale = scaler.get_scale()
    factor = scale / float(total_valid)

    bounds = list(range(0, n_rows, microbatch_size))
    n_microbatches = len(bounds)
    hook_handle = None
    overflow = False

    try:
        for mb_idx, start in enumerate(bounds):
            end = min(start + microbatch_size, n_rows)
            mb = batch.slice(start, end)
            n_mb = mb.num_valid_tokens
            if n_mb == 0:
                # 空 microbatch：不贡献梯度，也不改变 T。
                continue

            if overflow_in_microbatch is not None and mb_idx == overflow_in_microbatch:
                # 模拟缩放后反传产生的 inf/nan 梯度（损失本身仍可能是有限的）。
                hook_handle = model.fc1.weight.register_hook(
                    lambda grad: grad * float("inf")
                )

            loss_sum, n_check = model.summed_loss(mb.token_ids, mb.labels, mb.mask)
            assert n_check == n_mb
            (loss_sum * factor).backward()

            if hook_handle is not None:
                hook_handle.remove()
                hook_handle = None

            # 累积中一旦发现非有限梯度：立即终止，本次逻辑更新整体失败。
            if not _grads_finite(model):
                overflow = True
                break
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    def _fail() -> _AccumResult:
        # 回滚边界：丢弃全部累积梯度（含之前 microbatch 的有效部分），
        # 不反缩放、不裁剪、不 step、不调度，只调整 scale。
        _clear_grads(optimizer)
        new_scale = scaler.record_overflow()
        return _AccumResult(
            STATUS_OVERFLOW, total_valid, n_microbatches, new_scale
        )

    if overflow:
        return _fail()

    # 完整累积后统一反缩放，并再次核验有限性。
    for p in model.parameters():
        if p.grad is not None:
            p.grad.div_(scale)
            if not torch.isfinite(p.grad).all().item():
                return _fail()

    # 全局 norm（全部参数一起），在正确归一化之后、裁剪之前。
    grad_norm_sq = torch.zeros((), device=batch.token_ids.device)
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sq.add_(p.grad.detach().pow(2).sum())
    grad_norm = float(grad_norm_sq.sqrt().item())

    clipped = False
    clip_coef = None
    if max_norm is not None and grad_norm > max_norm:
        clip_coef = float(max_norm / (grad_norm + 1e-12))
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip_coef)
        clipped = True

    return _AccumResult(
        STATUS_UPDATED,
        total_valid,
        n_microbatches,
        scale,
        grad_norm=grad_norm,
        clipped=clipped,
        clip_coef=clip_coef,
    )


def logical_train_step(
    model: TinyTokenModel,
    optimizer: AdamW,
    scheduler: Optional[WarmupCosineScheduler],
    scaler: LossScaler,
    batch: Batch,
    microbatch_size: int,
    max_norm: Optional[float] = None,
    overflow_in_microbatch: Optional[int] = None,
) -> StepResult:
    """执行一次（或跳过/失败一次）逻辑更新。

    一次成功的逻辑更新恰好对应：优化器一步（含解耦 weight decay）、
    调度器一步、scaler 记录一次成功。
    """
    acc = accumulate_gradients(
        model,
        optimizer,
        scaler,
        batch,
        microbatch_size,
        max_norm=max_norm,
        overflow_in_microbatch=overflow_in_microbatch,
    )

    if acc.status != STATUS_UPDATED:
        return StepResult(
            acc.status,
            acc.n_valid_tokens,
            acc.n_microbatches,
            acc.scale,
        )

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    scaler.record_success()
    _clear_grads(optimizer)

    return StepResult(
        STATUS_UPDATED,
        acc.n_valid_tokens,
        acc.n_microbatches,
        scaler.get_scale(),
        grad_norm=acc.grad_norm,
        clipped=acc.clipped,
        clip_coef=acc.clip_coef,
    )


def naive_mean_microbatch_train_step(
    model: TinyTokenModel,
    optimizer: AdamW,
    scheduler: Optional[WarmupCosineScheduler],
    batch: Batch,
    microbatch_size: int,
    max_norm: Optional[float] = None,
) -> StepResult:
    """反例：对每个 microbatch 的均值损失再等权平均。

    累积的是 ``(1/K) * Σ_i (1/n_i) Σ_{j in i} grad(l_j)``，
    与正确目标 ``(1/T) Σ_j grad(l_j)`` 一般不相等；
    只有在所有 n_i 相等（且无空组）时才碰巧相等。
    此处保留“一次优化器 step”，仅用于隔离证明缩放公式本身的错误。
    """
    _clear_grads(optimizer)
    n_rows = batch.token_ids.shape[0]
    bounds = list(range(0, n_rows, microbatch_size))
    K = len(bounds)
    per_group_weight = 1.0 / K
    total_valid = batch.num_valid_tokens
    if total_valid == 0:
        return StepResult(STATUS_SKIPPED, 0, 0, 1.0)

    for start in bounds:
        end = min(start + microbatch_size, n_rows)
        mb = batch.slice(start, end)
        n_mb = mb.num_valid_tokens
        if n_mb == 0:
            continue
        loss_sum, _ = model.summed_loss(mb.token_ids, mb.labels, mb.mask)
        (loss_sum * (per_group_weight / n_mb)).backward()

    grad_norm_sq = torch.zeros((), device=batch.token_ids.device)
    for p in model.parameters():
        if p.grad is not None:
            grad_norm_sq.add_(p.grad.detach().pow(2).sum())
    grad_norm = float(grad_norm_sq.sqrt().item())
    if max_norm is not None and grad_norm > max_norm:
        coef = max_norm / (grad_norm + 1e-12)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(coef)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    _clear_grads(optimizer)
    return StepResult(STATUS_UPDATED, total_valid, K, 1.0, grad_norm=grad_norm)


def per_microbatch_update_train_step(
    model: TinyTokenModel,
    optimizer: AdamW,
    scheduler: Optional[WarmupCosineScheduler],
    scaler: LossScaler,
    batch: Batch,
    microbatch_size: int,
    max_norm: Optional[float] = None,
) -> list[StepResult]:
    """反例：每个 microbatch 各自做一次参数/优化器/调度器更新。

    即使每个 microbatch 内部损失按其有效 token 数归一化，多步非线性更新
    （Adam 矩、逐变化的学习率、weight decay 复利）也与一次全批次更新不相等。
    空 microbatch 不产生更新。
    """
    results: list[StepResult] = []
    n_rows = batch.token_ids.shape[0]
    for start in range(0, n_rows, microbatch_size):
        end = min(start + microbatch_size, n_rows)
        mb = batch.slice(start, end)
        if mb.num_valid_tokens == 0:
            results.append(StepResult(STATUS_SKIPPED, 0, 1, 1.0))
            continue

        # 把该 microbatch 当成自己的“逻辑 batch”单独完成一次更新
        # （注意：这会让传入的 scaler 也按 microbatch 推进——这本身也是不等价来源之一）。
        res = logical_train_step(
            model,
            optimizer,
            scheduler,
            scaler,
            mb,
            max(1, mb.token_ids.shape[0]),
            max_norm=max_norm,
        )
        results.append(res)
    return results
