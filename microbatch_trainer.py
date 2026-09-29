"""Gradient accumulation over arbitrary microbatch partitions.

The mathematical objective of one logical batch is

    L(theta) = (1/N) * sum_{i in valid tokens} l_i(theta),
    N = total number of valid tokens in the logical batch.

A logical batch may be split into microbatches of any size (including
empty ones, and ones whose valid-token count is zero).  This module
implements the accumulation / loss-scaling / rollback protocol that makes
every such partition equivalent, up to floating-point rounding, to a
single full-batch update.  See README.md for the derivation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Tuple

import torch
import torch.nn as nn

UPDATED = "updated"
OVERFLOW = "overflow"
SKIPPED_NO_TOKENS = "skipped_no_tokens"

# loss_sum_fn(model, microbatch) -> (sum of per-token losses as a 0-dim
# tensor, number of valid tokens as int).  The sum must NOT be divided by
# the microbatch token count; normalisation happens exactly once, after
# accumulation, with the logical-batch token count N.
LossSumFn = Callable[[nn.Module, Any], Tuple[torch.Tensor, int]]


@dataclass
class UpdateResult:
    status: str
    total_tokens: int
    microbatches_seen: int
    grad_norm: Optional[float]
    scale: float
    lr: float


class MicrobatchTrainer:
    """Atomic logical-batch updates built from arbitrary microbatch splits.

    Protocol per logical update:
      1. zero all grads; for every microbatch, backward on
         ``scale * (sum of per-token losses)`` and accumulate into .grad.
      2. After every backward, verify every accumulated grad is finite.
         Non-finite -> the whole logical update fails: clear all grads,
         decay the scale, and mutate nothing else (params, Adam moments,
         optimizer step count and scheduler stay untouched).
      3. If the logical batch contained zero valid tokens, skip the update
         entirely (no optimizer.step(), hence no decoupled weight decay,
         no scheduler step).
      4. Otherwise multiply accumulated grads by 1/(scale*N) exactly once,
         clip the global grad norm once, then call optimizer.step() and
         scheduler.step() exactly once.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        max_grad_norm: Optional[float] = None,
        init_scale: float = 1.0,
        scale_backoff: float = 0.5,
        min_scale: float = 1e-12,
        growth_interval: int = 0,
        growth_factor: float = 2.0,
    ) -> None:
        if not (0.0 < scale_backoff < 1.0):
            raise ValueError("scale_backoff must be in (0, 1)")
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.scale = float(init_scale)
        self.scale_backoff = float(scale_backoff)
        self.min_scale = float(min_scale)
        self.growth_interval = int(growth_interval)
        self.growth_factor = float(growth_factor)
        self._consecutive_successes = 0

    # ------------------------------------------------------------------ API

    def logical_update(
        self, microbatches: Iterable[Any], loss_sum_fn: LossSumFn
    ) -> UpdateResult:
        """Run one atomic logical update over an arbitrary microbatch split."""
        status, total_tokens, seen = self.prepare_grads(microbatches, loss_sum_fn)

        if status == OVERFLOW:
            self.scale = max(self.scale * self.scale_backoff, self.min_scale)
            self._consecutive_successes = 0
            return UpdateResult(OVERFLOW, total_tokens, seen, None, self.scale,
                                self._current_lr())

        if status == SKIPPED_NO_TOKENS:
            return UpdateResult(SKIPPED_NO_TOKENS, 0, seen, None, self.scale,
                                self._current_lr())

        grad_norm = self._global_norm()
        if self.max_grad_norm is not None:
            clip_coef = self.max_grad_norm / (grad_norm + 1e-12)
            if clip_coef < 1.0:
                with torch.no_grad():
                    for p in self._params():
                        if p.grad is not None:
                            p.grad.mul_(clip_coef)

        # The single point where params / Adam moments / step count advance.
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()

        self._consecutive_successes += 1
        if self.growth_interval and \
                self._consecutive_successes >= self.growth_interval:
            self.scale = min(self.scale * self.growth_factor, 1.0)
            self._consecutive_successes = 0

        return UpdateResult(UPDATED, total_tokens, seen, grad_norm, self.scale,
                            self._current_lr())

    def prepare_grads(
        self, microbatches: Iterable[Any], loss_sum_fn: LossSumFn
    ) -> Tuple[str, int, int]:
        """Accumulate and normalise grads without stepping the optimizer.

        On success, .grad holds the gradient of the logical-batch objective
        (unclipped).  On overflow or when the logical batch has no valid
        tokens, all grads are cleared and nothing else is mutated.
        Returns (status, total_valid_tokens, microbatches_seen).
        """
        self.optimizer.zero_grad(set_to_none=True)
        total_tokens = 0
        seen = 0
        for mb in microbatches:
            loss_sum, n_valid = loss_sum_fn(self.model, mb)
            seen += 1
            n_valid = int(n_valid)
            total_tokens += n_valid
            if n_valid == 0:
                # An all-masked microbatch contributes exactly zero gradient;
                # skipping its backward is mathematically identical.
                continue
            (loss_sum * self.scale).backward()
            if not self._grads_finite():
                self.optimizer.zero_grad(set_to_none=True)
                return OVERFLOW, total_tokens, seen

        if total_tokens == 0:
            self.optimizer.zero_grad(set_to_none=True)
            return SKIPPED_NO_TOKENS, 0, seen

        inv = 1.0 / (self.scale * total_tokens)
        with torch.no_grad():
            for p in self._params():
                if p.grad is not None:
                    p.grad.mul_(inv)
        return UPDATED, total_tokens, seen

    # ------------------------------------------------------------- internal

    def _params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _grads_finite(self) -> bool:
        for p in self._params():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                return False
        return True

    def _global_norm(self) -> float:
        # Computed in float64 over parameters in fixed order, so the norm
        # (and hence clipping) is independent of the microbatch partition
        # up to the rounding of the accumulated grads themselves.
        total = 0.0
        for p in self._params():
            if p.grad is not None:
                total += float(p.grad.detach().double().pow(2).sum())
        return math.sqrt(total)

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])
