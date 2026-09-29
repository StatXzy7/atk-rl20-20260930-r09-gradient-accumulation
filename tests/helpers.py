"""Shared utilities for the microbatch-equivalence test suite."""

from __future__ import annotations

import copy
from typing import Dict

import torch

from microbatch_trainer import MicrobatchTrainer
from tiny_model import make_model


def make_trainer(seed: int = 0, dtype: torch.dtype = torch.float64,
                 max_grad_norm=None, lr: float = 0.05,
                 weight_decay: float = 0.1, **trainer_kw) -> MicrobatchTrainer:
    model = make_model(seed=seed, dtype=dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: 1.0 / (1.0 + 0.1 * step))
    return MicrobatchTrainer(model, opt, sched, max_grad_norm=max_grad_norm,
                             **trainer_kw)


def full_state(trainer: MicrobatchTrainer) -> Dict[str, torch.Tensor]:
    """Params + Adam moments + Adam step counters, as named tensors."""
    state: Dict[str, torch.Tensor] = {}
    opt_state = trainer.optimizer.state
    for name, p in trainer.model.named_parameters():
        state[f"param::{name}"] = p.detach().clone()
    for name, p in trainer.model.named_parameters():
        st = opt_state.get(p)
        if not st:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            state[f"adam::{name}::{key}"] = st[key].detach().clone()
        step = st["step"]
        step = step.detach().clone() if torch.is_tensor(step) \
            else torch.tensor(float(step))
        state[f"adam::{name}::step"] = step.to(torch.float64)
    return state


def max_state_diff(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> float:
    assert a.keys() == b.keys(), (a.keys() ^ b.keys())
    worst = 0.0
    for key in a:
        worst = max(worst, float((a[key].double() - b[key].double()).abs().max()))
    return worst


def assert_state_bitwise_equal(a: Dict[str, torch.Tensor],
                               b: Dict[str, torch.Tensor]) -> None:
    assert a.keys() == b.keys(), (a.keys() ^ b.keys())
    for key in a:
        assert torch.equal(a[key], b[key]), f"state mismatch at {key}"


def snapshot(trainer: MicrobatchTrainer) -> dict:
    return {
        "params": [p.detach().clone() for p in trainer.model.parameters()],
        "opt_state_dict": copy.deepcopy(trainer.optimizer.state_dict()),
        "last_lr": list(trainer.scheduler.get_last_lr()),
        "scale": trainer.scale,
    }


def assert_unchanged(trainer: MicrobatchTrainer, snap: dict) -> None:
    """Bitwise check that params / Adam state / scheduler / scale did not move."""
    for p, ref in zip(trainer.model.parameters(), snap["params"]):
        assert torch.equal(p.detach(), ref)
    now = trainer.optimizer.state_dict()
    ref = snap["opt_state_dict"]
    assert now["state"].keys() == ref["state"].keys()
    for k in now["state"]:
        for field in now["state"][k]:
            assert torch.equal(torch.as_tensor(now["state"][k][field]),
                               torch.as_tensor(ref["state"][k][field]))
    assert list(trainer.scheduler.get_last_lr()) == snap["last_lr"]
    assert trainer.scale == snap["scale"]


def adam_step_count(trainer: MicrobatchTrainer) -> int:
    state = trainer.optimizer.state_dict()["state"]
    if not state:
        return 0
    return int(torch.as_tensor(next(iter(state.values()))["step"]).item())


def flat_grads(trainer: MicrobatchTrainer) -> torch.Tensor:
    return torch.cat([
        p.grad.detach().reshape(-1).double()
        for p in trainer.model.parameters()
        if p.grad is not None
    ])
