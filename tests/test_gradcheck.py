"""Finite-difference verification of the accumulated gradient against the
exact logical-batch objective L = (sum of valid token losses) / N."""

import torch

from microbatch_trainer import UPDATED
from tiny_model import (make_logical_batch, make_model, split_batch,
                        token_cross_entropy_sum)
from helpers import make_trainer

H = 1e-6


def objective(model, batch) -> float:
    loss_sum, n = token_cross_entropy_sum(model, batch)
    return float(loss_sum) / n


def finite_difference_grad(model, batch):
    grads = []
    with torch.no_grad():
        for p in model.parameters():
            g = torch.zeros_like(p)
            for idx in range(p.numel()):
                flat = p.reshape(-1)
                orig = flat[idx].item()
                flat[idx] = orig + H
                f_plus = objective(model, batch)
                flat[idx] = orig - H
                f_minus = objective(model, batch)
                flat[idx] = orig
                g.reshape(-1)[idx] = (f_plus - f_minus) / (2 * H)
            grads.append(g)
    return torch.cat([g.reshape(-1) for g in grads])


def test_accumulated_grad_matches_finite_difference():
    torch.manual_seed(0)
    batch = make_logical_batch(seed=123)
    trainer = make_trainer()
    # Accumulate through an uneven partition, then compare against central
    # finite differences of the logical-batch objective.
    status, n, _ = trainer.prepare_grads(split_batch(batch, [3, 3, 4]),
                                         token_cross_entropy_sum)
    assert status == UPDATED and n == 32
    analytic = torch.cat([p.grad.detach().reshape(-1)
                          for p in trainer.model.parameters()])

    ref_model = make_model(seed=0)
    numeric = finite_difference_grad(ref_model, batch)

    abs_err = (analytic - numeric).abs()
    rel_err = abs_err / numeric.abs().clamp_min(1e-3)
    assert abs_err.max() < 1e-6
    assert rel_err.max() < 1e-4


def test_objective_is_token_mean_not_microbatch_mean():
    """The accumulated gradient must equal autograd on the exact objective
    computed in one shot over the whole logical batch."""
    batch = make_logical_batch(seed=123)
    ref_model = make_model(seed=0)
    loss_sum, n = token_cross_entropy_sum(ref_model, batch)
    (loss_sum / n).backward()
    exact = torch.cat([p.grad.detach().reshape(-1)
                       for p in ref_model.parameters()])

    trainer = make_trainer()
    trainer.prepare_grads(split_batch(batch, [2, 3, 5]),
                          token_cross_entropy_sum)
    accumulated = torch.cat([p.grad.detach().reshape(-1)
                             for p in trainer.model.parameters()])
    assert float((exact - accumulated).abs().max()) < 1e-12
