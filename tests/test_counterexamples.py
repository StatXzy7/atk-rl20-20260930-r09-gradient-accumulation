"""Counterexamples: popular but wrong accumulation schemes must NOT match
the full-batch update.  These tests pin down *why* the protocol in
microbatch_trainer.py is the correct one."""

import torch

from tiny_model import (make_logical_batch, make_model, split_batch,
                        token_cross_entropy_sum)
from helpers import make_trainer

PARTITION = [4, 4, 2]  # valid tokens per microbatch: 11, 11, 10


def reference_grad():
    trainer = make_trainer()
    batch = make_logical_batch(seed=123)
    trainer.prepare_grads(split_batch(batch, [10]), token_cross_entropy_sum)
    return torch.cat([p.grad.detach().reshape(-1)
                      for p in trainer.model.parameters()])


def test_mean_of_means_is_wrong_for_uneven_token_counts():
    """Averaging per-microbatch *means* with equal weights reweights token i
    by 1/n_{m(i)} instead of 1/N; it is a different objective."""
    model = make_model(seed=0)
    batch = make_logical_batch(seed=123)
    microbatches = split_batch(batch, PARTITION)
    counts = [token_cross_entropy_sum(model, mb)[1] for mb in microbatches]
    assert len(set(counts)) > 1  # the partition is genuinely uneven

    model.zero_grad(set_to_none=True)
    for mb in microbatches:
        loss_sum, n = token_cross_entropy_sum(model, mb)
        (loss_sum / n / len(microbatches)).backward()
    mean_of_means = torch.cat([p.grad.detach().reshape(-1)
                               for p in model.parameters()])

    ref = reference_grad()
    rel = float((mean_of_means - ref).norm() / ref.norm())
    assert rel > 1e-2  # demonstrably NOT equivalent


def test_token_weighted_mean_is_right():
    """Weighting microbatch means by n_m/N recovers the exact objective --
    this is the correct way to combine per-microbatch means."""
    model = make_model(seed=0)
    batch = make_logical_batch(seed=123)
    microbatches = split_batch(batch, PARTITION)
    total = sum(token_cross_entropy_sum(model, mb)[1] for mb in microbatches)

    model.zero_grad(set_to_none=True)
    for mb in microbatches:
        loss_sum, n = token_cross_entropy_sum(model, mb)
        (loss_sum / total).backward()
    weighted = torch.cat([p.grad.detach().reshape(-1)
                          for p in model.parameters()])

    ref = reference_grad()
    assert float((weighted - ref).abs().max()) < 1e-12


def test_stepping_per_microbatch_is_not_equivalent():
    """Running optimizer.step() on every microbatch applies Adam's
    preconditioning, the LR schedule and decoupled weight decay M times
    instead of once -- a different trajectory entirely."""
    batch = make_logical_batch(seed=123)

    naive_model = make_model(seed=0)
    naive_opt = torch.optim.AdamW(naive_model.parameters(), lr=0.05,
                                  betas=(0.9, 0.95), weight_decay=0.1)
    for mb in split_batch(batch, PARTITION):
        loss_sum, n = token_cross_entropy_sum(naive_model, mb)
        naive_opt.zero_grad(set_to_none=True)
        (loss_sum / n).backward()
        naive_opt.step()

    trainer = make_trainer()
    trainer.logical_update(split_batch(batch, PARTITION),
                           token_cross_entropy_sum)

    diffs = [float((p1.detach() - p2.detach()).abs().max())
             for p1, p2 in zip(naive_model.parameters(),
                               trainer.model.parameters())]
    assert max(diffs) > 1e-3  # far outside any rounding tolerance


def test_clipping_each_microbatch_is_not_equivalent():
    """Clipping is nonlinear: clip(sum of grads) != sum of clip(grads).
    Clipping per microbatch therefore cannot match full-batch clipping."""
    max_norm = 0.05
    batch = make_logical_batch(seed=123)

    # Naive: normalise and clip every microbatch contribution, then add.
    naive_model = make_model(seed=0)
    microbatches = split_batch(batch, PARTITION)
    total = sum(token_cross_entropy_sum(naive_model, mb)[1]
                for mb in microbatches)
    accumulated = [torch.zeros_like(p) for p in naive_model.parameters()]
    for mb in microbatches:
        loss_sum, n = token_cross_entropy_sum(naive_model, mb)
        if n == 0:
            continue
        naive_model.zero_grad(set_to_none=True)
        (loss_sum / total).backward()
        torch.nn.utils.clip_grad_norm_(list(naive_model.parameters()),
                                       max_norm)
        for buf, p in zip(accumulated, naive_model.parameters()):
            buf += p.grad.detach()

    # Correct: accumulate, normalise, clip once (what the trainer does).
    trainer = make_trainer(max_grad_norm=max_norm)
    trainer.prepare_grads(microbatches, token_cross_entropy_sum)
    norm = trainer._global_norm()
    clip = min(1.0, max_norm / (norm + 1e-12))
    correct = torch.cat([(p.grad.detach() * clip).reshape(-1)
                         for p in trainer.model.parameters()])

    naive = torch.cat([b.reshape(-1) for b in accumulated])
    rel = float((naive - correct).norm() / correct.norm())
    assert rel > 1e-2
