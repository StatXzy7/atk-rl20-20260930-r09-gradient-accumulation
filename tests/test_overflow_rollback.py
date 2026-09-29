"""Loss-scaling protocol: overflow mid-accumulation must roll back the
entire logical update atomically."""

import torch

from microbatch_trainer import OVERFLOW, SKIPPED_NO_TOKENS, UPDATED
from tiny_model import make_logical_batch, split_batch, token_cross_entropy_sum
from helpers import (adam_step_count, assert_state_bitwise_equal,
                     assert_unchanged, full_state, make_trainer,
                     max_state_diff, snapshot)

PARTITION = [4, 4, 2]


def poison_at(microbatch_index):
    """Wrap token_cross_entropy_sum so one microbatch produces non-finite
    gradients (simulating an fp16-style overflow in the backward pass)."""
    counter = {"i": 0}

    def loss_fn(model, mb):
        loss_sum, n = token_cross_entropy_sum(model, mb)
        i = counter["i"]
        counter["i"] += 1
        if i == microbatch_index:
            loss_sum = loss_sum * float("inf")
        return loss_sum, n

    return loss_fn


def test_overflow_mid_accumulation_rolls_back_everything():
    batch1 = make_logical_batch(seed=123)
    batch2 = make_logical_batch(seed=456)

    ref = make_trainer()
    ref.logical_update(split_batch(batch1, PARTITION), token_cross_entropy_sum)
    ref.logical_update(split_batch(batch2, PARTITION), token_cross_entropy_sum)

    cur = make_trainer()
    r1 = cur.logical_update(split_batch(batch1, PARTITION),
                            token_cross_entropy_sum)
    assert r1.status == UPDATED
    snap = snapshot(cur)  # non-trivial Adam state after the first update

    # Overflow injected in the *middle* microbatch of the second update.
    r2 = cur.logical_update(split_batch(batch2, PARTITION), poison_at(1))
    assert r2.status == OVERFLOW
    assert r2.scale == snap["scale"] * 0.5

    # Nothing may have partially advanced.
    assert_unchanged(cur, {**snap, "scale": snap["scale"] * 0.5})
    assert adam_step_count(cur) == 1
    assert all(p.grad is None for p in cur.model.parameters())

    # Retrying the same logical batch with the reduced scale succeeds and,
    # because the scale is a power of two, is bitwise identical to the
    # reference that never saw an overflow (see README, sec. 4).
    r3 = cur.logical_update(split_batch(batch2, PARTITION),
                            token_cross_entropy_sum)
    assert r3.status == UPDATED
    assert_state_bitwise_equal(full_state(ref), full_state(cur))
    assert ref.scheduler.get_last_lr() == cur.scheduler.get_last_lr()


def test_overflow_on_first_update_leaves_optimizer_state_empty():
    cur = make_trainer()
    batch = make_logical_batch(seed=123)
    params_before = [p.detach().clone() for p in cur.model.parameters()]

    r = cur.logical_update(split_batch(batch, PARTITION), poison_at(0))
    assert r.status == OVERFLOW
    assert adam_step_count(cur) == 0
    assert len(cur.optimizer.state) == 0
    for p, ref in zip(cur.model.parameters(), params_before):
        assert torch.equal(p.detach(), ref)
    assert cur.scheduler.get_last_lr() == [0.05]


def test_consecutive_overflows_decay_scale_geometrically():
    cur = make_trainer()
    batch = make_logical_batch(seed=123)
    scales = []
    for _ in range(3):
        r = cur.logical_update(split_batch(batch, PARTITION), poison_at(2))
        assert r.status == OVERFLOW
        scales.append(r.scale)
    assert scales == [0.5, 0.25, 0.125]
    assert adam_step_count(cur) == 0


def test_scale_grows_back_after_configured_successes():
    cur = make_trainer(init_scale=0.25, growth_interval=2)
    batch = make_logical_batch(seed=123)
    for _ in range(2):
        r = cur.logical_update(split_batch(batch, PARTITION),
                               token_cross_entropy_sum)
        assert r.status == UPDATED
    assert cur.scale == 0.5
    # An overflow resets the success streak.
    r = cur.logical_update(split_batch(batch, PARTITION), poison_at(1))
    assert r.status == OVERFLOW
    assert cur.scale == 0.25


def test_zero_token_logical_batch_skips_update_and_weight_decay():
    cur = make_trainer(weight_decay=0.5)  # large decay: a step would be visible
    batch = make_logical_batch(seed=123)
    batch["mask"][:] = False
    snap = snapshot(cur)

    r = cur.logical_update(split_batch(batch, PARTITION),
                           token_cross_entropy_sum)
    assert r.status == SKIPPED_NO_TOKENS
    assert r.total_tokens == 0
    assert_unchanged(cur, snap)  # bitwise: no step, hence no weight decay
    assert adam_step_count(cur) == 0

    # Sanity: a normal batch with the same trainer DOES move the params,
    # so the bitwise-equality assertion above is meaningful.
    good = make_logical_batch(seed=123)
    r2 = cur.logical_update(split_batch(good, PARTITION),
                            token_cross_entropy_sum)
    assert r2.status == UPDATED
    assert adam_step_count(cur) == 1
    moved = any(not torch.equal(p.detach(), ref)
                for p, ref in zip(cur.model.parameters(), snap["params"]))
    assert moved


def test_partially_empty_logical_batch_still_updates():
    """Microbatches with zero valid tokens contribute nothing, but the
    logical update itself must still happen (and match the reference)."""
    cur = make_trainer()
    ref = make_trainer()
    batch = make_logical_batch(seed=123)  # seqs 2..4 fully masked
    r = cur.logical_update(split_batch(batch, [2, 3, 5]),
                           token_cross_entropy_sum)
    assert r.status == UPDATED
    assert r.total_tokens == 32
    ref.logical_update(split_batch(batch, [10]), token_cross_entropy_sum)
    assert max_state_diff(full_state(ref), full_state(cur)) < 1e-12
