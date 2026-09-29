"""Different microbatch partitions must reproduce the full-batch update."""

import pytest
import torch

from microbatch_trainer import UPDATED
from tiny_model import make_logical_batch, split_batch, token_cross_entropy_sum
from helpers import make_trainer, full_state, max_state_diff, flat_grads

# Uneven splits, a zero-length microbatch ([4,0,6]), and a zero-valid-token
# microbatch ([2,3,5]: sequences 2..4 are fully masked out in the fixture).
PARTITIONS = [[10], [4, 4, 2], [1] * 10, [3, 3, 4], [2, 3, 5], [7, 2, 1],
              [4, 0, 6]]

TOL_F64 = 1e-12


@pytest.mark.parametrize("partition", PARTITIONS)
def test_gradient_matches_full_batch(partition):
    batch = make_logical_batch(seed=123)
    ref = make_trainer()
    cur = make_trainer()
    status_ref, n_ref, _ = ref.prepare_grads(split_batch(batch, [10]),
                                             token_cross_entropy_sum)
    status_cur, n_cur, _ = cur.prepare_grads(split_batch(batch, partition),
                                             token_cross_entropy_sum)
    assert status_ref == status_cur == UPDATED
    assert n_ref == n_cur == 32
    diff = float((flat_grads(ref) - flat_grads(cur)).abs().max())
    assert diff < TOL_F64


@pytest.mark.parametrize("partition", PARTITIONS)
def test_two_sequential_updates_match_full_batch(partition):
    """Params, Adam moments, step counts and scheduler must track the
    full-batch reference through two logical updates (the second update
    depends on the Adam state produced by the first)."""
    ref = make_trainer()
    cur = make_trainer()
    for seed in (123, 456):
        batch = make_logical_batch(seed=seed)
        r_ref = ref.logical_update(split_batch(batch, [10]),
                                   token_cross_entropy_sum)
        r_cur = cur.logical_update(split_batch(batch, partition),
                                   token_cross_entropy_sum)
        assert r_ref.status == r_cur.status == UPDATED
        assert r_ref.total_tokens == r_cur.total_tokens
        assert r_ref.grad_norm == pytest.approx(r_cur.grad_norm, abs=1e-12)
    assert max_state_diff(full_state(ref), full_state(cur)) < TOL_F64
    assert ref.scheduler.get_last_lr() == cur.scheduler.get_last_lr()


@pytest.mark.parametrize("partition", PARTITIONS)
def test_updates_match_with_clipping(partition):
    """Global-norm clipping happens once, after accumulation, and stays
    partition-independent (norm computed in float64)."""
    ref = make_trainer(max_grad_norm=0.05)
    cur = make_trainer(max_grad_norm=0.05)
    for seed in (123, 456):
        batch = make_logical_batch(seed=seed)
        r_ref = ref.logical_update(split_batch(batch, [10]),
                                   token_cross_entropy_sum)
        r_cur = cur.logical_update(split_batch(batch, partition),
                                   token_cross_entropy_sum)
        assert r_ref.grad_norm > 0.05  # clipping is actually active
        assert r_ref.grad_norm == pytest.approx(r_cur.grad_norm, abs=1e-12)
    assert max_state_diff(full_state(ref), full_state(cur)) < TOL_F64


def test_float32_partition_tolerance():
    """In float32 the same equivalences hold, with rounding-level tolerance."""
    batch = make_logical_batch(seed=123, dtype=torch.float32)
    ref = make_trainer(dtype=torch.float32)
    cur = make_trainer(dtype=torch.float32)
    ref.logical_update(split_batch(batch, [10]), token_cross_entropy_sum)
    cur.logical_update(split_batch(batch, [1] * 10), token_cross_entropy_sum)
    assert max_state_diff(full_state(ref), full_state(cur)) < 1e-5
