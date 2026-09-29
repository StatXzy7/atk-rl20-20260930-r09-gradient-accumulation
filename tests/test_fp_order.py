"""Floating-point summation-order analysis.

Different partitions sum the same mathematical quantities in different
orders, so bitwise equality is NOT promised.  These tests bound the
observed divergence and show it shrinks with the mantissa width, i.e. it
is pure rounding noise rather than a systematic error (contrast with the
counterexamples, whose error is ~1e-2 relative in any precision).
"""

import torch

from tiny_model import make_logical_batch, split_batch, token_cross_entropy_sum
from helpers import flat_grads, make_trainer

PARTITIONS = {"full": [10], "per_sequence": [1] * 10, "uneven": [4, 4, 2]}


def grads_by_partition(dtype):
    batch = make_logical_batch(seed=123, dtype=dtype)
    out = {}
    for name, partition in PARTITIONS.items():
        trainer = make_trainer(dtype=dtype)
        trainer.prepare_grads(split_batch(batch, partition),
                              token_cross_entropy_sum)
        out[name] = flat_grads(trainer)
    return out


def relative_spread(dtype):
    grads = grads_by_partition(dtype)
    ref = grads["full"]
    ref_norm = float(ref.norm())
    return {
        name: float((g - ref).abs().max()) / ref_norm
        for name, g in grads.items()
    }


def test_float64_spread_is_at_rounding_level():
    spread = relative_spread(torch.float64)
    print("\nfloat64 relative grad spread:", spread)
    assert max(spread.values()) < 1e-13


def test_float32_spread_is_larger_but_still_rounding_level():
    spread = relative_spread(torch.float32)
    print("\nfloat32 relative grad spread:", spread)
    assert max(spread.values()) < 1e-5


def test_spread_scales_with_mantissa_width():
    s64 = max(relative_spread(torch.float64).values())
    s32 = max(relative_spread(torch.float32).values())
    # ~2^23 more mantissa bits must buy orders of magnitude of agreement;
    # a systematic error would be precision-independent.
    assert s32 > 1e3 * s64
