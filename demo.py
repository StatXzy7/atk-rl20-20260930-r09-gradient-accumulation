"""End-to-end demonstration: partition equivalence, overflow rollback,
and the counterexamples.  Run:  python3 demo.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

import torch

from helpers import flat_grads, full_state, make_trainer, max_state_diff
from microbatch_trainer import OVERFLOW, UPDATED
from tiny_model import (make_logical_batch, make_model, split_batch,
                        token_cross_entropy_sum)

PARTITIONS = [[10], [4, 4, 2], [1] * 10, [3, 3, 4], [2, 3, 5], [4, 0, 6]]


def main():
    print("== 1. gradient spread across partitions (rounding only) ==")
    for dtype in (torch.float64, torch.float32):
        batch = make_logical_batch(seed=123, dtype=dtype)
        grads = {}
        for part in PARTITIONS:
            t = make_trainer(dtype=dtype)
            t.prepare_grads(split_batch(batch, part), token_cross_entropy_sum)
            grads[tuple(part)] = flat_grads(t)
        ref = grads[(10,)]
        for part, g in grads.items():
            rel = float((g - ref).abs().max()) / float(ref.norm())
            print(f"  {str(dtype):15s} partition {str(list(part)):18s} "
                  f"rel. max grad diff vs full batch: {rel:.3e}")

    print("\n== 2. full state after two logical updates (float64) ==")
    ref = make_trainer()
    for seed in (123, 456):
        ref.logical_update(split_batch(make_logical_batch(seed=seed), [10]),
                           token_cross_entropy_sum)
    for part in PARTITIONS[1:]:
        cur = make_trainer()
        for seed in (123, 456):
            cur.logical_update(split_batch(make_logical_batch(seed=seed), part),
                               token_cross_entropy_sum)
        print(f"  partition {str(part):18s} max |params+Adam state| diff: "
              f"{max_state_diff(full_state(ref), full_state(cur)):.3e}")

    print("\n== 3. overflow mid-accumulation: atomic rollback ==")
    clean = make_trainer()
    hit = make_trainer()
    b1, b2 = make_logical_batch(seed=123), make_logical_batch(seed=456)
    clean.logical_update(split_batch(b1, [4, 4, 2]), token_cross_entropy_sum)
    clean.logical_update(split_batch(b2, [4, 4, 2]), token_cross_entropy_sum)
    hit.logical_update(split_batch(b1, [4, 4, 2]), token_cross_entropy_sum)
    counter = {"i": 0}

    def poison(model, mb):
        ls, n = token_cross_entropy_sum(model, mb)
        i = counter["i"]
        counter["i"] += 1
        return (ls * float("inf")) if i == 1 else ls, n

    r = hit.logical_update(split_batch(b2, [4, 4, 2]), poison)
    print(f"  injected inf in microbatch 1 -> status={r.status}, "
          f"scale {1.0} -> {r.scale}")
    r = hit.logical_update(split_batch(b2, [4, 4, 2]), token_cross_entropy_sum)
    assert r.status == UPDATED
    diff = max_state_diff(full_state(clean), full_state(hit))
    print(f"  retried with scale=0.5 -> max state diff vs never-overflowed "
          f"reference: {diff} (bitwise identical)")

    print("\n== 4. counterexamples (must NOT match) ==")
    batch = make_logical_batch(seed=123)
    ref_g = make_trainer()
    ref_g.prepare_grads(split_batch(batch, [10]), token_cross_entropy_sum)
    ref = flat_grads(ref_g)

    m = make_model(seed=0)
    mbs = split_batch(batch, [4, 4, 2])
    for mb in mbs:
        ls, n = token_cross_entropy_sum(m, mb)
        (ls / n / len(mbs)).backward()
    mom = torch.cat([p.grad.detach().reshape(-1) for p in m.parameters()])
    print(f"  mean-of-means gradient:      rel err {float((mom - ref).norm() / ref.norm()):.3e}")

    naive = make_model(seed=0)
    opt = torch.optim.AdamW(naive.parameters(), lr=0.05, betas=(0.9, 0.95),
                            weight_decay=0.1)
    for mb in mbs:
        ls, n = token_cross_entropy_sum(naive, mb)
        opt.zero_grad(set_to_none=True)
        (ls / n).backward()
        opt.step()
    t = make_trainer()
    t.logical_update(mbs, token_cross_entropy_sum)
    d = max(float((a.detach() - b.detach()).abs().max())
            for a, b in zip(naive.parameters(), t.model.parameters()))
    print(f"  step per microbatch:         max param diff {d:.3e}")


if __name__ == "__main__":
    main()
