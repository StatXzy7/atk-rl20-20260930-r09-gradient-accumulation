"""两个必须失败的反例：等权均值 microbatch、逐 microbatch 更新参数。"""
from __future__ import annotations

import torch

from microbatch_trainer.trainer import (
    logical_train_step,
    naive_mean_microbatch_train_step,
    per_microbatch_update_train_step,
)
from .helpers import fresh, make_batch


def _ref_grads():
    model, opt, sch, sc = fresh()
    batch = make_batch(seed=0)
    logical_train_step(model, opt, sch, sc, batch, microbatch_size=9)
    return [p.detach().clone() for p in model.parameters()]


def test_naive_equal_weight_mean_gives_wrong_gradient():
    batch = make_batch(seed=0)
    model, opt, sch, _ = fresh()
    # ms=1 ⇒ 每个 microbatch 就是一行，有效 token 数 6,4,5,...,0。
    naive_mean_microbatch_train_step(model, opt, sch, batch, microbatch_size=1)
    wrong = [p.detach().clone() for p in model.parameters()]
    ref = _ref_grads()

    diffs = [(w - r).abs().max().item() for w, r in zip(wrong, ref)]
    assert max(diffs) > 1e-6, (
        "等权平均各组均值应与按 token 归一化不同，却碰巧相同；请检查数据是否不均匀"
    )


def test_naive_equal_weight_mean_coincides_when_group_sizes_equal():
    # 反例的反例：构造两组有效 token 数完全相等的数据，两种缩放才重合。
    g = torch.Generator().manual_seed(7)
    n_rows, seq, vocab = 4, 5, 37
    ids = torch.randint(0, vocab, (n_rows, seq), generator=g)
    labels = torch.randint(0, vocab, (n_rows, seq), generator=g)
    # 每行恰好 3 个有效 token ⇒ ms=2 的两组各有 6 个。
    mask = torch.zeros(n_rows, seq)
    mask[:, :3] = 1.0
    from microbatch_trainer import Batch

    batch = Batch(ids, mask, labels)

    m1, o1, s1, sc1 = fresh()
    logical_train_step(m1, o1, s1, sc1, batch, microbatch_size=2)
    m2, o2, s2, _ = fresh()
    naive_mean_microbatch_train_step(m2, o2, s2, batch, microbatch_size=2)
    err = max(
        (a - b).abs().max().item()
        for a, b in zip(m1.parameters(), m2.parameters())
    )
    assert err < 1e-11


def test_per_microbatch_updates_diverge():
    batch = make_batch(seed=0)

    ref_model, ref_opt, ref_sch, ref_sc = fresh()
    logical_train_step(ref_model, ref_opt, ref_sch, ref_sc, batch,
                       microbatch_size=9)

    bad_model, bad_opt, bad_sch, bad_sc = fresh()
    results = per_microbatch_update_train_step(
        bad_model, bad_opt, bad_sch, bad_sc, batch, microbatch_size=1
    )

    # 非空 microbatch 各触发一次真正的更新（空组跳过）。
    updated = [r for r in results if r.status == "updated"]
    assert len(updated) == 8
    assert bad_opt.t == 8
    assert bad_sch.step_count == 8

    diffs = [
        (a - b).abs().max().item()
        for a, b in zip(bad_model.parameters(), ref_model.parameters())
    ]
    assert max(diffs) > 1e-5, "逐 microbatch 更新不应与一次全批次更新等价"
