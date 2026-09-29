"""Loss scaling 协议：非有限梯度导致整次逻辑更新失败并严格回滚。"""
from __future__ import annotations

import torch

from microbatch_trainer.trainer import logical_train_step
from .helpers import (
    fresh,
    full_snapshot,
    assert_snapshot_unchanged,
    make_all_pad_batch,
    make_batch,
)


def test_mid_microbatch_overflow_rolls_back_everything():
    batch = make_batch(seed=0)
    model, opt, sch, sc = fresh(dtype=torch.float32, init_scale=2.0**10)
    before = full_snapshot(model, opt, sch, sc)

    # microbatch 划分 [6,4,5,3,6,2,1,6,0]；在第 3 个（idx=2，已累积两组后）注入。
    result = logical_train_step(
        model, opt, sch, sc, batch, microbatch_size=1, overflow_in_microbatch=2
    )

    assert result.status == "overflow"
    assert result.n_valid_tokens == 33
    # 之前 microbatch 的梯度已随全部累积梯度一并清空。
    for p in model.parameters():
        assert p.grad is None
    # 参数 / Adam 一阶二阶矩 / 步数 / 调度器全部保持更新前。
    assert_snapshot_unchanged(before, model, opt, sch, sc)
    # 只有 scale 发生调整。
    assert sc.scale == before["scale"] * 0.5
    assert opt.t == 0 and sch.step_count == 0


def test_last_microbatch_overflow_also_full_rollback():
    batch = make_batch(seed=0)
    model, opt, sch, sc = fresh(dtype=torch.float32, init_scale=2.0**10)
    before = full_snapshot(model, opt, sch, sc)

    # ms=4 ⇒ 分组 [前4行=18, 中4行=11, 末1行=0]，向最后一个非空组 idx=1 注入。
    result = logical_train_step(
        model, opt, sch, sc, batch, microbatch_size=4, overflow_in_microbatch=1
    )
    assert result.status == "overflow"
    assert_snapshot_unchanged(before, model, opt, sch, sc)
    assert sc.scale == before["scale"] * 0.5


def test_recovery_after_overflow():
    batch = make_batch(seed=0)
    model, opt, sch, sc = fresh(dtype=torch.float32, init_scale=2.0**10)
    bad = logical_train_step(
        model, opt, sch, sc, batch, 1, overflow_in_microbatch=2
    )
    assert bad.status == "overflow"
    assert sc.scale == 2.0**9

    # 降低 scale 后不再注入，同一份数据应当成功完成一次完整更新。
    good = logical_train_step(model, opt, sch, sc, batch, 1)
    assert good.status == "updated"
    assert opt.t == 1 and sch.step_count == 1
    assert good.grad_norm is not None and good.grad_norm > 0


def test_scale_backoff_repeats_until_safe():
    batch = make_batch(seed=0)
    model, opt, sch, sc = fresh(dtype=torch.float32, init_scale=2.0**12)
    for expected_scale in [2.0**11, 2.0**10, 2.0**9]:
        res = logical_train_step(
            model, opt, sch, sc, batch, 1, overflow_in_microbatch=2
        )
        assert res.status == "overflow"
        assert sc.scale == expected_scale
        assert opt.t == 0 and sch.step_count == 0


def test_all_pad_logical_batch_is_skip_without_weight_decay():
    batch = make_all_pad_batch()
    model, opt, sch, sc = fresh(dtype=torch.float64, weight_decay=0.5)
    before = full_snapshot(model, opt, sch, sc)

    result = logical_train_step(model, opt, sch, sc, batch, microbatch_size=2)
    assert result.status == "skipped"
    assert result.n_valid_tokens == 0
    # 即使 weight_decay 很大，跳过时参数（含衰减）也完全不变。
    assert_snapshot_unchanged(before, model, opt, sch, sc)
    assert opt.t == 0 and sch.step_count == 0
    for p in model.parameters():
        assert p.grad is None


def test_partition_with_empty_microbatch_still_updates_correctly():
    # ms=4 时最后一组（第 9 行）有效 token 为 0：应被忽略而不是除以零。
    batch = make_batch(seed=0)
    model, opt, sch, sc = fresh()
    result = logical_train_step(model, opt, sch, sc, batch, microbatch_size=4)
    assert result.status == "updated"
    assert result.n_valid_tokens == 33
    assert result.n_microbatches == 3
    assert opt.t == 1 and sch.step_count == 1


def test_empty_group_in_middle_is_skipped_not_weighted():
    # 自定义数据：行 2 全 pad，使 ms=2 的分组计数为 [10, 0, 11, 6, 6]。
    g = torch.Generator().manual_seed(42)
    vocab, n_rows, seq = 37, 5, 6
    ids = torch.randint(0, vocab, (n_rows, seq), generator=g)
    labels = torch.randint(0, vocab, (n_rows, seq), generator=g)
    mask = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # 6
         [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],   # 4
         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # 0：中间整组为空
         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],   # 6
         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]   # 6
    )
    from microbatch_trainer import Batch

    batch = Batch(ids, mask, labels)
    assert batch.num_valid_tokens == 22

    # 全批次 vs ms=2（中间组空）必须等价。
    m_full, o_full, s_full, sc_full = fresh()
    r_full = logical_train_step(m_full, o_full, s_full, sc_full, batch, microbatch_size=5)
    m_mb, o_mb, s_mb, sc_mb = fresh()
    r_mb = logical_train_step(m_mb, o_mb, s_mb, sc_mb, batch, microbatch_size=2)

    assert r_full.status == r_mb.status == "updated"
    err = max(
        (a - b).abs().max().item()
        for a, b in zip(m_full.parameters(), m_mb.parameters())
    )
    assert err < 1e-11
    assert o_mb.t == 1 and s_mb.step_count == 1


def test_nonfinite_after_unscale_still_atomic_failure(monkeypatch):
    # 构造“累积期有限、反缩放后非有限”的极端情形：
    # 累积梯度全部为有限的极大值，除以 scale 仍为 inf（例如 scale 不足）。
    batch = make_batch(seed=0)
    model, opt, sch, sc = fresh(dtype=torch.float32, init_scale=1.0)
    before = full_snapshot(model, opt, sch, sc)

    handle = model.fc1.weight.register_hook(
        lambda g: g + torch.finfo(torch.float32).max
    )
    try:
        result = logical_train_step(model, opt, sch, sc, batch, microbatch_size=1)
    finally:
        handle.remove()

    assert result.status == "overflow"
    assert_snapshot_unchanged(before, model, opt, sch, sc)
    assert opt.t == 0 and sch.step_count == 0
    for p in model.parameters():
        assert p.grad is None
