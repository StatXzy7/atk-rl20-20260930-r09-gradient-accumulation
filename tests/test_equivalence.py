"""不同 microbatch 切分 ⇔ 一次全批次：梯度、裁剪、多步参数与 Adam 状态等价。"""
from __future__ import annotations

import torch

from microbatch_trainer.trainer import logical_train_step
from .helpers import (
    PARTITIONS,
    fresh,
    grads_after_accumulate,
    make_batch,
    max_abs,
    named_params_max_diff,
    states_max_diff,
)


ATOL = 1e-12


def test_gradients_identical_across_partitions():
    batch = make_batch(seed=0)
    ref_model, ref_opt, _, ref_sc = fresh()
    ref_acc, ref_grads = grads_after_accumulate(
        ref_model, ref_opt, ref_sc, batch, microbatch_size=9
    )
    assert ref_acc.status == "updated"
    assert ref_acc.n_valid_tokens == 33

    for ms, counts in PARTITIONS[1:]:
        model, opt, _, sc = fresh()
        acc, grads = grads_after_accumulate(model, opt, sc, batch, microbatch_size=ms)
        assert acc.status == "updated"
        assert acc.n_valid_tokens == 33
        assert grads.keys() == ref_grads.keys()
        err = max_abs(ref_grads, grads)
        assert err < ATOL, f"partition {ms} (counts {counts}) grad err {err}"


def test_global_norm_matches_full_batch():
    batch = make_batch(seed=0)
    ref_model, ref_opt, _, ref_sc = fresh()
    ref_acc, _ = grads_after_accumulate(ref_model, ref_opt, ref_sc, batch, 9)
    norms = []
    for ms, _ in PARTITIONS[1:]:
        model, opt, _, sc = fresh()
        acc, _ = grads_after_accumulate(model, opt, sc, batch, ms)
        norms.append(acc.grad_norm)
        assert abs(acc.grad_norm - ref_acc.grad_norm) < ATOL
    assert len(set(round(n, 12) for n in norms)) == 1


def test_clipping_applied_once_after_normalization():
    # 阈值取在真实 norm 之下，确保确实发生裁剪。
    batch = make_batch(seed=0)
    max_norm = 0.05
    ref_model, ref_opt, _, ref_sc = fresh()
    ref_acc, ref_grads = grads_after_accumulate(
        ref_model, ref_opt, ref_sc, batch, 9, max_norm=max_norm
    )
    assert ref_acc.clipped is True
    for ms, _ in PARTITIONS[1:]:
        model, opt, _, sc = fresh()
        acc, grads = grads_after_accumulate(model, opt, sc, batch, ms, max_norm=max_norm)
        assert acc.clipped is True
        assert abs(acc.clip_coef - ref_acc.clip_coef) < ATOL
        assert max_abs(ref_grads, grads) < ATOL
        # 裁剪后范数等于阈值。
        norm = torch.sqrt(
            sum(p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None)
        )
        assert abs(norm.item() - max_norm) < 1e-8


def test_multi_step_params_moments_scheduler_lr_equivalent():
    batches = [make_batch(seed=s) for s in range(5)]
    n_steps = len(batches)

    def trajectory(ms):
        model, opt, sch, sc = fresh()
        results = []
        for b in batches:
            results.append(
                logical_train_step(model, opt, sch, sc, b, microbatch_size=ms, max_norm=0.1)
            )
        return model, opt, sch, sc, results

    ref_model, ref_opt, ref_sch, ref_sc, ref_res = trajectory(9)
    for ms, _ in PARTITIONS[1:]:
        model, opt, sch, sc, res = trajectory(ms)
        perr = named_params_max_diff(ref_model, model)
        m_err, v_err = states_max_diff(ref_opt, opt)
        assert perr < 1e-11, f"ms={ms} param err {perr}"
        assert m_err < 1e-11 and v_err < 1e-11
        assert opt.t == n_steps == ref_opt.t
        assert sch.step_count == ref_sch.step_count == n_steps
        assert abs(opt.lr - ref_opt.lr) < 1e-15
        assert sc.scale == ref_sc.scale
        assert [r.clipped for r in res] == [r.clipped for r in ref_res]


def test_next_step_predictions_equivalent():
    # 等价参数 ⇒ 下一步前向 logits / loss 等价。
    batches = [make_batch(seed=s) for s in range(3)]
    held_out = make_batch(seed=99)

    def fit(ms):
        model, opt, sch, sc = fresh()
        for b in batches:
            logical_train_step(model, opt, sch, sc, b, ms, max_norm=0.1)
        return model

    m_full = fit(9)
    with torch.no_grad():
        ref_logits = m_full(held_out.token_ids)
    for ms, _ in PARTITIONS[1:]:
        m = fit(ms)
        with torch.no_grad():
            logits = m(held_out.token_ids)
        err = (logits - ref_logits).abs().max().item()
        assert err < 1e-11, f"ms={ms} logits err {err}"
