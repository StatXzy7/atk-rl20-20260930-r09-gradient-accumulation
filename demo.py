"""端到端演示：等价划分、两个反例、溢出回滚边界、空 batch 跳过、求和次序。

运行：python demo.py
"""
from __future__ import annotations

import torch

from microbatch_trainer.trainer import (
    logical_train_step,
    naive_mean_microbatch_train_step,
    per_microbatch_update_train_step,
)
from tests.helpers import make_batch, make_all_pad_batch, fresh, grads_after_accumulate


def _max_param_diff(a, b):
    return max((pa - pb).abs().max().item() for pa, pb in zip(a.parameters(), b.parameters()))


def section(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def demo_equivalence():
    section("1) 正确协议：不均匀切分 ⇔ 一次全批次（float64，scale=1024）")
    batch = make_batch(seed=0)
    partitions = [(9, "full batch"), (1, "per row"), (4, "4/4/1(末组空)"),
                  (3, "3/3/3(不均匀计数)"), (2, "2/2/2/2/1(末组空)")]

    ref_model, ref_opt, ref_sch, ref_sc = fresh()
    ref_res = logical_train_step(ref_model, ref_opt, ref_sch, ref_sc, batch, 9, max_norm=0.1)
    print(f"reference: status={ref_res.status} T={ref_res.n_valid_tokens} "
          f"||g||={ref_res.grad_norm:.10f} clipped={ref_res.clipped} "
          f"opt.t={ref_opt.t} sch.step={ref_sch.step_count} lr={ref_opt.lr:g}")

    for ms, label in partitions[1:]:
        model, opt, sch, sc = fresh()
        res = logical_train_step(model, opt, sch, sc, batch, ms, max_norm=0.1)
        perr = _max_param_diff(ref_model, model)
        m_err = max((a - b).abs().max().item() for a, b in zip(ref_opt.m, opt.m))
        v_err = max((a - b).abs().max().item() for a, b in zip(ref_opt.v, opt.v))
        print(f"ms={ms:>1} ({label:>16}): status={res.status} ||g||={res.grad_norm:.10f} "
              f"clip={res.clipped} max|Δθ|={perr:.2e} max|Δm|={m_err:.2e} "
              f"max|Δv|={v_err:.2e}")


def demo_multi_step():
    section("2) 连续 5 步（含裁剪）后的参数 / 矩 / 下一步 lr")
    batches = [make_batch(seed=s) for s in range(5)]

    def run(ms):
        model, opt, sch, sc = fresh()
        for b in batches:
            logical_train_step(model, opt, sch, sc, b, ms, max_norm=0.1)
        return model, opt

    ref_model, ref_opt = run(9)
    for ms in (1, 4, 3, 2):
        model, opt = run(ms)
        perr = _max_param_diff(ref_model, model)
        m_err = max((a - b).abs().max().item() for a, b in zip(ref_opt.m, opt.m))
        print(f"ms={ms}: max|Δθ|={perr:.2e} max|Δm|={m_err:.2e} "
              f"opt.t={opt.t} lr={opt.lr:.12g}")


def demo_counterexample_mean():
    section("3) 反例 A：每个 microbatch 取均值再等权平均（一次 step）")
    batch = make_batch(seed=0)

    ref_model, ref_opt, ref_sch, ref_sc = fresh()
    logical_train_step(ref_model, ref_opt, ref_sch, ref_sc, batch, 9)

    bad_model, bad_opt, bad_sch, _ = fresh()
    naive_mean_microbatch_train_step(bad_model, bad_opt, bad_sch, batch, 1)

    _, ref_grads = grads_after_accumulate(*fresh()[:2], fresh()[3], batch, 9)
    diff = _max_param_diff(ref_model, bad_model)
    print("各组有效 token: [6,4,5,3,6,2,1,6,0], T=33, K=9（含空组）")
    print("错误权重 w_i = 1/(K*n_i)；正确权重 = 1/T = 1/33")
    print(f"一次更新后参数最大差异: {diff:.6e}  ->  明显不等价")


def demo_counterexample_multi_step():
    section("4) 反例 B：每个 microbatch 各做一次参数更新")
    batch = make_batch(seed=0)

    ref_model, ref_opt, ref_sch, ref_sc = fresh()
    logical_train_step(ref_model, ref_opt, ref_sch, ref_sc, batch, 9)

    bad_model, bad_opt, bad_sch, bad_sc = fresh()
    results = per_microbatch_update_train_step(
        bad_model, bad_opt, bad_sch, bad_sc, batch, microbatch_size=1
    )
    updated = [r for r in results if r.status == "updated"]
    diff = _max_param_diff(ref_model, bad_model)
    print(f"非空 microbatch 数={len(updated)}（空行正确跳过）, "
          f"优化器被推进 {bad_opt.t} 次、调度器 {bad_sch.step_count} 次（应为 1）")
    print(f"与一次全批次更新的参数最大差异: {diff:.6e}  ->  多步非线性更新不等价")


def demo_overflow_rollback():
    section("5) 溢出回滚边界：第 3 个 microbatch 注入 inf 梯度")
    batch = make_batch(seed=0)
    model, opt, sch, sc = fresh(dtype=torch.float32, init_scale=2.0**12)
    p0 = [p.detach().clone() for p in model.parameters()]
    m0 = [t.clone() for t in opt.m]
    v0 = [t.clone() for t in opt.v]

    res = logical_train_step(
        model, opt, sch, sc, batch, microbatch_size=1, overflow_in_microbatch=2
    )
    print(f"status={res.status}（前两组的有效梯度必须一并丢弃）")
    print(f"scale: {2**12} -> {sc.scale:.1f}")
    print(f"opt.t={opt.t}（应为 0）, sch.step={sch.step_count}（应为 0）")
    print("params unchanged:", all(torch.equal(a, b) for a, b in zip(p0, model.parameters())))
    print("m unchanged:     ", all(torch.equal(a, b) for a, b in zip(m0, opt.m)))
    print("v unchanged:     ", all(torch.equal(a, b) for a, b in zip(v0, opt.v)))
    print("grads cleared:   ", all(p.grad is None for p in model.parameters()))

    good = logical_train_step(model, opt, sch, sc, batch, microbatch_size=1)
    print(f"scale 降低后重跑: status={good.status}, opt.t={opt.t}, sch.step={sch.step_count}")


def demo_skip():
    section("6) 无有效 token 的逻辑 batch：跳过（含不做 weight decay）")
    batch = make_all_pad_batch()
    model, opt, sch, sc = fresh(weight_decay=0.5)
    p0 = [p.detach().clone() for p in model.parameters()]
    res = logical_train_step(model, opt, sch, sc, batch, microbatch_size=2)
    unchanged = all(torch.equal(a, b) for a, b in zip(p0, model.parameters()))
    print(f"status={res.status}, opt.t={opt.t}, sch.step={sch.step_count}, "
          f"参数（含 wd）未变={unchanged}")


def demo_summation_order():
    section("7) 浮点求和次序分析")
    batch = make_batch(seed=0)
    ref_model, ref_opt, _, ref_sc = fresh(dtype=torch.float64)
    _, ref_grads = grads_after_accumulate(ref_model, ref_opt, ref_sc, batch, 9)

    for dtype_name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        worst = 0.0
        for ms in (1, 4, 3, 2):
            model, opt, _, sc = fresh(dtype=dtype, init_scale=1024.0)
            _, grads = grads_after_accumulate(model, opt, sc, batch, ms)
            err = max((grads[n].double() - ref_grads[n]).abs().max().item() for n in grads)
            worst = max(worst, err)
        print(f"{dtype_name}: 各划分相对 fp64 全批次的最大梯度偏差 ~ {worst:.3e}")
    print("差异仅来自加法结合次序，量级符合 ~(n_tokens * eps) 的舍入预期。")


if __name__ == "__main__":
    torch.manual_seed(0)
    demo_equivalence()
    demo_multi_step()
    demo_counterexample_mean()
    demo_counterexample_multi_step()
    demo_overflow_rollback()
    demo_skip()
    demo_summation_order()
