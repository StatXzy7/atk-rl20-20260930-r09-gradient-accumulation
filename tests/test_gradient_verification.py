"""核心梯度核验：双精度有限差分，以及 float32 求和次序误差的单独分析。"""
from __future__ import annotations

import torch

from microbatch_trainer import Batch
from .helpers import fresh, grads_after_accumulate, make_batch


def _tiny_batch(seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    vocab = 13
    ids = torch.randint(0, vocab, (3, 4), generator=g)
    labels = torch.randint(0, vocab, (3, 4), generator=g)
    mask = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0],
         [1.0, 1.0, 0.0, 0.0],
         [1.0, 0.0, 0.0, 0.0]]
    )
    return Batch(ids, mask, labels), vocab


def _objective(model, batch):
    """目标 = 有效 token 损失之和 / 有效 token 总数。"""
    loss_sum, n = model.summed_loss(batch.token_ids, batch.labels, batch.mask)
    return loss_sum / n


def test_central_finite_difference_matches_analytic():
    batch, vocab = _tiny_batch()

    # 用极小模型保证有限差分速度，且默认即双精度。
    from microbatch_trainer import TinyTokenModel, AdamW, LossScaler

    model = TinyTokenModel(vocab_size=vocab, hidden_dim=5, seed=42).double()
    opt = AdamW(model.parameters(), lr=1e-2)
    sc = LossScaler(init_scale=1.0, growth_interval=10**9)

    acc, grads = grads_after_accumulate(model, opt, sc, batch, microbatch_size=2)
    assert acc.status == "updated"
    assert acc.n_valid_tokens == 7

    eps = 1e-6
    checked = 0
    for name, p in model.named_parameters():
        analytic = grads[name]
        flat_p = p.detach().view(-1)
        flat_g = analytic.view(-1)
        # 每个参数张量抽查若干索引（小模型全查也可，取至多 4 个）。
        idxs = torch.linspace(0, flat_p.numel() - 1, steps=min(4, flat_p.numel())).long()
        for idx in idxs.tolist():
            orig = flat_p[idx].item()
            flat_p[idx] = orig + eps
            fp = _objective(model, batch).item()
            flat_p[idx] = orig - eps
            fm = _objective(model, batch).item()
            flat_p[idx] = orig
            numeric = (fp - fm) / (2 * eps)
            analytical = flat_g[idx].item()
            rel = abs(numeric - analytical) / max(1.0, abs(analytical))
            assert rel < 1e-7, (
                f"{name}[{idx}] numeric={numeric:.10g} analytic={analytical:.10g} rel={rel}"
            )
            checked += 1
    assert checked > 0


def test_float32_summation_order_analysis():
    """float32 下不同分组仅引入求和次序舍入；误差有界且远小于任何真实划分差异。"""
    batch = make_batch(seed=0)

    # 双精度参照。
    ref_model, ref_opt, _, ref_sc = fresh(dtype=torch.float64)
    _, ref_grads = grads_after_accumulate(ref_model, ref_opt, ref_sc, batch, 9)

    partitions = [9, 1, 4, 3, 2]
    worst_vs_ref = 0.0
    worst_pairwise = 0.0
    all_grads = {}
    for ms in partitions:
        model, opt, _, sc = fresh(dtype=torch.float32, init_scale=1024.0)
        _, grads = grads_after_accumulate(model, opt, sc, batch, ms)
        all_grads[ms] = grads
        err_vs_ref = max(
            (grads[n].double() - ref_grads[n]).abs().max().item() for n in grads
        )
        worst_vs_ref = max(worst_vs_ref, err_vs_ref)

    for i, a in enumerate(partitions):
        for b in partitions[i + 1:]:
            err = max(
                (all_grads[a][n] - all_grads[b][n]).abs().max().item()
                for n in all_grads[a]
            )
            worst_pairwise = max(worst_pairwise, err)

    print(f"\n[fp32] worst grad err vs fp64 ref   = {worst_vs_ref:.3e}")
    print(f"[fp32] worst pairwise partition err = {worst_pairwise:.3e}")

    # 小模型 + 33 个 token，float32 分组次序误差应在 1e-6 量级以内。
    assert worst_vs_ref < 1e-5
    assert worst_pairwise < 1e-5

    # 双精度下的次序误差应再小几个数量级，证明差异确实只是舍入。
    d_grads = {}
    for ms in partitions:
        model, opt, _, sc = fresh(dtype=torch.float64)
        _, grads = grads_after_accumulate(model, opt, sc, batch, ms)
        d_grads[ms] = grads
    d_pair = max(
        (d_grads[9][n] - d_grads[ms][n]).abs().max().item()
        for ms in partitions[1:]
        for n in d_grads[9]
    )
    print(f"[fp64] worst pairwise partition err = {d_pair:.3e}")
    assert d_pair < 1e-14
    assert d_pair < worst_pairwise / 1000.0
