"""测试共享工具：小规模数据、可复现状态、快照与比较。"""
from __future__ import annotations


import torch

from microbatch_trainer import (
    AdamW,
    Batch,
    LossScaler,
    TinyTokenModel,
    WarmupCosineScheduler,
)
from microbatch_trainer.trainer import accumulate_gradients

VOCAB = 37
HIDDEN = 24
SEQ = 6
N_ROWS = 9
# 每行有效 token 数刻意不均匀，最后一行为全 pad（合计 33）。
LENGTHS = [6, 4, 5, 3, 6, 2, 1, 6, 0]

# 多种不均匀切分：整批 / 逐行 / 含末尾空组 / 不整除 / 含中间小计数。
PARTITIONS = [
    (9, [33]),
    (1, [6, 4, 5, 3, 6, 2, 1, 6, 0]),
    (4, [18, 15, 0]),
    (3, [15, 11, 7]),
    (2, [10, 8, 8, 7, 0]),
]


def make_batch(seed: int = 0, n_rows: int = N_ROWS, seq_len: int = SEQ, vocab: int = VOCAB) -> Batch:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, vocab, (n_rows, seq_len), generator=g)
    labels = torch.randint(0, vocab, (n_rows, seq_len), generator=g)
    mask = torch.zeros(n_rows, seq_len, dtype=torch.float32)
    for i, length in enumerate(LENGTHS[:n_rows]):
        if length:
            mask[i, :length] = 1.0
    return Batch(ids, mask, labels)


def make_all_pad_batch(seed: int = 1, n_rows: int = 4, seq_len: int = SEQ) -> Batch:
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, VOCAB, (n_rows, seq_len), generator=g)
    labels = torch.randint(0, VOCAB, (n_rows, seq_len), generator=g)
    mask = torch.zeros(n_rows, seq_len, dtype=torch.float32)
    return Batch(ids, mask, labels)


def fresh(
    dtype=torch.float64,
    lr: float = 1e-2,
    weight_decay: float = 0.02,
    init_scale: float = 1024.0,
    model_seed: int = 123,
    warmup: int = 2,
    total_steps: int = 12,
):
    model = TinyTokenModel(vocab_size=VOCAB, hidden_dim=HIDDEN, seed=model_seed).to(dtype)
    opt = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = WarmupCosineScheduler(opt, warmup_steps=warmup, total_steps=total_steps)
    scaler = LossScaler(init_scale=init_scale, growth_interval=10**9)
    return model, opt, sch, scaler


def grads_after_accumulate(model, opt, scaler, batch, microbatch_size, max_norm=None, inject=None):
    acc = accumulate_gradients(
        model, opt, scaler, batch, microbatch_size,
        max_norm=max_norm, overflow_in_microbatch=inject,
    )
    grads = {}
    if acc.status == "updated":
        grads = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.grad is not None
        }
    return acc, grads


def full_snapshot(model, opt, sch, scaler):
    return {
        "params": [p.detach().clone() for p in model.parameters()],
        "m": [t.clone() for t in opt.m],
        "v": [t.clone() for t in opt.v],
        "t": opt.t,
        "lr": opt.lr,
        "sch_step": sch.step_count,
        "scale": scaler.scale,
    }


def assert_snapshot_unchanged(snap, model, opt, sch, scaler, atol=0.0, check_scale=False):
    for a, p in zip(snap["params"], model.parameters()):
        assert torch.equal(a, p.detach())
    for a, b in zip(snap["m"], opt.m):
        assert torch.equal(a, b)
    for a, b in zip(snap["v"], opt.v):
        assert torch.equal(a, b)
    assert opt.t == snap["t"]
    assert opt.lr == snap["lr"]
    assert sch.step_count == snap["sch_step"]
    if check_scale:
        assert scaler.scale == snap["scale"]


def max_abs(grads_a, grads_b):
    return max((grads_a[n] - grads_b[n]).abs().max().item() for n in grads_a)


def named_params_max_diff(model_a, model_b):
    return max(
        (pa - pb).abs().max().item()
        for pa, pb in zip(model_a.parameters(), model_b.parameters())
    )


def states_max_diff(opt_a, opt_b):
    m = max((a - b).abs().max().item() for a, b in zip(opt_a.m, opt_b.m))
    v = max((a - b).abs().max().item() for a, b in zip(opt_a.v, opt_b.v))
    return m, v
