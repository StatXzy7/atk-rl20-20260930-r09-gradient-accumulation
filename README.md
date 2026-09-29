# 任意 microbatch 切分的等价梯度累积

实现一个训练步：一个逻辑 batch 可被任意切成 microbatch（各组有效 token 数
不同、可为零、最后一组可不满），数学目标是

```
逻辑 batch 内全部有效 token 的损失之和 / 有效 token 总数 T
```

任意切分在舍入容限内与一次全批次更新等价。**不**使用成熟训练器的
accumulation 功能；AdamW、warmup+余弦调度器、loss scaler 均在本仓库内实现。

## 正确协议（详见 `DERIVATION.md`）

1. 每个 microbatch 的有效 token 损失只**求和** $s_i$，按 `c / T` 反传
   （`c` 为 loss scale，`T` 为全逻辑 batch 有效 token 总数，空组跳过）。
2. 全部组累积完成、确认有限后，统一反缩放一次 `/c`。
3. 在正确归一化后的梯度上做一次全局 norm clipping，再调用一次 AdamW。
4. 一次成功的逻辑更新：优化器一步（含解耦 weight decay）、Adam $m$/$v$、
   步数、学习率调度器各前进一步。
5. 累积中发现非有限梯度：整次更新原子失败——清空全部累积梯度，参数/矩/步数/
   调度器不推进，仅下调 scale。
6. `T=0` 的逻辑 batch：跳过，包括不执行 weight decay。

## 目录

- `microbatch_trainer/model.py`：无 dropout / 无批统计的小型逐 token 模型与 `Batch`。
- `microbatch_trainer/trainer.py`：`logical_train_step`（正确协议）、
  `accumulate_gradients`（只累积不更新，便于测试）、两个反例
  `naive_mean_microbatch_train_step`、`per_microbatch_update_train_step`。
- `microbatch_trainer/optimizer.py`：解耦 weight decay 的 AdamW。
- `microbatch_trainer/scheduler.py`：线性 warmup + 余弦。
- `microbatch_trainer/scaler.py`：loss scale 协议（退避/增长）。
- `DERIVATION.md`：完整推导、反例分析与浮点求和次序讨论。
- `tests/`：等价性、溢出回滚边界、空 batch 跳过、反例、双精度有限差分、
  fp32/fp64 求和次序分析。
- `demo.py`：端到端演示。

## 运行

```bash
python -m pytest tests/ -q
python demo.py
```

依赖：`torch`（CPU 即可）、`pytest`。测试主用 float64 断言严格等价
（容限 1e-11），float32 单独分析求和次序（容限 1e-5）。
