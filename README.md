# 任意 microbatch 切分下的等价梯度累积

目标：把一个**逻辑 batch** 任意切成若干 microbatch（大小可不同、可为空、
有效 token 数可为零、最后一组可以不满），训练结果在舍入容限内与
**一次性全批次更新**完全等价。不借用任何成熟训练器的 accumulation 功能，
核心协议在 `microbatch_trainer.py` 中从头实现。

模型刻意选用无 dropout、无 BatchNorm/LayerNorm 的小型 MLP
（`tiny_model.py`），优化器为 AdamW：这两类组件本来就**不应**承诺
跨划分等价（随机性、批间统计差异），把它们排除后，唯一允许的差异来源是
浮点舍入。

## 1. 数学目标与推导

设逻辑 batch 的有效 token 集合为 `V`，`N = |V|`。目标函数是

```
L(θ) = (1/N) · Σ_{i∈V} ℓ_i(θ)        （有效 token 损失之和 / 有效 token 总数）
```

把逻辑 batch 任意划分成 microbatch `B_1..B_M`，记 `V_m = V ∩ B_m`，
`n_m = |V_m|`，则 `Σ_m n_m = N`。梯度恒等式：

```
∇L = (1/N) · Σ_{i∈V} ∇ℓ_i
    = (1/N) · Σ_m Σ_{i∈V_m} ∇ℓ_i
    = (1/N) · Σ_m g_m,   其中 g_m = ∇( Σ_{i∈V_m} ℓ_i )
```

`g_m` 正是 **microbatch 内未归一化的 token 损失和** 的梯度。于是正确协议为：

1. 每个 microbatch 对 `scale · (token 损失和)` 反传，累加进 `.grad`；
   全掩码（`n_m = 0`）的 microbatch 贡献恒为零梯度，直接跳过其 backward。
2. 全部累积完后**只做一次** `× 1/(scale·N)` 归一化。
3. 对归一化后的完整梯度算全局范数并裁剪（一次）。
4. `optimizer.step()` 与 `scheduler.step()` 各一次。

实数意义下该结果与划分方式无关（加法交换/结合律）；浮点意义下只差
求和次序引入的舍入（见第 4 节）。

### 为什么不能等权平均每个 microbatch 的均值

`(1/M) Σ_m (1/n_m) Σ_{i∈V_m} ℓ_i` 给 token `i` 的权重是
`1/(M·n_{m(i)})` 而不是 `1/N`：它优化的是**另一个目标函数**，
仅当所有 `n_m` 相等时才偶然相同。正确做法是按 `n_m/N` 加权
（`test_token_weighted_mean_is_right`），而对 token-sum 反传再统一除 N
正是该加权的无除法实现。实测（划分 [4,4,2]，各 microbatch 有效 token
11/11/10）mean-of-means 梯度相对误差 **5.0e-2**，远超任何舍入容限
（`test_mean_of_means_is_wrong_for_uneven_token_counts`）。

### 为什么 loss scaling 是无损的

反传对标量种子梯度是线性的：`∇(S·f) = S·∇f`，且每个 VJP 中间量都随之
精确缩放。当 `S` 取 2 的幂时，二进制浮点乘除 `S` 只改指数、不改尾数，
`(S·a + S·b) = S·(a+b)` 逐项精确，因此"缩放 → 累加 → 乘 1/(S·N)"
与未缩放路径**逐位一致**（不发生上/下溢时）。这也是溢出后可以用减半的
scale 安全重试的依据：实测重试结果与从未溢出的参照 max diff = 0.0
（`test_overflow_mid_accumulation_rolls_back_everything`）。

### 为什么 clipping 只能在最后做

全局范数裁剪是非线性算子：`clip(g₁+g₂) ≠ clip(g₁)+clip(g₂)`。
对每个 microbatch 分别裁剪再相加与对完整梯度裁剪一次不可交换
（`test_clipping_each_microbatch_is_not_equivalent`，相对误差 > 1e-2）。
梯度范数用 float64、按固定参数顺序计算，使裁剪系数本身不引入额外的
划分相关性。

### AdamW / 调度器 / weight decay 的原子性

AdamW 的 step 计数、一阶/二阶矩、解耦 weight decay（`θ ← θ(1−ηλ)`）
全部只在 `optimizer.step()` 内推进；学习率只在 `scheduler.step()` 内
推进。协议把这两个调用放在**唯一**的变更点上，且以
`(无溢出) ∧ (N>0)` 为门控：每次真正成功的逻辑更新恰好各前进一步，
失败或跳过时一步都不走。weight decay 不需要特殊处理——不 step 自然不
decay（`test_zero_token_logical_batch_skips_update_and_weight_decay`
用 `weight_decay=0.5` 逐位验证参数未动）。

## 2. 溢出回滚协议（loss scaling 状态机）

每次逻辑更新是一个**原子**状态机：

```
prepare_grads:
  zero_grad → 逐 microbatch: backward(scale·token_sum) → 检查全部梯度有限
    ├─ 发现 inf/NaN → zero_grad（清空全部累积梯度）
    │                 scale ← max(scale·backoff, min_scale)
    │                 连续成功计数清零；其它一概不动 → 返回 OVERFLOW
    ├─ N = 0        → zero_grad；不 step、不 decay、不调度 → SKIPPED_NO_TOKENS
    └─ 成功         → 梯度 ×1/(scale·N) → 全局范数裁剪(一次)
                      → optimizer.step() → scheduler.step()
                      → 连续成功计数+1（达到 growth_interval 则 scale 回涨）
```

关键点：溢出可能出现在累积**中途**（第 k 个 microbatch），此时前 k−1 个
microbatch 的梯度已经污染 `.grad`，但参数、Adam 矩、step 计数、调度器
从未被触碰——`step()` 尚未执行，回滚因此是平凡的，不需要任何快照恢复，
只需清空梯度并调整 scale。测试在第二个 microbatch 注入 `inf`，验证回滚
边界：参数/Adam 状态/调度器与更新前**逐位**一致，`.grad` 全部清空，
scale 减半；用同一逻辑 batch 以新 scale 重试后与干净参照逐位一致
（`tests/test_overflow_rollback.py`）。

## 3. 零有效 token 的逻辑 batch

`N = 0` 时目标函数是 0/0，未定义。本实现将其**定义为跳过更新**：
不反传、不 step（因此不 weight decay）、不推进调度器、不改 scale。
注意区分两种"空"：逻辑 batch 中**部分** microbatch 全掩码是正常的
（贡献零梯度，更新照常，见 `test_partially_empty_logical_batch_still_updates`
与划分 `[2,3,5]`/`[4,0,6]` 的等价性测试）；**全部**无效才跳过。

## 4. 浮点求和次序分析

不同划分只改变加法的**次序**（microbatch 内归约 + 跨 microbatch 累加），
不承诺逐位相等，只承诺舍入容限内等价。实测梯度相对偏差（对全批次参照，
`tests/test_fp_order.py` / `python3 demo.py`）：

| 精度    | 最大相对梯度偏差（6 种划分） |
|---------|------------------------------|
| float64 | ≤ 7.7e-17                    |
| float32 | ≤ 6.2e-8                     |

两者相差约 10⁹ ≈ 2²⁹，正好是 float32→float64 多出的尾数位数——差异随
尾数宽度缩放，证明它是**舍入噪声**而非系统误差。判别标准：正确协议的
误差随精度指数级消失；错误协议（下节）的误差在任何精度下都保持在
10⁻² 量级。

## 5. 反例（实测，`tests/test_counterexamples.py`）

| 错误做法 | 后果 | 实测 |
|----------|------|------|
| 等权平均各 microbatch 均值 | 优化了另一个目标（token 权重 1/(M·n_m)） | 梯度相对误差 5.0e-2 |
| 每个 microbatch 都 `step()` | Adam 预条件、调度器、weight decay 作用 M 次 | 一次更新后参数最大差 1.9e-1 |
| 每个 microbatch 分别裁剪 | 非线性算子不可交换 | 梯度相对误差 > 1e-2 |
| 溢出后已部分推进 | 需要快照才能回滚，协议上不可接受 | 本实现由构造排除（step 是唯一变更点） |

## 6. 文件与运行

- `microbatch_trainer.py` — 累积 / 缩放 / 裁剪 / 原子回滚协议
- `tiny_model.py` — 小模型、合成数据、token-sum 损失、任意切分工具
- `tests/test_equivalence.py` — 7 种划分（含空 microbatch、零 token
  microbatch、不满的末组）× 梯度 / 两次更新全状态 / 裁剪激活 的等价性
- `tests/test_overflow_rollback.py` — 中途注入溢出、回滚边界、scale
  衰减/回涨、零 token 跳过
- `tests/test_gradcheck.py` — 中心有限差分（float64）核验累积梯度
- `tests/test_counterexamples.py` — 上述反例
- `tests/test_fp_order.py` — 求和次序影响的定量界
- `demo.py` — 一键打印本 README 中的所有实测数字

```bash
python3 -m pytest tests/ -q     # 37 个测试
python3 demo.py                 # 复现实测数字
```
