# 任意 microbatch 切分的等价梯度累积：推导、协议与反例

## 1. 符号与数学目标

设一个**逻辑 batch** 被任意切成若干 microbatch。对其中第 $i$ 个 microbatch：

- $\mathcal{T}_i$：该组内有效 token（mask=1）的下标集合，$n_i = |\mathcal{T}_i|$；
- $\ell_j(\theta)$：第 $j$ 个有效 token 对模型参数 $\theta$ 的逐 token 损失
  （这里是交叉熵）。允许 $n_i = 0$，也允许最后一个组样本数更少。

整个逻辑 batch 的有效 token 总数

$$
T = \sum_i n_i.
$$

需求规定的训练目标是**逻辑 batch 内所有有效 token 的损失之和除以有效 token 总数**：

$$
L(\theta) = \frac{1}{T}\sum_i\sum_{j\in\mathcal{T}_i}\ell_j(\theta).
$$

由微分线性性，目标梯度为

$$
\nabla L(\theta)
= \frac{1}{T}\sum_i\sum_{j\in\mathcal{T}_i}\nabla\ell_j(\theta).
\tag{1}
$$

式 (1) 与“怎么分组、分多少组、哪组为空、最后一组是否不满”**完全无关**：
分组只决定求和项的书写次序，不改变数学和。这是整个等价性的根据。

## 2. 正确的 loss 缩放

前向时每个 microbatch 返回的是**未归一化的求和**（关键：不能在组内先除 $n_i$）：

$$
s_i = \sum_{j\in\mathcal{T}_i}\ell_j,\qquad n_i=0\ \text{时跳过该组}.
$$

AMP 风格的 loss scale 记为 $c$。统一按

$$
\boxed{\,\text{backward scale}_i = \frac{c}{T}\,}
$$

对每组反传。累积到 `.grad` 上的是

$$
g^{\text{scaled}} = \sum_i \frac{c}{T}\sum_{j\in\mathcal{T}_i}\nabla\ell_j
= c\,\nabla L.
$$

**所有组累积完成后**，再统一反缩放一次：

$$
g = g^{\text{scaled}}/c = \nabla L.
$$

要点：

1. 分母是**全逻辑 batch** 的 $T$，在任何前向之前即可由 mask 统计得到；
   不能用运行时“已见 token 数”之类的在线量，也不能用 microbatch 数 $K$。
2. $n_i=0$ 的组既不贡献 $s_i$（跳过），也不改变 $T$；因此不会出现除零。
3. 反缩放在累积结束后做一次，而不是每组各做一次（后者会引入额外的逐组舍入）。
4. 取 $c$ 为 2 的幂时，`*c` 与 `/c` 在二进制浮点下大多精确，双精度下
   不同划分能达到逐位/近逐位一致（实测参数差 $\sim10^{-17}$）。

## 3. 全局范数裁剪的位置

先得到正确归一化、反缩放后的梯度，再把**全部参数张量**作为一个整体计算

$$
\|g\|_2 = \sqrt{\sum_p \|g_p\|_2^2},
\qquad
g_p \leftarrow \frac{\max\text{-norm}}{\max(\|g\|_2,\max\text{-norm})}\,g_p.
$$

因为各划分在裁剪前得到的是同一个 $g$（仅求和次序差异），它们得到同一个
$\|g\|_2$ 和同一个裁剪系数；裁剪后仍然一致。裁剪必须在累积与反缩放**之后**、
优化器 `step` **之前**，且每个逻辑更新只做一次。若每组各自裁剪，则是非线性
操作，不同划分不再等价。

## 4. 优化器 / 调度器：一次逻辑更新恰好前进一步

裁剪后调用一次 AdamW（解耦 weight decay，论文式更新）：

$$
m_t=\beta_1 m_{t-1}+(1-\beta_1)g,\quad
v_t=\beta_2 v_{t-1}+(1-\beta_2)g^{\odot2},
$$

$$
\theta \leftarrow \theta-\eta_t\!\left(
\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}+\lambda\theta
\right),
\qquad
\hat m_t=\frac{m_t}{1-\beta_1^t},\ \hat v_t=\frac{v_t}{1-\beta_2^t}.
$$

由于各划分在 `step` 前梯度一致（舍入容限内），且 $m_{t-1},v_{t-1},\eta_t,\lambda$
都相同，更新后的 $\theta,m_t,v_t$ 一致。学习率调度器与优化器步数一一对应：
**一次真正成功的逻辑更新**才让优化器 $t$、调度器、Adam 一阶/二阶矩前进一步；
解耦 weight decay 也只在这一步发生。

本实现中调度器约定当前更新使用 $\eta(t)$（$t$ 为 1 起的更新序号）：
warmup 段 $\eta(t)=\eta_{\max}t/t_{\text{warmup}}$，之后余弦衰减。
因此第一次更新学习率非零。

## 5. Loss scaling 协议与失败原子性

每个 microbatch 反传后立即检查该参数全部 `.grad` 是否有限
（`isfinite` 全为真）。若发现 inf/NaN：

1. **立即终止**剩余 microbatch；
2. 丢弃**全部**累积梯度（`set_to_none=True`），包括之前组已经累积的有效部分；
3. **不**反缩放、**不**裁剪、**不**调用优化器 `step`、**不**推进调度器；
4. 参数、$m$、$v$、优化器步数、调度器位置、当前 lr 全部保持更新前；
5. 唯一允许的变化是 scaler：动态/固定地按 `backoff_factor`（默认 0.5）下调 $c$。

这保证“一次逻辑更新”是**原子**的：要么完整成功并推进一次，要么什么都没发生
（除 scale 外）。scale 下调后用相同数据重跑，在梯度恢复有限时成功，
随后才进行第 4 节的唯一一次优化器与调度器步进。成功时 scaler 记录一次成功，
连续成功达到 `growth_interval` 后上调一次。

## 6. 无有效 token 的逻辑 batch：跳过

若 $T=0$，目标 (1) 无定义。按需求将其定义为**跳过更新**：不前向反传、不裁剪、
不调用优化器/调度器——特别地，**解耦 weight decay 也不执行**（参数不发生
$(1-\eta\lambda)$ 缩放），优化器步数与调度器均不前进。这与“溢出失败”是
两种不同状态（`skipped` 与 `overflow`），但都不构成一次成功的逻辑更新。

## 7. 浮点求和次序：何时能“逐位等价”

式 (1) 的结合次序随划分而变：全批次是一次大求和；逐 microbatch 是分组部分和
再相加。浮点加法不满足结合律，所以

$$
\text{fl}\Big(\sum_{\text{full}}\big)\ne
\text{fl}\Big(\sum_i\text{fl}(\sum_{j\in i}\cdot)\big)
$$

可能有 $O(T\varepsilon)$ 的相对差异。这不是算法误差，而是舍入。实测
（见 `demo.py` 与 `tests/test_gradient_verification.py`）：

| 精度 | 各划分相对 fp64 全批次的最大梯度偏差（本小模型，T=33） |
|---|---|
| float64 | $\sim 10^{-17}$ |
| float32 | $\sim 10^{-8}$ |

缩小误差的工程手段：双精度累积、scale 取 2 的幂、累积结束后只反缩放一次。
等价性断言在 float64 下用 $10^{-11}$ 容限，float32 用 $10^{-5}$ 容限，
并把“次序影响”单独作为一项分析而非混进等价性结论。

## 8. 两个反例（为什么常见写法错）

### 反例 A：microbatch 均值的等权平均

先在每组内部取均值 $s_i/n_i$，再对 $K$ 个组等权平均，得到的梯度是

$$
g^{\text{naive}}=\frac{1}{K}\sum_{i:n_i>0}\frac{1}{n_i}
\sum_{j\in\mathcal{T}_i}\nabla\ell_j.
$$

与式 (1) 相比，token $j\in\mathcal{T}_i$ 的权重是 $1/(Kn_i)$ 而不是 $1/T$。
只有当所有非空组 $n_i$ 相等（且空组处理一致）时两者才碰巧相等；只要各组
有效 token 数不同，token 权重就被扭曲。测试数据各组计数为
`[6,4,5,3,6,2,1,6,0]`，一次（仅此一次）优化器更新后参数最大差约 $10^{-2}$。
另构造“每组计数恰好相等”的数据，两者差回到 $10^{-11}$，验证了该反例的边界。

### 反例 B：每个 microbatch 各更新一次参数

即使每组内部用该组自己的 $n_i$ 正确归一化，逐组 `step` 会导致：

- 优化器被推进多次：$m_t,v_t$ 的偏差修正、$\eta_t$ 的调度都不同；
- 后续组的梯度是在**已被前面组改变**的 $\theta$ 上算出的；
- 解耦 weight decay 被复利施加多次。

这些非线性使“多步小更新”不等于“一步大更新”。测试中 8 个非空组触发 8 次
优化器/调度器步进，与一次全批次更新的参数最大差约 $6\times10^{-2}$。
因此禁止“每组更新一次参数后声称等价”。

## 9. 模型选择

为排除“本来就不应承诺等价”的因素，`TinyTokenModel` 仅含
Embedding / Linear / GELU（无 dropout、无 BatchNorm/LayerNorm 的批统计、
无随机性），损失对逐位置 token 计算并由 mask 屏蔽 pad。AdamW、调度器、
loss scaler 均在本仓库内实现，不使用成熟训练器的 accumulation 功能，
以确保每一步的精确语义可被测试直接断言。

## 10. 关键结论清单

- 统一权重：每组反传乘 $c/T$，$T$ 是**全逻辑 batch** 有效 token 总数；
  组内损失只求和、不归一化；空组跳过；最后一组可不满。
- 反缩放一次：全部累积、确认有限后再统一除 $c$。
- 裁剪一次：正确归一化后的全局 norm 裁剪，位于 `step` 前。
- 步进一次：成功的逻辑更新 ⇒ AdamW（含 wd）、$t$、调度器各前进一步。
- 失败原子性：累积中任何非有限梯度 ⇒ 清空全部累积梯度、回滚边界内无任何
  参数/矩/步数/调度器变化，仅下调 scale。
- $T=0$：跳过整个更新，包括不做 weight decay。
