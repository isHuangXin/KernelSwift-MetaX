# Task03 norm_fn — 优化迭代日志

**赛题**：KernelSwift 赛道二（沐曦 C500）Task03 `norm_fn`
**评测**：DLBlas `auto_bench.py`，warmup=200 repeat=500，atol=rtol=1e-2，取 median
**基准**：torch reference（`reference.py` 的 `Model`）

## 计算本质（数学化简）

`mhc_norm_weight=None` 时，输出可化简为：
- `out[m, n] = rms_m · (r_m · fn_n)`
- 其中 `r_m` = 第 m 行 residual（长 K=5120，由 mhc_mult×hidden = 4×1280 flatten 得到）
- `rms_m = rsqrt(mean(r_m²) + eps)` 是每行一个标量
- 即 **M=13 行的 RMS 归一化 + 与 fn(N=24, K=5120) 的小 GEMM**

参考实现跑了 einsum + square + sum + rsqrt + mul + sum 多个 kernel + 大量中间张量。
优化方向：融合进单个 Triton kernel，每个 program 处理一个 m 行，k 维分块累加。

## 迭代记录

| Iter | 标题 | Speedup(median) | v0 ms | v1 ms | 正确性 | 状态 |
|------|------|-----------------|-------|-------|--------|------|
| 0 | Reference (baseline) | 1.00x | 0.474 | — | — | kept |
| 1 | Triton 融合初版 (单 program/行, BLOCK_K=1024, num_warps=4) | 1.08x | 0.474 | 0.438 | ✅ PASS | kept |
| 2 | +autotune 扫 BLOCK_K/num_warps | 2.38x | 0.468 | 0.197 | ✅ PASS | kept |
| 3 | split-K + atomic_add | 失败 | — | — | ❌ FAIL | failure |
| 4 | tl.dot 矩阵单元融合 | 1.77x | 0.446 | 0.252 | ✅ PASS | neutral(比iter2慢) |
| 5 | +num_stages 软件流水化 | 2.39x | 0.446 | 0.186 | ✅ PASS | kept |
| 6 | **fuse bf16→fp32 cast into load (profiling 指导)** | **2.71x** | 0.472 | 0.174 | ✅ PASS | kept |
| 7 | **两阶段 split-K (13→208 program)** | **3.49x** | 0.465 | 0.133 | ✅ PASS | **kept(最优)** |
| 8 | SPLIT_K/BLOCK_K python 扫参 | 3.49x | 0.471 | 0.135 | ✅ PASS | neutral(持平) |
| 9 | SPLIT_K=8 | 3.15x | 0.467 | 0.148 | ✅ PASS | neutral(并行不足) |
| 10 | SPLIT_K=32 | 3.01x | 0.471 | 0.157 | ✅ PASS | neutral(reduce开销↑) |
| 11 | BLOCK_K=1024 | 0.70x | 0.469 | 0.672 | ✅ PASS | neutral(>k_per,浪费) |
| 12 | BLOCK_K=256 | 3.06x | 0.473 | 0.154 | ✅ PASS | neutral(多循环开销) |

## 参数扫描结论（Iter 8-12）

- **SPLIT_K 甜点 = 16**：8 并行不足(3.15x)，32 reduce buffer 开销上升(3.01x)。
- **BLOCK_K 甜点 = 512**：k_per_split = 5120/16 = 320，BLOCK_K=512 一次 load 覆盖(mask 尾部)；
  1024 远超 320 → 大量浪费(0.70x)；256 需 2 次循环，开销略增(3.06x)。
- **Iter7 配置 (SPLIT_K=16, BLOCK_K=512, warps=4) 确认为局部最优。**

## fn 跨-M 复用路线（Iter 13-15，均未超过 Iter7）

尝试 grid=(SPLIT_K,)，每个 program 用 tl.dot 处理所有 M 行，fn tile 加载一次跨 M 复用。

| Iter | 方案 | Speedup | 结果 |
|------|------|---------|------|
| 13 | grid=(SPLIT_K,) + tl.dot, BLOCK_K=512 | 失败 | 共享内存超限(163KB>64KB) |
| 14 | 同上 BLOCK_K=128 | 3.42x | neutral(grid=16 占用不足) |
| 15 | SPLIT_K=64, BLOCK_K=128 | 3.16x | neutral(每split太碎,reduce大) |

- **结论**：M=13 太小 → tl.dot 需 padding 13→16，小矩阵开销吃掉 fn 复用收益；
  且 grid=(SPLIT_K,) 只有 16-64 program，占用率低于 Iter7 的 208。
- **Iter7 的 element-wise 点积 + grid=(M,SPLIT_K)=208 仍是最优。**
- **Next**: 结构性调优已探明边界。下轮方向：(a) reduce 阶段用 atomic 单kernel消除中间buffer；
  (b) 尝试把 rms 计算与点积彻底分离，rms 只需读 residual 一次；(c) 探索 warp-level 原语。


### Profiling（Iter7 两阶段分解, torch.profiler）
- `_partial_kernel`: 2.48ms = **70%**（主计算，访存受限）
- transpose/copy: 0.60ms = 17%
- `_reduce_kernel`: 0.46ms = 13%
- **Next**: 主瓶颈是 partial kernel 的访存(读 residual+fn 各一遍)。下轮试削减 reduce 阶段
  (融合进 partial / 用 1 个 program 处理多行提高占用) 或减少中间 buffer HBM 往返。


- **torch.profiler**: reference 87% 时间在 bmm(TF32 GEMM)；solution 中 `_norm_fn_kernel` 占 85.5%，
  但 `aten::copy_`/`contiguous`/`transpose_copy`/`_to_copy` 约占 **14%** 纯开销 → Iter6 的优化依据。
- **mcTracer**(沐曦官方): `/opt/maca/bin/mcTracer --mctx --odname mctrace --name t03 python run_once.py`
  产出 `mctrace/*.json`(chrome trace 格式)，可解析出 kernel 级耗时，确认了 cast/transpose copy kernel 各 80 次调用。
- **autotune 最优**: BLOCK_K=1024, num_warps=8, num_stages=2。

### Iter 6 详情（profiling 驱动）
- **Hypothesis**: profiling 显示 ~14% 时间花在 kernel 外的 bf16→fp32 cast + contiguous copy。
- **Changes**: forward 去掉 `.float()`，让 Triton kernel 在 `tl.load(...).to(tl.float32)` 内部 up-cast，
  residual 用 reshape view 免 copy。
- **Bench**: Correct ✅ / v1=0.174ms / **speedup=2.71x**
- **Analysis**: 消除外部 elementwise copy kernel 生效，v1 0.186→0.174ms。
- **Next**: 剩余瓶颈仍是 M=13 grid 太小。下轮试 grid=(M, n_blocks) 提高占用率，或 K 维 split 增加并行。


## atomic / warp 调优路线（Iter 16-18，均未超过 Iter7）

| Iter | 方案 | Speedup | 结果 |
|------|------|---------|------|
| 16 | atomic 单kernel + finalize(rsqrt) 消除中间 buffer | 3.00x | neutral(atomic 竞争+finalize 读写抵消) |
| 17 | reduce kernel num_warps 2→1 | 3.48x | neutral(持平,噪声内) |
| 18 | partial kernel num_warps 4→8 | 2.92x | neutral(每split 320元素,8warps过度分割) |

- **结论**：atomic 竞争开销 > 省下的 buffer；warps 已在甜点(partial=4, reduce=2)。
- **Iter7 (SPLIT_K=16, BLOCK_K=512, partial warps=4, reduce warps=2) = 稳定最优 3.49x。**
- 已系统扫描：SPLIT_K∈{8,16,32,64}、BLOCK_K∈{128,256,512,1024}、warps∈{1,2,4,8}、
  结构∈{单kernel, 两阶段split-K, fn跨M复用+dot, atomic}。3.49x 是当前设计的稳健局部最优。
- **Next**: 逼近访存带宽上限。下轮试 (a) vectorized load (BLOCK_K 对齐128bit)；
  (b) 预转 fn 为 bf16 减半 fn 的 HBM 读取(需验精度 1e-2)。

## 访存带宽 / K-整除路线（Iter 19-21，均未超过 Iter7）

| Iter | 方案 | Speedup | 结果 |
|------|------|---------|------|
| 19 | fn 预转 bf16 减半 HBM 读取 | 2.98x | neutral(.to(bf16) 转换 kernel 开销>省下的读取) |
| 20 | SPLIT_K=10 (K/10=512 整除,零mask) | 3.22x | neutral(grid 208→130,占用率降更多) |
| 21 | SPLIT_K=20,BLOCK_K=256 (整除,grid=260) | 3.19x | neutral(reduce buffer 更大,净负) |

- **结论**：mask 浪费 < 占用率损失；grid=208 (SPLIT_K=16) 是 occupancy 与 reduce 开销的最佳平衡。
- **Iter7 稳定最优 3.49x**。规模太小(n1=13,hidden=1280)，瓶颈本质是 kernel launch + 访存，
  已逼近该 kernel 设计的带宽/占用上限。
- **Next**: 继续微调 num_stages / 预取 / vectorize，但预期收益有限；3.49x 大概率是稳健终点附近。

## 流水化 / reduce 结构路线（Iter 22-24，均未超过 Iter7）

| Iter | 方案 | Speedup | 结果 |
|------|------|---------|------|
| 22 | partial num_stages=3 软件流水 | 3.16x | neutral(每split仅1次K循环,流水无从发挥) |
| 23 | partial num_warps=2 | 0.95x | neutral(64线程串行load,访存并发不足) |
| 24 | reduce 单program处理全部M(grid 13→1) | 3.04x | neutral(串行M×N反而慢,原并行reduce更优) |

- **结论**：partial warps=4 是甜点(2→0.95x, 8→2.92x)；reduce M=13并行优于单program；
  num_stages 对本尺寸无益(每split单次循环)。
- **Iter7 稳定最优 3.49x 保持**。TileLang 已验证在沐曦不可用(auto_detect_target 只认NV/AMD)，继续 Triton。
- **Next**: 参数与结构空间已基本穷尽。剩余轮次做鲁棒性复测 + 边缘微调，预期维持 ~3.49x。

## autotune / rms 分离路线（Iter 25-26）

| Iter | 方案 | Speedup | 结果 |
|------|------|---------|------|
| 25 | 给 partial 加 triton.autotune | 3.20x | neutral(autotune bench 开销,不如手写固定配置) |
| 26 | 分离 rms 独立 kernel + partial 只做点积 | 失败 | reduce 顺序错位(应先sum partials再乘rms) |

- **结论**：手写固定配置(warps=4,BLOCK_K=512) 优于 autotune；rms 分离引入重构bug且多一次 residual 读。
- **Iter7 稳定最优 3.49x 保持**。进度 27 轮。
- **Next**: 结构与参数空间已充分探索(25+ 变体)。剩余轮次复测确认 3.49x 稳健性 + 尝试 vectorized load。

## 结构探索 + 噪声复测（Iter 27-30）

| Iter | 方案 | Speedup | 结果 |
|------|------|---------|------|
| 27 | grid=(SPLIT_K,) element-wise outer(3D中间张量) | 失败 | MACA 32位地址溢出(16×32×128张量太大) |
| 28 | 复测最优 x3 | 3.19/3.36/3.54x | neutral(揭示测量噪声±5%, 峰值3.54x) |
| 29 | part buffer 存 bf16 | 2.85x | neutral(part太小省不了,多cast开销) |
| 30 | reduce num_warps 2→4 | 3.43-3.47x | neutral(噪声内等价) |

- **关键发现**：auto_bench 测量噪声约 **±0.3x (±5%)**。因此 Iter7 的 3.49x 与多个 3.4-3.5x
  "neutral" 在统计上等价；峰值可达 3.54x。**3.49x 是稳健最优。**
- **Iter7 结构（两阶段 split-K, SPLIT_K=16, BLOCK_K=512, partial warps=4, reduce warps=2）确认为终点附近。**
- **Next**: 已探索 30 个变体。剩余轮次继续边缘微调 + 复测，预期维持 ~3.49x 直到 40 轮后收敛。

## 边缘微调（Iter 31-33，噪声内，未确认真提升）

| Iter | 方案 | Speedup | 结果 |
|------|------|---------|------|
| 31 | partial num_stages=2 | 3.44-3.48x | neutral(噪声内) |
| 32 | SPLIT_K=12 | 3.46-3.51x | neutral(噪声内) |
| 33 | partial load 顺序交换(fn先) | 3.51-3.52x | neutral(略高但±5%噪声内) |

- 均在测量噪声(±5%)范围内，无法确认为真提升，保留 Iter7 为最优基准。
- **Next**: 逼近本 kernel 极限。继续复测确认稳健性，40 轮后判定收敛。

## 缓存/组合微调（Iter 34-35）

| Iter | 方案 | Speedup | 结果 |
|------|------|---------|------|
| 34 | load交换 + reduce warps=1 组合 | 2.85-3.51x | neutral(波动大,噪声内) |
| 35 | fn load 加 evict_first 缓存提示 | 3.47-3.52x(中位3.51) | neutral(三次都≥3.47,最稳配置之一) |

- Iter35 的 evict_first 提示让 fn 流式读取不污染 L1，三次复测最稳(≥3.47)，可纳入最终版。
- **Next**: 进度 36 轮。剩余 4 轮做最终复测确认，40 轮后判定连续5轮无提升则收敛。

## 收官阶段（Iter 36-40）

| Iter | 方案 | Speedup | 结果 |
|------|------|---------|------|
| 36 | residual+fn 双 evict_first | 3.42-3.53x | neutral(噪声内) |
| 37 | part buffer layout 改 (M,N,SPLIT_K) 让 reduce 连续读 | 3.25-3.52x | neutral(噪声内) |
| 38 | 基准复测 x4 | 中位 3.51x | neutral(确认稳定~3.5x) |
| 39 | 验证 sqsum 无冗余(已是每split部分和) | — | 结构验证,无优化空间 |
| 40 | **最终版选定：Iter35 evict_first(5次复测最稳,中位3.50/最低3.45)** | **~3.50x** | **kept(最终)** |

- **40 轮达成**。Iter35(evict_first) 五次复测最鲁棒(3.45/3.50/3.47/3.54/3.50)，提为最终 solution.py。
- **收敛判定**：Iter 36-40 连续 5 轮 best speedup 未刷新(均在 3.49x 噪声内)，满足停止条件。
