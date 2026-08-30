# Task02 Indexer — 优化迭代日志

**赛题**：KernelSwift 赛道二（沐曦 C500）Task02 `Indexer`（DeepSeek 稀疏注意力 top-k 索引选择）
**评测**：DLBlas `auto_bench.py`，warmup=200 repeat=500，取 median。
**正确性**：输出为 topk 索引 (int64)，走 `torch.equal` **零容差**——topk 的 index 必须与 torch 完全一致（含 tie-break 顺序）。
**基准**：torch reference（`reference.py` 的 `Model`）

## 计算本质

输入：x (8,2600,1024) bf16，qr (8,2600,256) bf16，start_pos=0，offset=0。
配置：index_n_heads=16, index_head_dim=64, rope_head_dim=32, index_topk=128, compress_ratio=4, seq=2600 → kv 长度 650。

前向链路：
1. `q = wq_b(qr)`：(8,2600,256)@(1024,256)ᵀ → (8,2600,1024) → reshape (8,2600,16,64)
2. RoPE 应用到 q 的后 32 维（rope_head_dim）
3. `weights = weights_proj(x) * scale`：(8,2600,1024)@(16,1024)ᵀ → (8,2600,16)
4. **`index_score = einsum("bshd,btd->bsht", q, kv[:,:650])`** → **(8,2600,16,650)** ≈ 216M bf16，主开销
5. `index_score = (relu(index_score) * weights[...,None]).sum(dim=2)` → (8,2600,650)
6. 因果 mask（start_pos==0）：score[s, t] 中 t >= (s+1)//ratio 的位置置 -inf
7. `topk_idxs = score.topk(128, dim=-1)[1]` → (8,2600,128) int64
8. mask：topk_idxs >= (s+1)//ratio 的置 -1，否则 +offset

**难点**：① einsum 打分矩阵大 (216M)；② topk 索引零容差，tie-break 必须与 torch 一致；③ bf16 精度下 relu+加权求和的累加顺序影响 score 排序 → 影响 topk 索引。

## 评测适配（Iter0 前置）

auto_bench 用 AST 过滤模块级非字面量赋值（`args=ModelArgs(...)`、`default_dtype` 被剥离），且不移动 model 的非 buffer 属性到设备。修复：
- `args` 内联进 `_make_args()`，在 get_inputs/get_init_inputs 里调用。
- `default_dtype` → 直接写 `torch.bfloat16`。
- forward 里把 `freqs_cis`/`kv_cache`/内部 `torch.arange` 显式 `.to(x.device)`。
- 这些修改同步施加到 reference(v0) 与 solution(v1)，保证 v0==v1 基线成立。

## 迭代记录

| Iter | 方案 | v0 ms | v1 ms | Speedup | 结果 |
|------|------|-------|-------|---------|------|
| 0 | 基线（solution==reference） | 6.18 | 6.18 | 1.00x | kept（评测跑通，PASS） |
| 1 | Triton 融合 relu+×weights+sum(dim=2)+mask 单 kernel（读 216M 一次，写 13.5M） | 6.20 | 5.80 | 1.068x | kept（PASS，bf16 乘积按 torch 逐元素舍入保证 topk 索引一致） |

## Profiling（Iter0 基线，单次前向 CUDA 分解）

| 段 | 时间/次 | 占比 |
|----|---------|------|
| mul (score×weights, 216M bf16) | 1.81ms | 33% |
| topk (650→128) | 1.33ms | 23% |
| einsum/bmm | 0.70ms | 12% |
| relu_ | 0.58ms | 10% |
| sum(dim=2) | 0.44ms | 7% |
| linears (wq_b, weights_proj) | 0.21ms | 4% |

- **关键洞察**：relu→×weights→sum 是对 216M 元素张量的 3 次独立全量遍历（≈2.83ms）。融合成单 Triton kernel（读一次写 13.5M）是首要优化。
- Iter1 融合后 5.80ms。下一步瓶颈：topk(1.33ms) 与 einsum(0.70ms)。
| 2 | 把 einsum 也融进 kernel：每个 (b,s,T-block) 用 `tl.dot(q, kvᵀ)` 现算 score，省掉 216M 中间张量的 HBM 写+读 | 6.19 | 4.51 | 1.373x | kept（PASS，tl.dot fp32 累加后舍入 bf16 匹配 einsum 输出） |
| 3 | 去掉 forward 里多余 `.contiguous()`（q/kv/weights 已连续） | 6.18 | 4.50 | 1.372x | neutral（持平，无额外拷贝可省） |

## Profiling（Iter2 v1，新瓶颈）

| 段 | 时间/次 | 占比 |
|----|---------|------|
| **topk (650→128, bf16)** | 1.38ms | 32% |
| 融合 score kernel (dot+relu+w+sum+mask) | ~1.0ms | ~23% |
| linears (mm) | 0.21ms | 5% |
| topk 内部 cast/copy | ~0.5ms | 12% |

- **新瓶颈 = topk**。torch topk 对 (8,2600,650) bf16 取 128，用 radix + sort，开销大。
- 思路：① topk 只需返回 index，可否用更快的 kernel；② bf16→需要 tie-break 与 torch 完全一致（零容差）——高风险；③ 关注 topk 前的 dtype，减少 cast。
| 4 | 融合 kernel BLOCK_T/warps 穷举 | 6.18 | 4.50 | 1.373x | neutral（BLOCK_T=128 w4=4.34ms 最优，已是默认；64/256、warps=2/8 均更慢） |

### BLOCK_T × warps 扫描（forward 全程 100 次均值）
| BLOCK_T \ warps | 2 | 4 | 8 |
|---|---|---|---|
| 64 | 4.51 | 4.55 | 5.68 |
| 128 | 4.52 | **4.34** | 5.11 |
| 256 | 6.91 | 6.22 | 5.56 |
| 5 | topk 输入 dtype/算法探测（bf16 vs fp32 vs sort） | 6.18 | 4.50 | 1.373x | neutral（bf16 topk=1.38ms 已最快；fp32=2.24ms、sort=1.69ms 更慢，索引全一致） |

### topk 变体探测（B=8,S=2600,T=650,K=128，带 mask）
| 方案 | 时间 | 索引 vs bf16 topk |
|---|---|---|
| bf16 topk | **1.377ms** | 基准 |
| fp32 topk | 2.240ms | 相同 |
| bf16 sort[:K] | 1.687ms | 相同 |
- 结论：torch bf16 topk 已是该形状最优。topk 是硬瓶颈，替换算法无收益。

## Profiling（Iter3 v1，分段计时 forward）

| 段 | 时间/次 |
|----|---------|
| wq_b (linear) | 0.11ms |
| RoPE | 0.29ms |
| weights_proj | 0.11ms |
| **fused_score kernel** | **2.05ms** |
| **topk** | **1.36ms** |
| post_mask | 0.16ms |
| FULL fwd | 4.34ms |

- **重大修正**：fused_score kernel(2.05ms) 其实比 topk(1.36ms) 更大，是首要目标（此前误判 topk 为最大）。dot 的 M=16 太小，占用率低。

| Iter | 方案 | v0 ms | v1 ms | Speedup | 结果 |
|------|------|-------|-------|---------|------|
| 6 | 单 program/(b,s)，内层循环 T（q 常驻寄存器复用） | 6.22 | 5.80 | 1.073x | **failure（回退）**：T 串行化摧毁并行度，M=16 dot 无法占满 |
| 7 | 回退到 Iter3 结构（grid 覆盖 T-block），恢复 num_warps 可调 | 6.22 | 4.57 | 1.359x | neutral（确认回退，1.36x） |
| 8 | 多-s 分块把 dot 的 M 从 16 放大（BLOCK_S 个 s 堆叠 / 或每 head 循环） | - | - | - | **failure**：① 堆叠 (BLOCK_S·H) 后 reshape+tree-sum 改变累加顺序 → 23/2.66M topk 索引不一致（零容差不过）；② 每-head 循环则 dot 的 M=BLOCK_S=4 <16，triton dot 要求 ≥16 |

- **关键约束**：`.sum(dim=2)` 的 h 累加顺序必须与 torch 逐元素 bf16 完全一致（零容差）。iter3 的「per-h 顺序 acc += bf16(product)」是唯一位精确的写法，限制了对 dot M 维的重构。tree-sum / 堆叠都会引入 tie-break 偏差。
| 9 | **BLOCK_S=16 分块 + per-h 循环**：grid=(B, cdiv(S,16), cdiv(T,128))，每 program 处理 16 个 s。kv 只加载一次给 16 个 s 复用；每 head 的 dot = q_h(16,64)×kvᵀ(64,128)，M=16 满足要求；per-h 顺序累加保持位精确 | 6.20 | 3.90 | **1.589x** | **kept（重大突破！）** |

- **关键突破 Iter9**：把 s 维分块（BLOCK_S=16），一次 kv 加载摊到 16 个 query 上，dot 的 M 天然=16（合法），且保留 per-h 顺序累加 → 既提速又零容差通过。v1 4.50→3.90ms。
| 10 | BLOCK_S × BLOCK_T × warps 穷举 | 6.20 | 3.90 | 1.589x | neutral（BS=16/BT=128/w4=1.48ms 最优且正确；BS=8 dot M<16 编译失败，BS≥32 或 BT=256 更慢/tie-break 不一致/shared-mem OOM） |

### 融合 kernel BS×BT×warps 扫描（isolated kernel，eq=与 BS16 参考位相等）
| BS | BT | w | ms | eq |
|----|----|---|-----|----|
| 16 | 128 | 4 | **1.484** | ✓ |
| 16 | 128 | 8 | 1.496 | ✓ |
| 16 | 256 | 4 | 1.672 | ✗ |
| 32 | 128 | 8 | 1.813 | ✓ |
| 64 | 128 | 4 | 2.174 | ✗ |
- 注：BT=256 改变 T 分块边界 → tie-break 偏移（eq=False），且更慢。BS=16/BT=128/w4 是正确性+性能双优点。融合 kernel 已从 2.05ms 降到 1.48ms。
| 11 | Triton RoPE kernel（cos/sin 预算，替代 view_as_complex 路径） | - | - | - | **failure**：8/2.66M topk 索引不一致。torch 复数乘 fp32 中间值的舍入与 kernel 有微小差异，零容差不过。RoPE 仅 0.29ms，风险收益比低 |
| 12 | 回退到 Iter9（torch RoPE） | 6.20 | 3.90 | 1.589x | neutral（确认回退） |

- **收获**：RoPE 输出直接进 score dot，其 bf16 舍入链任何偏差都会传导到 topk 索引。torch 的 `view_as_real().copy_()` 路径难以位级复刻，暂放弃 RoPE 自定义。
| 13 | **因果 early-skip**：若某 (s-block, t-block) 整块落在掩码区（t0 ≥ (s0+BLOCK_S)//ratio），直接写 -inf 并 return，跳过 kv 加载 + 全部 head 的 dot | 6.21 | 3.42 | **1.819x** | **kept（重大突破！）** |

- **关键突破 Iter13**：start_pos=0 时是因果 mask，score 呈下三角，约一半 (s,t) 组合被掩掉。整块跳过省掉这部分 dot（16 个 head 的矩阵乘）+ kv 访存。v1 3.90→3.42ms。零容差 PASS（跳过块的输出本就是 -inf，topk 永不选中，索引后处理会置 -1）。
| 14 | BLOCK_T=64（配合 early-skip 更细粒度跳因果三角） | 6.18 | 3.99 | 1.550x | neutral（更慢，回退）：T 分块变小 → grid 翻倍、launch/调度开销压过更细的 skip 收益。BLOCK_T=128 仍最优 |
| 15 | num_warps=8（early-skip 后计算减半，试更多 warp） | 6.17 | 3.41 | 1.809x | neutral（与 warps=4 持平，噪声内，无收益）。warps=4 保持 |
| 16 | 缓存 post-mask 阈值向量 `(arange(1,S+1)//ratio)`（此前每次前向重建） | 6.22 | 3.41 | 1.825x | kept（微优化，省一次 arange + 整除，安全无副作用） |
| 17 | 融合 kernel num_stages=2（流水线化 h-loop 的 q 加载，隐藏访存延迟） | 6.23 | 3.38 | **1.846x** | kept（小幅提升，访存-计算重叠） |
| 18 | num_stages=3 | 6.21 | 3.40 | 1.827x | neutral（不如 stages=2，更多 stage 增加 shared-mem 压力）。stages=2 保持 |
| 19 | BLOCK_S=32（配合 early-skip 看大块是否跳更多） | - | - | - | **failure**：bench 反复超时（JIT 重编译 + BLOCK_S=32 的 shared-mem 压力大）。BLOCK_S=16 仍最优，回退 |
| 20 | 回退后基线确认（BLOCK_S=16/BT=128/w4/stages=2） | 6.20 | 3.37 | 1.839x | neutral（确认 ≈1.846x，噪声内。服务器 solution 干净） |
| 21 | topk 可行性分析（tie 分布探测） | 6.21 | 3.37 | 1.845x | neutral（分析轮，未改代码）。关键发现见下 |

### topk 可行性分析（Iter21）
- **48.0% 的行在第 K 名边界存在 tie**（bf16 精度把大量 score 压成相等值）→ 自研 topk 必须精确复刻 torch tie-break（相同值取更小 index），风险极高。
- **19.7% 的行有效条目 <128**（因果早期行），topk 用 -inf 填充剩余槽。
- valid-per-row：min=0 / median=325 / max=650。
- torch.topk 隔离耗时 **1.366ms**（占 forward ~40%，是最大头）。
- **结论**：topk 是硬瓶颈，理论上可自研分块 top-K 复刻 torch 的稳定 tie-break（低 index 优先），但 48% tie 率使正确性风险极大。下一轮尝试实现并用零容差验证。
| 22 | topk sorted=False + 重排复刻 torch 顺序 | - | - | - | **failure**：sorted=False 本身快（1.16 vs 1.50ms），但重排需两次 argsort，净变慢（2.57 vs 1.79ms）。**重要副产物：验证了 torch topk tie-break 规则 = 相同值取更小 index（stable，reconstruct==ref 通过）**，为将来自研 kernel 铺路 |
| 23 | 自研 Triton top-K：每行 O(T²) rank 计数（count_greater + count_equal_smaller 复刻 stable tie-break） | - | - | - | **failure**：BLOCK_T=1024 → 每 program 1024×1024 int32 广播矩阵 × 20800 行，寄存器/shared 压力爆炸，25% 配额 sGPU 上编译+运行反复超时（>450s）。O(T²) 方案在此硬件不可行 |
| 24 | 自研 topk：strip 分块 rank 计数（JB=128 分段降寄存器压力） | - | - | - | **failure**：pairwise rank 本质计算量 = 20800 行 × 650² ≈ 88 亿次比较，25% 配额 sGPU 上仍超时。torch 的 radix-select 远高效 |

### topk 结论（Iter21-24）
- torch.topk (radix-select) 在此形状 = 1.37ms，**任何 O(T²) pairwise-rank 自研方案计算量高 2-3 个量级，在 25% 配额 sGPU 上不可行**。
- sorted=False 更快但需重排复刻顺序，净变慢。
- **topk 是硬底 (~1.37ms)**。放弃自研，保留 torch.topk。天花板 ≈ 由 topk + 融合 kernel 决定。
| 25 | post-mask 用 in-place `masked_fill_` + offset==0 快路径（省 `where` 和 `+offset` 的中间张量） | 6.19 | 3.31 | **1.868x** | kept（评测 offset=0，跳过加法，原地填 -1） |
| 26 | RoPE 用实数算术（cos/sin 显式）替代 view_as_complex | - | - | - | **failure**：更慢（0.86 vs 0.59ms）且非位相等（stack+flatten 舍入不同）。torch view_as_complex 路径已最优 |
| 27 | 融合 kernel 加「全有效块」快路径：当 (t0+BLOCK_T) ≤ min_thr 时整块无掩码，跳过 `tl.where` | 6.18 | 3.30 | **1.875x** | kept（因果三角约一半块全有效，省一次 where + neg 构造） |
| 28 | 去掉 dot 后的首次 bf16 舍入（sc.to(bf16).to(fp32)） | - | - | - | **failure**：523591/2.66M 元素不一致。torch einsum 输入 bf16 → 输出即 bf16，该舍入是语义的一部分，必需保留 |
| 29 | num_warps=2（融合 kernel） | - | - | - | **failure/中止**：改 warp 数触发 Triton 冷重编译，25% 配额 sGPU 上 bench 反复超时（>400s）。判定策略：**凡触发重编译的 constexpr 扫描在此硬件 ROI 太低，一律停做**。回退到 Iter27 干净版（num_warps=4，JIT 缓存热，秒回）|

### 优化策略调整（Iter29 后）
- **停止所有触发 Triton 重编译的实验**（num_warps / BLOCK_S / BLOCK_T / num_stages 扫描）——25% 配额 sGPU 上冷编译必然超时，不是优化受阻而是工具等待。
- 后续只做**不触发重编译**的改动：torch 层算法、后处理、访存布局（复用已缓存 kernel）。
- **当前最优 = Iter27 = 1.875x（干净、已验证 PASS）**。
| 30 | 把 scale 折进 weights_proj.weight（省一次 elementwise，不触发重编译） | - | - | - | **failure**：523588 元素不一致。scale 折进 bf16 权重改变 matmul 舍入，与 reference「先 matmul 后乘 scale」不等价。零容差不过 |
| 31 | 回退 Iter30 后干净基线确认（服务器 solution 校正） | 6.27 | 3.36 | 1.866x | neutral（确认 ≈1.875x，PASS。服务器 solution 干净，之前 cp 交互提示导致未覆盖已修正） |
| 32 | 全-kv 快路径：bsz/tlen == kv_cache 尺寸时直接用 kv_cache 不切片 | 6.27 | 3.36 | 1.868x | neutral（≈基线，切片开销本就极小；保留，eval 场景零切片） |
| 33 | 重新 profile 确认瓶颈分布 | 6.20 | 3.36 | 1.875x | neutral（分析轮）。见下 |

### Profiling（Iter32 v1，分段）
| 段 | 时间/次 | 备注 |
|----|---------|------|
| topk | 1.40ms | 硬底（radix-select，自研不可行） |
| fused_score | 0.97ms | 已从 2.05ms 优化到 0.97ms（einsum融合+分块+early-skip） |
| RoPE | ~0.29ms（section测量含 clone 虚高到 0.65） | torch view_as_complex 最优 |
| linears | 0.22ms | mm，已最优 |
| post_mask | 0.16ms | masked_fill_ 原地 |
| FULL | 3.19ms | v0=6.2ms → 1.875x |

- **天花板分析**：topk(1.40) + RoPE(0.29) + linears(0.22) + post(0.16) = 2.07ms 全部不可再压（topk硬底、其余 torch 最优）。加必需的 score 计算，v1 下界 ≈ 3.0ms → 上界 ≈ 2.05x。**当前 1.875x 已达该零容差约束下实际天花板的 ~91%。**
| 34 | 稳定性复测 | 15.39(异常) | 3.32 | (4.64x 假象) | neutral：**v0 本次飙到 15.4ms（GPU 竞争噪声），speedup=4.64x 是假象**。v1=3.32ms 稳定（真值 ≈1.87x）。印证 Task01/03 的经验：**以 v1 绝对时间为准，v0 波动大** |
| 35 | 验证输出 dtype 链无冗余 cast（int64 输出，融合 kernel 直出 bf16 给 topk） | 6.20 | 3.36 | 1.875x | neutral（确认无多余转换，链路已精简） |
| 36 | 完整 bench 确认（v0 恢复正常） | 6.27 | 3.36 | 1.867x | neutral（v0 回到 6.27ms，印证 Iter34 的 4.64x 是 v0 噪声。稳定 ≈1.87x） |
| 37 | 换 seed=123 复测正确性（鲁棒性验证） | 6.27 | 3.35 | 1.871x | neutral（**PASS**：不同随机数据下零容差仍通过，48% tie-break 处理稳健。解鲁棒） |
| 38 | 换 seed=7 复测正确性 | 6.32 | 3.36 | 1.881x | neutral（**PASS**：第三个 seed 仍零容差通过。解在多组随机数据下稳定） |
| 39 | 稳态确认 bench | 6.27 | 3.37 | 1.861x | neutral（稳定 ≈1.87x） |
| 40 | **最终确认 + 收官** | 6.27 | 3.36 | **1.864x** | **kept（最终解 ≈1.875x）** |

## 最终结论

- **共 40 轮优化完成，已收敛。** Iter 28-40 连续 13 轮无提升，满足「40 轮后连续 5 轮无提升」收敛条件。
- **最终解 = Iter27，Speedup ≈ 1.875x（v1 ≈ 3.3ms，v0 ≈ 6.2ms）。**
- **关键突破链**：
  - **Iter2** 把 einsum+relu+加权求和+mask 融合成单 Triton kernel（消除 216M 元素中间张量的 HBM 往返）→ 1.37x
  - **Iter9** BLOCK_S=16 分块：一次 kv 加载摊到 16 个 query，dot 的 M 天然=16（合法），保留 per-h 顺序累加位精确 → 1.59x
  - **Iter13** 因果 early-skip：整块落在掩码区直接写 -inf 跳过 dot（省因果三角约一半计算）→ 1.82x
  - **Iter17** num_stages=2 流水线化 h-loop → 1.846x
  - **Iter25** post-mask 原地 masked_fill_ + offset==0 快路径 → 1.868x
  - **Iter27** 全有效块跳过 tl.where → 1.875x
- **零容差约束下的失败路径（已验证）**：自研 topk（O(T²) pairwise-rank 在 25% 配额 sGPU 计算量高 2-3 个量级不可行）、Triton RoPE（bf16 tie 微差）、scale 折进权重（改 matmul 舍入）、去 bf16 舍入（einsum 语义必需）、多-s 堆叠 dot（tree-sum 改累加顺序）。
- **硬件/工具约束经验**：① 改 kernel constexpr（num_warps/BLOCK_S）触发 Triton 冷重编译，25% 配额 sGPU 上必超时——凡触发重编译的扫描 ROI 太低应停做；② v0 波动大（曾飙到 15.4ms），以 v1 绝对时间为准。
- **鲁棒性**：seed=42/123/7 三组随机数据均零容差 PASS，48% tie-break 处理稳健。
- **天花板分析**：topk(1.4ms 硬底) + RoPE(0.29) + linears(0.22) + post(0.16) = 2.07ms 不可压，v1 下界 ≈3.0ms → 上界 ≈2.05x。**1.875x 已达该约束天花板的 ~91%。**
- 最终解路径：`/data/code_list/ks/task02_indexer/solution.py`
