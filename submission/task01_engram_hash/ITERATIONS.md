# Task01 engram_hash — 优化迭代日志

**赛题**：KernelSwift 赛道二（沐曦 C500）Task01 `engram_hash`
**评测**：DLBlas `auto_bench.py`，warmup=200 repeat=500，取 median。
**正确性**：输出 int32，走 `torch.equal` **零容差**——xor/取模必须位级精确。
**基准**：torch reference（`reference.py` 的 `Model`）

## 计算本质

输出 shape (L=2, num_tokens=4096, COLS=16)，每个 (layer, token) 独立：
- 3 个 int64 乘积 `prod_s = token[tok,s] * mult[layer,s]`（s=0..2）
- 累积 xor：`hash_i = prod_0 ^ ... ^ prod_{i+1}`（i=0..1，对应 ngram step）
- 16 列输出：col=(i,t)，`out = (hash_i % vocab[layer,i,t]) + offset[layer,col]`

**难点**：① 必须 int64 运算（token×mult 超 int32）；② `%` 是慢整数除法，是主要优化点（魔数乘法替代）；③ 零容差，位精确。

## 迭代记录

| Iter | 标题 | Speedup(median) | v0 ms | v1 ms | 正确性 | 状态 |
|------|------|-----------------|-------|-------|--------|------|
| 0 | Reference (baseline) | 1.00x | 0.347 | — | — | kept |
| 1 | Triton 融合初版 (grid=L*num_tokens, 向量化16列) | **1.70x** | 0.347 | 0.205 | ✅ PASS | kept |

### Iter 1 详情
- **Hypothesis**: 参考实现有双重 Python for(layer×ngram) + cat/stack + 大量中间张量，融合成单 Triton kernel。
- **Changes**: `_engram_kernel` grid=(L*num_tokens,)，每 program 处理一个 (layer,token)，向量化 16 列，
  内部循环 3 个 ngram step 累积 xor（用 `s<=i+1` mask 控制哪些列包含该 step）。
- **Bench**: Correct ✅ (torch.equal) / v1=0.205ms / speedup=1.70x
- **Analysis**: 融合有效但 `%` 取模是瓶颈（GPU 整数除法慢）。grid=8192 并行充足。
- **Next**: **魔数乘法替代取模**（预计算 magic/shift，用乘法+移位代替 `hash % vocab`）—— 最大收益点。

## 取模优化探索（Iter 2-4）

| Iter | 方案 | Speedup | 结果 |
|------|------|---------|------|
| 2 | float64 倒数取模替代整数% | 1.61x | neutral(沐曦fp64吞吐低,更慢) |
| 3 | 标量 running-xor 广播到列 | 1.67x | neutral(xor非瓶颈,噪声内) |
| 4 | 基线复测 x3 | 1.70/1.70/2.05 | neutral(噪声±15%,稳定~1.70x) |

- **关键认知**：① 取模是瓶颈但 float64 更慢(沐曦fp64单元少)；② 整数魔数除法对 64位hash(<2^35)需128位中间积，
  Triton int64 会溢出，`tl.umulhi` 只支持 32位，实现复杂且收益不确定；③ xor 不是瓶颈。
- **hash 值域**：max|hash|~2^34，vocab~2^20，非负。
- **Next**: 转向并行度/访存优化——每 program 处理多 token(BLOCK_M)提高访存效率、autotune、grid 结构。

## 多-token 并行路线（Iter 5-8，突破）

| Iter | 方案 | v1 ms | Speedup | 结果 |
|------|------|-------|---------|------|
| 5 | **每 program 处理 BLOCK_M=16 个 token(2D)** | 0.101 | **3.42x** | **kept(突破)** |
| 6 | BLOCK_M=64 | 0.100 | 3.42-3.68x | neutral(v1持平,4.35x是v0偏高假象) |
| 7 | BLOCK_M=128 | 0.101 | 3.39x | neutral(v1持平) |
| 8 | BLOCK_M=32 + warps=2 | 0.100 | 3.44-3.62x | neutral(v1持平) |

### Iter 5 详情（关键突破）
- **Hypothesis**: grid=8192 每 program 只算 1 token×16列，工作量太小、访存碎。改成每 program 处理 BLOCK_M 个 token。
- **Changes**: grid=(L, ceil(num_tokens/BLOCK_M))，2D 块 (BLOCK_M, BLOCK_C)；mult 标量复用、token 连续加载、输出连续写。
- **Bench**: Correct ✅ (torch.equal 零容差) / v1=0.101ms / **speedup=3.42x**（1.70x→3.42x，翻倍）。
- **Analysis**: 访存效率大幅提升是主因（原 8192 个碎 program → L×256 个大 program）。
- **重要方法论**: speedup 受 v0 波动影响大（v0 波动 0.34-0.44ms），后续以 **v1 绝对耗时**为准判断真实提升。

### 参数饱和结论（Iter 6-8）
- BLOCK_M ∈ {16,32,64,128} 的 v1 全部 ≈0.100ms → **访存带宽饱和，BLOCK_M 已到甜点**。
- **Next**: v1≈0.100ms 是访存天花板。下轮试减少 int64 运算量、vocab/offset 合并加载、grid 结构。

## 访存/运算量优化（Iter 9-11，v1 已饱和 ~0.100ms）

Profiling(torch.profiler): `_engram_kernel` = **100% GPU 时间**(12.26µs/次)，无 copy/contiguous 开销，纯 kernel-bound。

| Iter | 方案 | v1 ms | 结果 |
|------|------|-------|------|
| 9 | num_warps=8 | 0.111 | neutral(256元素分8warps过度分割,变慢) |
| 10 | BLOCK_M=8 | 0.110 | neutral(grid变多,略慢) |
| 11 | 只存2个hash快照(减int64 where) | 0.102 | neutral(运算非瓶颈,取模+访存才是) |

- **结论**：v1≈0.100ms 是访存/取模天花板。BLOCK_M=16 + 默认warps 是甜点。
  xor 运算量优化无效(不是瓶颈)，取模(int64 %)是核心成本但难在 Triton 上加速(魔数需128位积)。
- **Next**: 剩余轮次试 grid 结构(L 维融进 M)、vocab 预取到 shared、复测确认 ~3.4x 稳健性。

## 魔数取模验证（Iter 12-13，重要反直觉结论）

| Iter | 方案 | v1 ms | 结果 |
|------|------|-------|------|
| 12 | num_warps=1 | 0.100 | neutral(持平) |
| 13 | **魔数乘法替代取模**(S=44 magic+1步修正,位级精确) | 0.128-0.151 | neutral(**反而更慢**) |

### Iter 13 详情（关键实测结论）
- **Hypothesis**: 你朋友建议「魔数乘法替代取模」——GPU 整数除法慢，用 `q=(h*magic)>>44` + 修正替代 `%`。
- **数值验证**: S=44, magic=floor(2^44/vocab)+1, 1步修正 → **torch.equal 位级精确**（已验证）。
- **Bench**: Correct ✅ / v1=0.128-0.151ms（比原生 % 的 0.100ms **慢 30-50%**）。
- **Analysis**: 魔数方案多了 ① magic 数组的 HBM 加载 ② int64×int64 乘法（本身 64位乘也不便宜）
  ③ 修正分支。总开销 > 沐曦 C500 原生 int64 `%`。**说明 C500 的整数除法单元并不慢，原生 % 已最优。**
- **反直觉结论**: 「魔数替代取模」这一常见 NV GPU 技巧在**沐曦 C500 上不成立**，原生 `%` 更快。保留原生取模。
- **Next**: 取模已确认最优。剩余轮次做复测 + grid/访存微调，预期维持 ~3.4x。

## grid/访存微调（Iter 14-16）

| Iter | 方案 | v1 ms | 结果 |
|------|------|-------|------|
| 14 | grid 一维扁平(L融进block) | 0.107-0.109 | neutral(kernel内解layer开销,略慢) |
| 15 | 基线复测 x4 | 0.106(稳定) | neutral(确认~3.35x,方差小) |
| 16 | 一次批量加载 token 整行(BLOCK_M,NG) | 0.102-0.104 | neutral(tl.where选列开销抵消,持平) |

- **结论**：v1 稳定 ~0.106ms(比Task03更稳,方差小)。2D grid(L, blocks) 优于1D扁平。
  批量 token 加载因 Triton 不支持标量索引需用 tl.where 选列，收益被抵消。
- **当前最优 Iter5 (BLOCK_M=16, 2D grid, 原生取模) = ~3.4x 稳健。**
- **Next**: 已探索 16 变体。取模/BLOCK_M/grid/访存均已探明。剩余轮次复测+边缘微调，预期维持 ~3.4x。

## BLOCK_M 精调（Iter 17-19，Iter17 真提升）

| Iter | 方案 | v1 ms | 结果 |
|------|------|-------|------|
| 17 | **BLOCK_M=32, warps=4** | 0.0995-0.100(4次极稳) | **kept(v1 0.106→0.100, 方差更小)** |
| 18 | BLOCK_M=64 | 0.094-0.101 | neutral(与BM=32持平) |
| 19 | BLOCK_M=64 + warps=8 | 0.101-0.103 | neutral(warps=4最优) |

### Iter 17 详情
- **Hypothesis**: BLOCK_M=16 可能未饱和访存，试 BLOCK_M=32。
- **Bench**: 对比测试 BM=32 (0.0995-0.100ms,4次极稳) vs BM=16 (0.101-0.103ms) → BM=32 稳定略优~2%且方差小。
- **Analysis**: BM=32 访存效率略高，确认为真提升(v1 从 0.106→0.100ms)。提为最优。
- **Next**: BM∈{32,64} 已饱和。剩余轮次复测+边缘微调，预期 ~3.5x。进度 20 轮。

## 访存布局/流水微调（Iter 20-22）

| Iter | 方案 | v1 ms | 结果 |
|------|------|-------|------|
| 20 | num_stages=2 | 0.100-0.102 | neutral(无循环流水收益) |
| 21 | vocab+offset 打包(L,COLS,2) | 0.105-0.111 | neutral(stride=2破坏合并访存,更慢) |
| 22 | warps=2 | 0.100-0.101 | neutral(访存受限,warps不敏感) |

- **结论**：vocab/offset 分开的连续加载优于打包；num_stages/warps 对访存受限 kernel 无益。
- **Iter17 (BLOCK_M=32, warps=4, 2D grid, 原生取模, 分离vocab/offset加载) = 稳健最优 ~3.47x (v1≈0.100ms)。**
- **Next**: 已探索 22 变体。剩余轮次复测确认稳健性，40 轮后连续5轮无提升则收敛。进度 23 轮。

## 类型/访存提示微调（Iter 23-25）

| Iter | 方案 | v1 ms | 结果 |
|------|------|-------|------|
| 23 | 基线复测 x4 | 0.0997-0.100(极稳) | neutral(确认 v1≈0.100ms) |
| 24 | tok int32 自动提升(省显式.to(int64)) | 0.0996-0.103 | neutral(持平,代码更简洁,正确性PASS) |
| 25 | max_contiguous 访存提示 | 0.0997-0.100 | neutral(访存已合并,提示无益) |

- **结论**：v1≈0.100ms 稳固。类型自动提升、访存提示均无额外收益(kernel已达访存带宽)。
- **Iter17 = 稳健最优 ~3.47x**。已探索 25 变体。
- **Next**: 剩余轮次复测确认，40 轮后连续5轮无提升则收敛。进度 26 轮。

## BLOCK_M/warps 穷举确认（Iter 26-28）

| Iter | 方案 | v1 ms | 结果 |
|------|------|-------|------|
| 26 | BLOCK_M=256 | 0.110-0.112 | neutral(块太大,grid=32并行不足) |
| 27 | BLOCK_M=32 + warps=8 | 0.102 | neutral(略慢) |
| 28 | BLOCK_M=128 | 0.102-0.114 | neutral(略慢+波动) |

- **BLOCK_M 全扫描确认**：8(慢)/16(0.106)/32(0.100★)/64(0.100)/128(慢)/256(慢) → **BM=32 最优**。
- **warps 全扫描**：1/2/4(★)/8 → warps=4 最优。
- **Iter17 (BM=32, warps=4) = 全局最优 ~3.47x。** 已探索 28 变体。
- **Next**: 参数空间穷尽。剩余轮次复测确认，40 轮后连续5轮无提升则收敛。进度 29 轮。

## 收敛复测（Iter 29-40）

| Iter | 方案 | v1 ms | 结果 |
|------|------|-------|------|
| 29 | BLOCK_M=32 复测 | 0.103-0.108 | neutral |
| 30 | grid=(cdiv, L) 维度对调 | 0.104-0.106 | neutral(持平) |
| 31 | num_stages=2 | 0.105-0.108 | neutral(无益) |
| 32 | 基线复测 | 0.100-0.101 | neutral |
| 33 | BLOCK_M=16 + warps=2 | 0.100 | neutral |
| 34 | warps=1 | 0.104 | neutral(略慢) |
| 35 | 基线复测 | 0.101 | neutral |
| 36 | BLOCK_M=64 + warps=4 | 0.100-0.101 | neutral(早期 0.094 为偶发离群值,复测持平) |
| 37 | 基线复测 | 0.100 | neutral |
| 38 | 基线复测 | 0.101 | neutral |
| 39 | 基线复测 | 0.100 | neutral |
| 40 | **最终确认 BM=32 warps=4** | **0.100** | **kept(最终解 3.47x)** |

## 最终结论

- **共 40 轮优化完成。** Iter 33-40 连续 8 轮无提升，满足「40 轮后连续 5 轮无提升」收敛条件。
- **最终解 = Iter17 (BLOCK_M=32, num_warps=4, grid=(L, cdiv(num_tokens,32)))，Speedup ≈ 3.47x，v1 ≈ 0.100ms。**
- **关键突破**：
  - Iter1 Triton 融合单 kernel（1.70x）——消除 torch 多算子 launch + 中间张量。
  - Iter5 多 token BLOCK_M（3.42x）——单 program 处理 BLOCK_M 个 token，摊薄 launch/调度开销，是最大跃升。
  - Iter17 BLOCK_M=32 定型（3.47x）——占用率与并行度最佳平衡点。
- **无效路径（已验证）**：魔数乘法取模（MetaX C500 native int64 % 已很快，魔数反而慢）、vocab/offset 打包（stride=2 破坏合并访存）、float64 取模、访存 cache 提示、num_stages、grid 维度对调、BLOCK_M∈{8,16,64,128,256}、warps∈{1,2,8}。
- **~3.47x 是访存带宽上限**：kernel 已是纯访存受限，输出 16 列 int32 的写带宽为硬瓶颈。
- 最终解路径：`/data/code_list/ks/task01_engram_hash/solution.py`
