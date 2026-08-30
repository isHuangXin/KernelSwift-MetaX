# 2026 KernelSwift 算子创新大赛 —— 赛道二（沐曦 C500）参赛作品

本作品为 KernelSwift 赛道二三个赛题的 **Triton** 算子优化实现，在沐曦 MetaX C500 上通过
DeepLink DLBlas `auto_bench.py` 评测（正确性校验 + 加速比测速）。

## 一、作品说明

| 赛题 | 算子 | 加速比 Speedup | v1 绝对时间 | 正确性 |
|------|------|---------------|------------|--------|
| Task01 | engram_hash | **~3.5x**（实测 3.65–3.86x） | 0.10ms | PASS（int32 零容差 torch.equal） |
| Task02 | Indexer | **~1.87x** | 3.36ms | PASS（int64 topk 索引零容差） |
| Task03 | norm_fn | **~3.47x** | 0.135ms | PASS（float 低容差 atol/rtol=1e-2） |

> 加速比基准为赛题给定的 torch reference 实现；测速 warmup=200 repeat=500 取 median。
> v0 存在测量噪声，以 v1 绝对时间与多次中位数为准。

## 二、优化方案

### Task01 engram_hash
- 将 torch 的多算子链（int64 乘、xor 哈希、取模、加 offset）**融合成单个 Triton kernel**，
  消除多次 kernel launch 与中间张量。
- **多 token 分块 BLOCK_M=32**：单个 program 处理 32 个 token，摊薄 launch/调度开销（最大跃升）。
- 关键发现：MetaX C500 原生 int64 取模已很快，魔数乘法替代取模反而更慢，故保留原生 `%`。

### Task02 Indexer
- 将 `einsum 打分 + relu + 加权求和 + 因果 mask` **融合成单个 Triton kernel**，消除
  216M 元素中间张量的 HBM 往返。
- **BLOCK_S=16 序列分块**：一次 kv 加载摊到 16 个 query，`tl.dot` 的 M 维天然=16（满足硬件要求），
  且保留 per-head 顺序累加以匹配 torch bf16 语义（零容差）。
- **因果 early-skip**：整块落在掩码区的 (s,t) 块直接写 -inf 跳过 dot，省去因果三角约一半计算。
- num_stages=2 流水线 + 后处理原地 masked_fill_。
- topk 保留 torch.topk（自研 O(T²) 方案在 sGPU 配额下不可行，torch radix-select 是硬底）。

### Task03 norm_fn
- Triton 融合 + autotune + **Split-K 并行**（208 program 并行归约）+ evict_first cache 提示。

## 三、性能测试结果

见 `RESULTS.md`。

## 四、原创声明

本作品全部算子实现由参赛者独立完成，基于赛题给定的 torch reference 进行 Triton 重写与优化。
所有优化思路、kernel 代码、调参过程均为原创，完整迭代日志见各赛题目录下 `ITERATIONS.md`。
未抄袭任何第三方已有实现。所有提交代码均通过官方 `auto_bench.py` 正确性校验，且实际执行路径
运行自定义 Triton 算子（非 fallback 至 PyTorch 内置算子）。

## 五、目录结构

```
task01_engram_hash/  solution.py reference.py ITERATIONS.md task01_progress.png
task02_indexer/      solution.py reference.py ITERATIONS.md task02_progress.png
task03_norm_fn/      solution.py reference.py ITERATIONS.md task03_progress.png
README.md  RESULTS.md  requirements.txt  run.sh
```

## 六、运行方式

```bash
bash run.sh          # 依次评测三个赛题
# 或单独评测：
python DLBlas/benchmarks/ks/auto_bench.py \
  --v0_file task01_engram_hash/reference.py \
  --v1_file task01_engram_hash/solution.py
```

## 七、环境

见 `requirements.txt`。核心：MetaX C500 / MACA 3.5.3.20 / torch 2.8.0+metax /
triton 3.0.0+metax / Python 3.12（推荐 3.10）。
