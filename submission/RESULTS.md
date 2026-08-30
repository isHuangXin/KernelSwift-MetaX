# 性能测试结果 —— KernelSwift 赛道二（沐曦 C500）

## 测试环境
- 硬件：MetaX C500（16GB sGPU 切片，25% 算力配额）
- 软件：MACA 3.5.3.20 | torch 2.8.0+metax | triton 3.0.0+metax | Python 3.12
- 评测：DLBlas `benchmarks/ks/auto_bench.py`，warmup=200 repeat=500，取 median
- 正确性：Task01/02 零容差（torch.equal），Task03 低容差（allclose atol/rtol=1e-2）

## 实测数据（多次运行）

### Task01 engram_hash
| 运行 | v0 (ms) | v1 (ms) | Speedup | 正确性 |
|------|---------|---------|---------|--------|
| 1 | 0.3837 | 0.0993 | 3.862x | PASS |
| 2 | 0.3777 | 0.1035 | 3.649x | PASS |

稳定加速比 ≈ **3.5x**（v1 ≈ 0.10ms，v0 有噪声）。

### Task02 Indexer
| 运行 | v0 (ms) | v1 (ms) | Speedup | 正确性 |
|------|---------|---------|---------|--------|
| 1 | 6.2753 | 3.3561 | 1.870x | PASS |

稳定加速比 ≈ **1.87x**（v1 ≈ 3.36ms）。

### Task03 norm_fn
| 运行 | v0 (ms) | v1 (ms) | Speedup | 正确性 |
|------|---------|---------|---------|--------|
| 1 | 0.4699 | 0.1353 | 3.473x | PASS |

稳定加速比 ≈ **3.47x**（v1 ≈ 0.135ms）。

## 说明
- v0（reference）在 sGPU 配额与系统竞争下存在测量噪声（曾观测到偶发飙升），
  以 v1 绝对时间和多次中位数判断真实性能。
- 三个赛题均通过正确性校验，实际执行路径运行自定义 Triton 算子（非 fallback）。
