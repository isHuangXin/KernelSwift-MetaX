#!/usr/bin/env bash
# 运行脚本 —— KernelSwift 赛道二（沐曦 C500）
# 依次评测三个赛题，输出正确性与加速比。
#
# 前置：已配置 MACA 环境的 MetaX C500，且已 clone DLBlas 评测框架。
# 用法：bash run.sh [DLBLAS_DIR]
#   DLBLAS_DIR 默认为 ../DLBlas（若不同请传参或修改）

set -e

DLBLAS_DIR="${1:-../DLBlas}"
BENCH="${DLBLAS_DIR}/benchmarks/ks/auto_bench.py"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ ! -f "$BENCH" ]; then
  echo "找不到评测脚本：$BENCH"
  echo "请先获取 DLBlas：git clone https://github.com/DeepLink-org/DLBlas"
  echo "然后：bash run.sh /path/to/DLBlas"
  exit 1
fi

for t in task01_engram_hash task02_indexer task03_norm_fn; do
  echo "==================== ${t} ===================="
  python "$BENCH" \
    --v0_file "${HERE}/${t}/reference.py" \
    --v1_file "${HERE}/${t}/solution.py"
  echo ""
done
