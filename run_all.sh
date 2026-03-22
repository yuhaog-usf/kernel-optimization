#!/bin/bash
# ─── Run all 3 versions at multiple matrix sizes ─────────────────
# Usage: bash run_all.sh
# Prereq: source ~/Yuhao/setup_cuda.sh && make all

set -e

SIZES=("512 512 512" "1024 1024 1024" "2048 2048 2048" "4096 4096 4096")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  GEMM + bias + ReLU: v1 (separate) vs v2 (fused) vs v3    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")/bin"

for size in "${SIZES[@]}"; do
    M=$(echo $size | cut -d' ' -f1)
    N=$(echo $size | cut -d' ' -f2)
    K=$(echo $size | cut -d' ' -f3)

    echo "──────────────── M=$M  N=$N  K=$K ────────────────"
    echo ""
    ./v1_separate $M $N $K
    echo ""
    ./v2_fused_serial $M $N $K
    echo ""
    ./v3_fused_concurrent $M $N $K
    echo ""
done

echo "Done."
