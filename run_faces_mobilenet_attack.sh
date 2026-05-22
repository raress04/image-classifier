#!/bin/bash
set -e

echo "=========================================="
echo " FACES OVERNIGHT: MobileNet-v2 Final Attack"
echo "=========================================="
echo " Config:"
echo "   - ub4 (backbone complet dezghetat)"
echo "   - weighted_ce (loss cu ponderi)"
echo "   - augall (geometric + color)"
echo "   - complex_head (MLP head pe classifier)"
echo "   - scheduler cosine (CosineAnnealingLR, T_max=120)"
echo "   - lr_backbone=1e-4, lr_head=1e-3 (diferentiat)"
echo "   - 120 epoci"
echo "=========================================="

echo ""
echo "[1/1] Faces | MobileNet-v2 | ub4 + complex_head + CosineAnnealing | 120 epoci"
.venv-ia2/bin/python3 main.py --dataset faces --model mobilenetv2 \
    --epochs 120 --augment all --img_size 224 \
    --unfreeze_blocks 4 --imbalance_strategy weighted_ce \
    --complex_head --scheduler cosine \
    --lr_backbone 1e-4 --lr_head 1e-3

echo ""
echo "=========================================="
echo " ANTRENARE MOBILENET-V2 FINALIZATA!"
echo "=========================================="
