#!/bin/bash
set -e

echo "=========================================="
echo " FACES OVERNIGHT: Multi-Seed Ensemble Training"
echo "=========================================="
echo " Config:"
echo "   - ResNet-18 (ub4)"
echo "   - complex_head, weighted_ce, augall"
echo "   - scheduler cosine, 120 epoci"
echo "   - lr_backbone=1e-4, lr_head=1e-3"
echo "   - Ruleaza seed 42 si seed 123"
echo "=========================================="

echo ""
echo "[1/2] Antrenare model cu SEED 42..."
.venv-ia2/bin/python3 main.py --dataset faces --model resnet18 \
    --epochs 120 --augment all --img_size 224 \
    --unfreeze_blocks 4 --imbalance_strategy weighted_ce \
    --complex_head --scheduler cosine \
    --lr_backbone 1e-4 --lr_head 1e-3 \
    --seed 42

echo ""
echo "[2/2] Antrenare model cu SEED 123..."
.venv-ia2/bin/python3 main.py --dataset faces --model resnet18 \
    --epochs 120 --augment all --img_size 224 \
    --unfreeze_blocks 4 --imbalance_strategy weighted_ce \
    --complex_head --scheduler cosine \
    --lr_backbone 1e-4 --lr_head 1e-3 \
    --seed 123

echo ""
echo "=========================================="
echo " ANTRENARE MULTI-SEED FINALIZATA!"
echo " Checkpoints salvate in checkpoints/"
echo "=========================================="
