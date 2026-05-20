#!/bin/bash
set -e

echo "=========================================="
echo " FACES: Atacul Final — Complex Head + CosineAnnealingLR"
echo "=========================================="
echo " Config:"
echo "   - ub4 (backbone complet dezghetat)"
echo "   - weighted_ce (loss cu ponderi per clasa)"
echo "   - augall (geometric + color)"
echo "   - complex_head (512->256->BN->ReLU->Dropout->7)"
echo "   - scheduler cosine (CosineAnnealingLR, T_max=120)"
echo "   - lr_backbone=1e-4, lr_head=1e-3 (diferentiat explicit)"
echo "   - 120 epoci"
echo "=========================================="

echo ""
echo "[1/1] Faces | ub4 + complex_head + CosineAnnealingLR | 120 epoci"
.venv-ia2/bin/python3 main.py --dataset faces --model resnet18 \
    --epochs 120 --augment all --img_size 224 \
    --unfreeze_blocks 4 --imbalance_strategy weighted_ce \
    --complex_head --scheduler cosine \
    --lr_backbone 1e-4 --lr_head 1e-3

echo ""
echo "=========================================="
echo " ANTRENARE FINALIZATA!"
echo " Checkpoint:"
echo "   checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_best.pt"
echo "=========================================="
