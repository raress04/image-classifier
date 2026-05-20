#!/bin/bash
set -e

echo "=========================================="
echo " FACES: Combinatii Optime (Round 3)"
echo "=========================================="
echo " Strategie: combinam cele mai bune retetele"
echo " din rundele anterioare:"
echo "   - ub4 (full unfreeze)"
echo "   - weighted_ce (cel mai bun imbalance)"
echo "   - augall (cel mai bun augment pt ub4)"
echo "   - no warmup (a performat mereu mai bine)"
echo "   - 75 epoci (mai mult timp de convergenta)"
echo "=========================================="

echo ""
echo "[1/3] Faces | ub4 + weighted_ce + augall | 75 epoci"
echo "      (Combinatia principala: best of all worlds)"
.venv-ia2/bin/python3 main.py --dataset faces --model resnet18 \
    --epochs 75 --augment all --img_size 224 \
    --unfreeze_blocks 4 --imbalance_strategy weighted_ce

echo ""
echo "[2/3] Faces | ub4 + weighted_ce + color | 75 epoci"
echo "      (Cel mai bun config anterior, dar cu mai multe epoci)"
.venv-ia2/bin/python3 main.py --dataset faces --model resnet18 \
    --epochs 75 --augment color --img_size 224 \
    --unfreeze_blocks 4 --imbalance_strategy weighted_ce

echo ""
echo "[3/3] Faces | ub4 + weighted_ce + augall | 50 epoci"
echo "      (Control: verifica daca augall+weighted_ce imbunatateste fata de color+weighted_ce la 50ep)"
.venv-ia2/bin/python3 main.py --dataset faces --model resnet18 \
    --epochs 50 --augment all --img_size 224 \
    --unfreeze_blocks 4 --imbalance_strategy weighted_ce

echo ""
echo "=========================================="
echo " TOATE CELE 3 RULARI FINALIZATE!"
echo " Checkpoints generate:"
echo "   checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_nowarmup_ub4_lr0.001_ep75_best.pt"
echo "   checkpoints/faces_resnet18_aug-color_bnFalse_imb-weighted_ce_nowarmup_ub4_lr0.001_ep75_best.pt"
echo "   checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_nowarmup_ub4_lr0.001_ep50_best.pt"
echo "=========================================="
