#!/bin/bash
set -e

# Configurare python path
PYTHON_BIN=".venv-ia2/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN=".venv-ia2/bin/python3"
fi
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python"
fi

echo "Folosim interpretorul Python: $PYTHON_BIN"
echo "=========================================="
echo " GENERARE SUBMISSIONS PENTRU MODELELE MULTI-SEED"
echo "=========================================="

# 1. Generare submisie model Base (ep120) cu TTA
echo ""
echo "[1/4] Generare submisie model Base (implicit)..."
$PYTHON_BIN submit_kaggle.py \
    --dataset faces \
    --model resnet18 \
    --img_size 224 \
    --unfreeze_blocks 4 \
    --complex_head \
    --tta \
    --checkpoint checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_best.pt \
    --output submissions/submission_faces_resnet18_ep120_best_tta.csv

# 2. Generare submisie model Seed 42 cu TTA
echo ""
echo "[2/4] Generare submisie model Seed 42..."
$PYTHON_BIN submit_kaggle.py \
    --dataset faces \
    --model resnet18 \
    --img_size 224 \
    --unfreeze_blocks 4 \
    --complex_head \
    --tta \
    --checkpoint checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_seed42_best.pt \
    --output submissions/submission_faces_resnet18_ep120_seed42_tta.csv

# 3. Generare submisie model Seed 123 cu TTA
echo ""
echo "[3/4] Generare submisie model Seed 123..."
$PYTHON_BIN submit_kaggle.py \
    --dataset faces \
    --model resnet18 \
    --img_size 224 \
    --unfreeze_blocks 4 \
    --complex_head \
    --tta \
    --checkpoint checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_seed123_best.pt \
    --output submissions/submission_faces_resnet18_ep120_seed123_tta.csv

# 4. Generare Ensemble (Soft-voting pe cele 3 modele) cu TTA
echo ""
echo "[4/4] Generare submisie ENSEMBLE (Base + Seed 42 + Seed 123)..."
$PYTHON_BIN submit_ensemble_kaggle.py \
    --dataset faces \
    --models resnet18 resnet18 resnet18 \
    --img_size 224 \
    --unfreeze_blocks 4 4 4 \
    --complex_heads 1 1 1 \
    --tta \
    --checkpoints \
        checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_best.pt \
        checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_seed42_best.pt \
        checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_seed123_best.pt \
    --output submissions/submission_faces_resnet18_ensemble_seeds_tta.csv

echo ""
echo "=========================================="
echo " TOATE SUBMISIILE AU FOST GENERATE CU SUCCES!"
echo " Gasiti fisierele CSV in directorul 'submissions/':"
echo "   - submission_faces_resnet18_ep120_best_tta.csv"
echo "   - submission_faces_resnet18_ep120_seed42_tta.csv"
echo "   - submission_faces_resnet18_ep120_seed123_tta.csv"
echo "   - submission_faces_resnet18_ensemble_seeds_tta.csv (Ensemble)"
echo "=========================================="
