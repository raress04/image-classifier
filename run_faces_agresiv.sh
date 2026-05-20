#!/bin/bash
set -e

echo "=========================================="
echo " FACES: Fine-Tuning Agresiv (Full Unfreeze)"
echo "=========================================="

# Strategia:
# - unfreeze_blocks=4 (dezghetat complet): backbone_lr=1e-4, head_lr=1e-3
# - imbalance_strategy=none (Faces nu e la fel de dezechilibrat ca Moles;
#   sampler-ul forta overfitting pe clasele minoritare mici)
# - augment=color (geometric strica alinierea fetei)
# - no warmup (a performat mai bine in rundele anterioare)
# - 50 epoci

echo ""
echo "[1/4] Faces ResNet18 | full unfreeze | imbalance=none | color | no warmup"
python3 main.py --dataset faces --model resnet18 \
    --epochs 50 --augment color --img_size 224 \
    --unfreeze_blocks 4 --imbalance_strategy none

echo ""
echo "[2/4] Faces ResNet18 | full unfreeze | imbalance=weighted_ce | color | no warmup"
python3 main.py --dataset faces --model resnet18 \
    --epochs 50 --augment color --img_size 224 \
    --unfreeze_blocks 4 --imbalance_strategy weighted_ce

echo ""
echo "[3/4] Faces ResNet18 | partial unfreeze (2 blocks) | imbalance=none | color | no warmup"
python3 main.py --dataset faces --model resnet18 \
    --epochs 50 --augment color --img_size 224 \
    --unfreeze_blocks 2 --imbalance_strategy none

echo ""
echo "[4/4] Faces ResNet18 | full unfreeze | imbalance=none | augment=all | no warmup"
python3 main.py --dataset faces --model resnet18 \
    --epochs 50 --augment all --img_size 224 \
    --unfreeze_blocks 4 --imbalance_strategy none

echo ""
echo "=========================================="
echo " TOATE CELE 4 RULARI PE FACES FINALIZATE!"
echo "=========================================="
