#!/bin/bash
set -e

echo "=========================================="
echo " FINE-TUNING IMBUNATATIT cu Unfreeze Selectiv"
echo "=========================================="

# --- MOLES: dezghetam ultimele 2 blocuri (layer3 + layer4) ---
echo ""
echo "[1/4] Moles ResNet18 | unfreeze_blocks=2 | sampler | geometric | no warmup"
python3 main.py --dataset moles --model resnet18 \
    --epochs 50 --augment geometric --img_size 224 \
    --unfreeze_blocks 2 --imbalance_strategy sampler

echo ""
echo "[2/4] Moles ResNet18 | unfreeze_blocks=2 | sampler | geometric | cu warmup"
python3 main.py --dataset moles --model resnet18 \
    --epochs 50 --augment geometric --img_size 224 \
    --unfreeze_blocks 2 --imbalance_strategy sampler --warmup

# --- FACES: dezghetam ultimele 2 blocuri (layer3 + layer4) ---
echo ""
echo "[3/4] Faces ResNet18 | unfreeze_blocks=2 | sampler | color | no warmup"
python3 main.py --dataset faces --model resnet18 \
    --epochs 50 --augment color --img_size 224 \
    --unfreeze_blocks 2 --imbalance_strategy sampler

echo ""
echo "[4/4] Faces ResNet18 | unfreeze_blocks=2 | sampler | color | cu warmup"
python3 main.py --dataset faces --model resnet18 \
    --epochs 50 --augment color --img_size 224 \
    --unfreeze_blocks 2 --imbalance_strategy sampler --warmup

echo ""
echo "=========================================="
echo " TOATE CELE 4 RULARI FINALIZATE!"
echo "=========================================="
