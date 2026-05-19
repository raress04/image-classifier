#!/bin/bash

# Ablatie completa Sectiunea 4 — Dezechilibru de clase
# Configuratie fixa: CNN + BN + augment=none, 20 epoci, lr=0.001, bs=64
# Se schimba doar --imbalance_strategy


echo "==========================================================="
echo "MOLES — Weighted CrossEntropy"
echo "==========================================================="
python3 main.py --dataset moles --model cnn --epochs 20 --batch_size 64 --batchnorm --augment none --imbalance_strategy weighted_ce

echo ""
echo "==========================================================="
echo "MOLES — WeightedRandomSampler"
echo "==========================================================="
python3 main.py --dataset moles --model cnn --epochs 20 --batch_size 64 --batchnorm --augment none --imbalance_strategy sampler

echo ""
echo "==========================================================="
echo "MOLES — Focal Loss"
echo "==========================================================="
python3 main.py --dataset moles --model cnn --epochs 20 --batch_size 64 --batchnorm --augment none --imbalance_strategy focal

echo "==========================================================="
echo "FACES — Weighted CrossEntropy"
echo "==========================================================="
python3 main.py --dataset faces --model cnn --epochs 20 --batch_size 64 --batchnorm --augment none --imbalance_strategy weighted_ce

echo ""
echo "==========================================================="
echo "FACES — WeightedRandomSampler"
echo "==========================================================="
python3 main.py --dataset faces --model cnn --epochs 20 --batch_size 64 --batchnorm --augment none --imbalance_strategy sampler

echo ""
echo "==========================================================="
echo "FACES — Focal Loss"
echo "==========================================================="
python3 main.py --dataset faces --model cnn --epochs 20 --batch_size 64 --batchnorm --augment none --imbalance_strategy focal

echo ""
echo "Ablatie Sectiunea 4 completa! Verifica graficele in graphs/"
