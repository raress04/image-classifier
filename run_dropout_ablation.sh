#!/bin/bash
set -e

echo "==========================================================="
echo "MLP, moles, dropout_prob=0.3"
echo "==========================================================="
python3 main.py --dataset moles --model mlp --epochs 20 --augment none --dropout_prob 0.3

echo "==========================================================="
echo "MLP, moles, dropout_prob=0.0"
echo "==========================================================="
python3 main.py --dataset moles --model mlp --epochs 20 --augment none --dropout_prob 0.0

echo "==========================================================="
echo "MLP, faces, dropout_prob=0.3"
echo "==========================================================="
python3 main.py --dataset faces --model mlp --epochs 20 --augment none --dropout_prob 0.3

echo "==========================================================="
echo "MLP, faces, dropout_prob=0.0"
echo "==========================================================="
python3 main.py --dataset faces --model mlp --epochs 20 --augment none --dropout_prob 0.0
