#!/bin/bash

echo "==========================================================="
echo "Antrenare MLP pe setul de date 'FACES' (Candid Camera)"
echo "==========================================================="
python3 main.py --dataset faces --model mlp --epochs 20 --batch_size 64

echo ""
echo "==========================================================="
echo "Antrenare CNN (FARA BatchNorm) pe setul de date 'FACES'"
echo "==========================================================="
python3 main.py --dataset faces --model cnn --epochs 20 --batch_size 64

echo ""
echo "==========================================================="
echo "Antrenare CNN (CU BatchNorm) pe setul de date 'FACES'"
echo "==========================================================="
python3 main.py --dataset faces --model cnn --epochs 20 --batch_size 64 --batchnorm

echo ""
echo "Toate antrenarile pe fete s-au terminat! Poti verifica logurile si graficele."

