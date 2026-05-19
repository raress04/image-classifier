#!/bin/bash

echo "==========================================================="
echo "Experiment FACES 1: Augmentari Geometrice"
echo "==========================================================="
python3 main.py --dataset faces --model cnn --epochs 20 --batch_size 64 --batchnorm --augment geometric

echo ""
echo "==========================================================="
echo "Experiment FACES 2: Augmentari de Culoare (Color)"
echo "==========================================================="
python3 main.py --dataset faces --model cnn --epochs 20 --batch_size 64 --batchnorm --augment color

echo ""
echo "==========================================================="
echo "Experiment FACES 3: Toate Augmentarile (All)"
echo "==========================================================="
python3 main.py --dataset faces --model cnn --epochs 20 --batch_size 64 --batchnorm --augment all

echo ""
echo "Toate antrenarile pe augmentari pentru FACES s-au terminat!"
