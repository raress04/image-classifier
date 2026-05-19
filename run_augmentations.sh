#!/bin/bash

echo "==========================================================="
echo "Experiment 1: Augmentari Geometrice"
echo "==========================================================="
python3 main.py --dataset moles --model cnn --epochs 20 --batch_size 64 --batchnorm --augment geometric

echo ""
echo "==========================================================="
echo "Experiment 2: Augmentari de Culoare (Color)"
echo "==========================================================="
python3 main.py --dataset moles --model cnn --epochs 20 --batch_size 64 --batchnorm --augment color

echo ""
echo "==========================================================="
echo "Experiment 3: Toate Augmentarile (All)"
echo "==========================================================="
python3 main.py --dataset moles --model cnn --epochs 20 --batch_size 64 --batchnorm --augment all

echo ""
echo "Toate antrenarile s-au terminat! Poti verifica logurile si graficele in folderul 'graphs/'."
