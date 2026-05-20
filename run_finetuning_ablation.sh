#!/bin/bash
set -e

echo "==========================================================="
echo "ResNet-18, moles, augment=geometric, FARA warmup"
echo "==========================================================="
python3 main.py --dataset moles --model resnet18 --epochs 30 --augment geometric --img_size 224

echo "==========================================================="
echo "ResNet-18, moles, augment=geometric, CU warmup"
echo "==========================================================="
python3 main.py --dataset moles --model resnet18 --epochs 30 --augment geometric --img_size 224 --warmup

echo "==========================================================="
echo "ResNet-18, faces, augment=geometric, FARA warmup"
echo "==========================================================="
python3 main.py --dataset faces --model resnet18 --epochs 30 --augment geometric --img_size 224

echo "==========================================================="
echo "ResNet-18, faces, augment=geometric, CU warmup"
echo "==========================================================="
python3 main.py --dataset faces --model resnet18 --epochs 30 --augment geometric --img_size 224 --warmup
