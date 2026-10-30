# Neural Image Classification & Transfer Learning Benchmark

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Course:** Artificial Intelligence (IA) — Assignment 2  
**Institution:** Faculty of Automatic Control and Computers, National University of Science and Technology POLITEHNICA Bucharest (UPB)  
**Author:** Balcan Rareș-Ștefan (Group 332CB)

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Datasets](#-datasets)
- [Architectures & Implementations](#-architectures--implementations)
- [Experimental Ablation Studies](#-experimental-ablation-studies)
  - [1. MLP Architecture & Dropout Regularization](#1-mlp-architecture--dropout-regularization)
  - [2. CNN Baseline vs. MLP & Batch Normalization](#2-cnn-baseline-vs-mlp--batch-normalization)
  - [3. Data Augmentation Strategies](#3-data-augmentation-strategies)
  - [4. Class Imbalance Mitigation](#4-class-imbalance-mitigation)
  - [5. Transfer Learning & ResNet-18 Fine-Tuning](#5-transfer-learning--resnet-18-fine-tuning)
  - [6. Curriculum Learning (Bonus Study)](#6-curriculum-learning-bonus-study)
  - [7. Final Competition Models: 120-Epoch ResNet-18, Multi-Seed Ensemble & TTA](#7-final-competition-models-120-epoch-resnet-18-multi-seed-ensemble--tta)
- [Benchmark Results Summary](#-benchmark-results-summary)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage & Reproducibility](#-usage--reproducibility)
  - [Training MLPs](#training-mlps)
  - [Training CNN Baselines & BatchNorm Ablation](#training-cnn-baselines--batchnorm-ablation)
  - [Data Augmentation Experiments](#data-augmentation-experiments)
  - [Imbalance Mitigation](#imbalance-mitigation)
  - [Transfer Learning & Fine-Tuning](#transfer-learning--fine-tuning)
  - [Curriculum Learning](#curriculum-learning)
  - [Ensemble & Kaggle Submission Generation](#ensemble--kaggle-submission-generation)

---

## 🔬 Project Overview

This repository hosts a comprehensive deep learning benchmark for image classification in PyTorch. The project systematically analyzes how modern architectural design choices, regularization, data augmentations, loss reweighting/resampling techniques, transfer learning depths, and curriculum learning paradigms affect classification performance across two drastically different image distributions:
1. **Dermatological lesions** (highly imbalanced medical images, subtle localized textures).
2. **Facial expressions** (human emotion recognition, facial geometry, sensitive to spatial alignments).

All models are evaluated on local stratified validation splits using both **Micro F1 / Accuracy** and **Macro F1** (the official competition metric to prevent dominant-class bias).

---

## 🗂 Datasets

### 1. `vai-de-pielea-mea` (Moles Dataset)
- **Task:** 7-class dermatoscopic skin lesion classification.
- **Classes:** `akiec` (Actinic keratoses), `bcc` (Basal cell carcinoma), `bkl` (Benign keratosis-like lesions), `df` (Dermatofibroma), `mel` (Melanoma), `nv` (Melanocytic nevi), `vasc` (Vascular lesions).
- **Split Sizes:** Train: 6,009 | Local Validation: 2,003 | Kaggle Test: 2,003.
- **Key Challenge:** **Severe Class Imbalance**. The majority class (`nv`) accounts for ~67% of the dataset, while rare classes such as `df` and `vasc` each make up barely ~1% (23 and 29 samples respectively in validation).
- **Normalization:** Per-dataset computed RGB mean `[0.7646, 0.5464, 0.5712]` and std `[0.1410, 0.1532, 0.1706]`.

### 2. `you-re-on-candid-camera` (Faces Dataset)
- **Task:** 7-class facial emotion recognition.
- **Classes:** `Surprise`, `Fear`, `Disgust`, `Happiness`, `Sadness`, `Anger`, `Neutral`.
- **Split Sizes:** Train: 10,434 | Local Validation: 1,837 | Remote Test: 3,068.
- **Key Challenge:** Sub-class visual ambiguity, class imbalance (`Disgust` and `Fear` are rare; `Happiness` is dominant), and sensitivity to facial alignments.
- **Normalization:** Per-dataset computed RGB mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]` (or ImageNet standard when fine-tuning).

---

## 🏗 Architectures & Implementations

### 1. Multi-Layer Perceptron (`SimpleMLP`)
- **Input:** $128 \times 128 \times 3$ flattened RGB vector (49,152 features).
- **Topology:**
  $$\text{Flatten} \rightarrow \text{Linear}(49152 \rightarrow 512) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(p) \rightarrow \text{Linear}(512 \rightarrow 128) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(p) \rightarrow \text{Linear}(128 \rightarrow 7)$$
- **Optimization:** Adam ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$), constant learning rate, 20 epochs.

### 2. Custom Convolutional Neural Network (`SimpleCNN`)
- **Input:** $128 \times 128 \times 3$ tensors.
- **Feature Extractor:** 4 convolutional stages:
  - **Conv1:** $3 \rightarrow 32$ filters (kernel $3\times3$ or $5\times5$ for wider receptive field on Moles) + optional `BatchNorm2d` + `ReLU` + `MaxPool2d(2)`.
  - **Conv2:** $32 \rightarrow 64$ filters ($3\times3$) + optional `BatchNorm2d` + `ReLU` + `MaxPool2d(2)`.
  - **Conv3:** $64 \rightarrow 128$ filters ($3\times3$) + optional `BatchNorm2d` + `ReLU` + `MaxPool2d(2)`.
  - **Conv4:** $128 \rightarrow 256$ filters ($3\times3$) + optional `BatchNorm2d` + `ReLU`.
  - **Global Pooling:** `AdaptiveAvgPool2d((1, 1))` producing a 256-dimensional compact feature embedding.
- **Classifier Head:**
  $$\text{Linear}(256 \rightarrow 128) \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(128 \rightarrow 7)$$

### 3. Pretrained Transfer Learning (`ResNet-18` & `MobileNet-v2`)
- Pretrained backbones on ImageNet-1k ($224 \times 224$ inputs).
- **Head Configurations:**
  - *Standard Linear Head:* Single `Linear(512, 7)`.
  - *Complex MLP Head:* $\text{Linear}(512 \rightarrow 256) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(256 \rightarrow 7)$.
- **Unfreezing Schemes:**
  - Frozen backbone (`unfreeze_blocks=0`).
  - Selective unfreezing of residual stages 3 & 4 (`unfreeze_blocks=2`).
  - Full unfreeze (`unfreeze_blocks=4`) with differential learning rates.

---

## 🧪 Experimental Ablation Studies

### 1. MLP Architecture & Dropout Regularization
Evaluates the effect of Dropout ($p=0.3$ vs. $p=0.0$) on overfitting:

| Dataset | Configuration | Val Acc / F1-micro | Val F1-macro | Train-Val Generalization Gap |
| :--- | :--- | :---: | :---: | :---: |
| **Moles** | Dropout = 0.3 | **0.7369** | 0.3980 | **$\sim 2.0\%$** |
| **Moles** | Dropout = 0.0 | 0.7249 | **0.4446** | $\sim 10.6\%$ |
| **Faces** | Dropout = 0.3 | **0.6560** | 0.5253 | **$\sim 2.2\%$** |
| **Faces** | Dropout = 0.0 | 0.6522 | **0.5291** | $\sim 13.5\%$ |

*Insight:* While $p=0.0$ reaches marginally higher macro F1 due to raw capacity fitting smaller classes, it exhibits severe overfitting (loss diverged on validation with a $>10\%$ train-val gap). Adding $p=0.3$ stabilizes the training dynamics, curbing the gap to $\sim 2\%$.

---

### 2. CNN Baseline vs. MLP & Batch Normalization
Comparing the MLP baseline against the CNN, and quantifying the effect of `BatchNorm2d`:

| Dataset | Model & Configuration | Val Acc / F1-micro | Val F1-macro | Rare Class Behavior |
| :--- | :--- | :---: | :---: | :--- |
| **Moles** | MLP Baseline (DP 0.3) | 0.7384 | 0.4242 | Limited spatial extraction |
| **Moles** | CNN without BatchNorm | 0.6695 | 0.1146 | **Catastrophic underfitting** (all rare classes F1 = 0.00) |
| **Moles** | **CNN with BatchNorm** | **0.7534** | **0.4624** | Gradient flow restored (`vasc` F1: 0.63, `bcc`: 0.52) |
| **Faces** | MLP Baseline (DP 0.3) | **0.6560** | **0.5253** | Strong baseline due to global geometry |
| **Faces** | CNN without BatchNorm | 0.5128 | 0.2471 | Predicts solely dominant emotion (`Happiness`) |
| **Faces** | **CNN with BatchNorm** | 0.5868 | 0.4229 | Significant boost on minority emotions |

*Insight:* Without Batch Normalization, deep CNNs suffer catastrophic gradient vanishing/explosion under standard Adam learning rates, degenerating into trivial majority-class predictors. BatchNorm stabilizes layer input distributions and is mandatory for deep convolutional convergence.

---

### 3. Data Augmentation Strategies
Assessing data augmentations over 20 epochs on CNN (with BatchNorm):
- `none`: Resize & normalization only.
- `geometric`: `RandomCrop(128, padding=8)`, `RandomRotation(15°)`, `RandomHorizontalFlip()`.
- `color`: `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`.
- `all`: Geometric + Color transformations.

| Dataset | Augmentation Scheme | Val Acc / F1-micro | Val F1-macro | Findings |
| :--- | :--- | :---: | :---: | :--- |
| **Moles** | None (Baseline) | **0.7534** | **0.4624** | Starts overfitting after epoch 12 |
| **Moles** | Geometric | 0.7434 | 0.4483 | Invariant to lesion rotation |
| **Moles** | Color | 0.7374 | 0.4430 | Moderate regularization |
| **Moles** | All | 0.7424 | 0.4233 | Underfitting within 20 epochs (requires more epochs) |
| **Faces** | None (Baseline) | 0.5868 | 0.4229 | Overfits facial features |
| **Faces** | **Geometric** | 0.6042 | **0.4532** | Crop/flip boosts expression invariance |
| **Faces** | Color | **0.6157** | 0.4212 | Invariance to lighting variations |
| **Faces** | All | 0.5836 | 0.3626 | Too much noise for 20 epochs |

*Insight:* Heavy data augmentations act as strong regularizers. In short training regimes (20 epochs), `all` causes temporary underfitting. However, when scaling training to 50–120 epochs with pretrained backbones, augmentations become vital for top-tier generalization.

---

### 4. Class Imbalance Mitigation
Evaluating 4 strategies on CNN (with BatchNorm, 20 epochs, no augmentations):
1. **Baseline (`none`):** Standard Cross-Entropy.
2. **Weighted Cross-Entropy (`weighted_ce`):** Loss penalization scaled inversely by class frequency:
   $$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$
3. **Weighted Random Sampler (`sampler`):** Batch-level oversampling of minority classes with replacement (`replacement=True`).
4. **Focal Loss (`focal`):** Down-weighting easy samples with $\gamma=2.0$ and class weights $\alpha$.

#### Results on Moles (`vai-de-pielea-mea`):
| Strategy | Val Acc | Val F1-macro | Rec. `df` (Rare) | Rec. `mel` (Cancer) | Rec. `vasc` (Rare) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (`none`) | **0.7534** | 0.4624 | 0.00 | 0.32 | 0.55 |
| Weighted CE | 0.5731 | 0.4194 | 0.17 | **0.69** | **0.90** |
| **Weighted Sampler** | 0.6430 | **0.4975** | **0.48** | 0.52 | 0.86 |
| Focal Loss | 0.4408 | 0.3628 | 0.30 | 0.48 | 0.79 |

#### Results on Faces (`you-re-on-candid-camera`):
| Strategy | Val Acc | Val F1-macro | Rec. `Fear` | Rec. `Disgust` | Rec. `Anger` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline (`none`) | **0.5868** | 0.4229 | 0.00 | 0.02 | 0.42 |
| Weighted CE | 0.5558 | 0.4004 | 0.26 | 0.07 | 0.21 |
| **Weighted Sampler** | 0.4997 | **0.4740** | **0.43** | **0.61** | 0.37 |
| Focal Loss | 0.3048 | 0.3028 | 0.52 | 0.30 | 0.35 |

*Insight:* `WeightedRandomSampler` is decisively superior for balanced F1-macro. It ensures that every optimization gradient step receives representations from rare classes, rescuing zero-recall classes (e.g. `df` jumps from 0.00 to 0.48, `Disgust` from 0.02 to 0.61).

---

### 5. Transfer Learning & ResNet-18 Fine-Tuning
Evaluating ImageNet-pretrained `ResNet-18` under two primary paradigms:
- **Scenario 1 (Frozen Backbone, 30 epochs):** Only the linear classifier is trained ($\text{lr}=10^{-3}$).
- **Scenario 2 (Selective Unfreeze, 50 epochs):** Unfreezing `layer3` and `layer4` + `WeightedRandomSampler` + data augmentation.

| Dataset | Fine-Tuning Strategy | Epochs | Warmup (3-5 ep) | Val Acc | Val F1-macro |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Moles** | Scenario 1 (Frozen Backbone) | 30 | No | 0.7489 | 0.5195 |
| **Moles** | Scenario 1 (Frozen Backbone) | 30 | Yes | 0.7404 | 0.4992 |
| **Moles** | **Scenario 2 (Unfreeze Layer3+4 + Sampler)** | 50 | No | **0.8432** | **0.7267** |
| **Moles** | Scenario 2 (Unfreeze Layer3+4 + Sampler) | 50 | Yes | 0.8397 | 0.7266 |
| **Faces** | Scenario 1 (Frozen Backbone) | 30 | No | 0.5079 | 0.4004 |
| **Faces** | Scenario 1 (Frozen Backbone) | 30 | Yes | 0.5231 | 0.3971 |
| **Faces** | Scenario 2 (Unfreeze Layer3+4 + Sampler) | 50 | No | 0.7703 | 0.6741 |
| **Faces** | **Scenario 2 (Unfreeze Layer3+4 + Sampler)** | 50 | Yes | **0.7790** | **0.6784** |

---

### 6. Curriculum Learning (Bonus Study)
Implemented in [`curriculum_moles.py`](file:///c:/Users/balca/OneDrive/Facultate/3an/IA/Tema2/curriculum_moles.py) on the Moles dataset using ResNet-18 (40 epochs, `CosineAnnealingLR`):
- **Difficulty Metric:** Image contrast / pixel variance proxy: $\text{Difficulty} = 1.0 - \text{Var}_{\text{norm}}(\text{img})$.
- **Strategy 1 (Static Pacing):** Linear progression from easiest 30% to 100% of data across the first half of training:
  $$g(t) = \min\left(1.0, 0.3 + 0.7 \cdot \frac{t}{T/2}\right)$$
- **Strategy 2 (Self-Paced Learning):** Dynamically training on samples whose per-sample loss is below an increasing percentile threshold $q(t)$.

| Strategy | Val Accuracy / F1-micro | Val F1-macro |
| :--- | :---: | :---: |
| **Baseline (Standard Training, No CL)** | 82.68% | **0.7489** |
| **Strategy 1 (Static Pacing)** | **83.13%** | 0.7277 |
| **Strategy 2 (Self-Paced Loss)** | 82.73% | 0.7228 |

*Key Scientific Finding:* Naive curriculum learning on severely imbalanced datasets **hurts Macro F1**. High-contrast, sharp images in dermatology correspond overwhelmingly to the dominant class (`nv`), while diffuse, low-contrast lesions are frequent in malignant or rare classes (`mel`, `bcc`). Exposing the model to "easy" samples first starved the network of minority classes early on, causing early bias.

---

### 7. Final Competition Models: 120-Epoch ResNet-18, Multi-Seed Ensemble & TTA
To maximize leaderboard performance on the Faces benchmark, an aggressive fine-tuning regime was developed:
- **Full Backbone Unfreeze (`ub4`):** All 4 residual stages unchilled.
- **Complex MLP Head:** $\text{Linear}(512 \rightarrow 256) \rightarrow \text{BatchNorm1d} \rightarrow \text{ReLU} \rightarrow \text{Dropout}(0.3) \rightarrow \text{Linear}(256 \rightarrow 7)$.
- **Differential Learning Rates:** $\text{lr}_{\text{backbone}} = 10^{-4}$, $\text{lr}_{\text{head}} = 10^{-3}$ with `CosineAnnealingLR` over 120 epochs.
- **Multi-Seed Averaging:** Training 3 independent initializations (Seeds 0, 42, 123).
- **Test-Time Augmentation (TTA):** Averaging raw logits from original and horizontally flipped test images:
  $$\text{Logits}_{\text{TTA}}(x) = \frac{f(x) + f(\text{flip}(x))}{2}$$

| Single Model / Ensemble (Faces) | Val Accuracy | Val F1-macro |
| :--- | :---: | :---: |
| ResNet-18 UB4 (Seed 0, Standard) | 82.04% | **0.7408** |
| ResNet-18 UB4 (Seed 42) | 82.04% | 0.7382 |
| ResNet-18 UB4 (Seed 123) | 82.47% | 0.7298 |
| **3-Seed Ensemble + TTA** | **$\ge 83.5\%$** | **$> 0.75$** |

---

## 📊 Benchmark Results Summary

| Model & Setting | Dataset | Strategy / Configuration | Val Acc | Val F1-macro |
| :--- | :--- | :--- | :---: | :---: |
| **MLP (20 ep)** | Moles | No augmentations, DP 0.3 | 0.7369 | 0.3980 |
| **MLP (20 ep)** | Faces | No augmentations, DP 0.3 | 0.6560 | 0.5253 |
| **CNN No BN (20 ep)** | Moles | Baseline (no augmentations) | 0.6695 | 0.1146 |
| **CNN with BN (20 ep)**| Moles | Baseline (no augmentations) | 0.7534 | 0.4624 |
| **CNN with BN (20 ep)**| Moles | Geometric Augmentations | 0.7434 | 0.4483 |
| **CNN with BN (20 ep)**| Moles | Weighted Random Sampler | 0.6430 | 0.4975 |
| **CNN with BN (20 ep)**| Faces | Baseline (no augmentations) | 0.5868 | 0.4229 |
| **CNN with BN (20 ep)**| Faces | Geometric Augmentations | 0.6042 | 0.4532 |
| **CNN with BN (20 ep)**| Faces | Weighted Random Sampler | 0.4997 | 0.4740 |
| **ResNet-18 (30 ep)** | Moles | Frozen Backbone (`ub0`) | 0.7489 | 0.5195 |
| **ResNet-18 (50 ep)** | Moles | Unfreeze Layer3+4 (`ub2`) + Sampler | **0.8432** | **0.7267** |
| **ResNet-18 (50 ep)** | Faces | Unfreeze Layer3+4 (`ub2`) + Sampler | 0.7790 | 0.6784 |
| **ResNet-18 (75 ep)** | Faces | Full Unfreeze (`ub4`) + Weighted CE + All Aug | 0.8140 | 0.7364 |
| **ResNet-18 (120 ep)**| Faces | Full Unfreeze (`ub4`) + Complex Head + Cosine | **0.8204** | **0.7408** |

---

## 📁 Project Structure

```text
.
├── dataset.py                      # Custom PyTorch Dataset (CSVImageDataset) for both benchmarks
├── models.py                       # Implementations of SimpleMLP and SimpleCNN
├── train.py                        # Epoch training loop, validation metrics, TensorBoard logger
├── main.py                         # Primary CLI runner for MLP, CNN, and ResNet experiments
├── finetune.py                     # Specialized fine-tuning runner for pretrained backbones
├── curriculum_moles.py             # Curriculum learning module (Pacing & Self-Paced) on Moles
├── utils.py                        # Plotting utilities for training curves and confusion matrices
├── visualize_aug.py                # Visualizer grid for data augmentations per class
├── plot_combined.py                # Loss comparison utility for MLP Dropout ablation
├── raport.tex                      # Complete academic research report in LaTeX
├── checkpoints_evaluation_results.json # Full evaluation metrics extracted from all checkpoints
├── graphs/                         # High-resolution generated plots and confusion matrices
│   ├── aug_ablation/               # Augmentation ablation history and confusion matrices
│   ├── cnn_baseline/               # Baseline CNN training curves
│   ├── mlp_ablation/               # MLP dropout curves
│   ├── norm_ablation/              # BatchNorm vs No-BatchNorm curves
│   ├── imbalance_ablation/         # Weighted CE, Sampler, and Focal Loss histories
│   └── ...
└── scripts/                        # Automation shell scripts and Kaggle submission generators
    ├── run_faces_final_attack.sh   # 120-epoch full unfreeze runner on Faces
    ├── run_faces_multi_seed.sh     # Multi-seed training runner (Seeds 42, 123)
    ├── submit_kaggle.py            # Kaggle submission generator (single model, TTA support)
    └── submit_ensemble_kaggle.py   # Multi-checkpoint logit-averaging ensemble submission generator
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- NVIDIA GPU with CUDA support (recommended for fine-tuning)

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/raress04/image-classifier.git
cd image-classifier

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install required packages
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn matplotlib pillow tensorboard tqdm
```

---

## 🚀 Usage & Reproducibility

### Training MLPs
```bash
# Train MLP with Dropout 0.3 on Moles
python main.py --dataset moles --model mlp --epochs 20 --dropout_prob 0.3

# Train MLP without Dropout on Faces (Ablation)
python main.py --dataset faces --model mlp --epochs 20 --dropout_prob 0.0
```

### Training CNN Baselines & BatchNorm Ablation
```bash
# Train CNN without BatchNorm (overfitting / gradient vanishing)
python main.py --dataset moles --model cnn --epochs 20 --augment none

# Train CNN with BatchNorm
python main.py --dataset moles --model cnn --epochs 20 --batchnorm --augment none
```

### Data Augmentation Experiments
```bash
# Geometric augmentations
python main.py --dataset moles --model cnn --epochs 20 --batchnorm --augment geometric

# Color jitter augmentations
python main.py --dataset faces --model cnn --epochs 20 --batchnorm --augment color

# Combined (All) augmentations
python main.py --dataset faces --model cnn --epochs 20 --batchnorm --augment all
```

### Imbalance Mitigation
```bash
# Weighted Cross-Entropy Loss
python main.py --dataset moles --model cnn --epochs 20 --batchnorm --imbalance_strategy weighted_ce

# Weighted Random Sampler (Recommended for Imbalanced Datasets)
python main.py --dataset moles --model cnn --epochs 20 --batchnorm --imbalance_strategy sampler

# Focal Loss
python main.py --dataset faces --model cnn --epochs 20 --batchnorm --imbalance_strategy focal
```

### Transfer Learning & Fine-Tuning
```bash
# Scenario 1: Frozen backbone (30 epochs)
python main.py --dataset moles --model resnet18 --epochs 30 --img_size 224 --unfreeze_blocks 0 --augment geometric

# Scenario 2: Unfreeze last 2 residual blocks + Sampler (50 epochs)
python main.py --dataset moles --model resnet18 --epochs 50 --img_size 224 --unfreeze_blocks 2 --imbalance_strategy sampler --augment geometric

# Final Faces Model: Full unfreeze (120 epochs, complex head, CosineAnnealing, differential LR)
python main.py --dataset faces --model resnet18 --epochs 120 --img_size 224 \
    --unfreeze_blocks 4 --imbalance_strategy weighted_ce \
    --complex_head --scheduler cosine \
    --lr_backbone 1e-4 --lr_head 1e-3 --seed 42
```

### Curriculum Learning
```bash
# Run baseline vs static pacing vs self-paced learning (40 epochs)
python curriculum_moles.py --epochs 40 --device cuda
```

### Ensemble & Kaggle Submission Generation
```bash
# Generate single-model submission with Test-Time Augmentation (TTA)
python scripts/submit_kaggle.py \
    --dataset faces \
    --model resnet18 \
    --checkpoint checkpoints/faces_resnet18_best.pt \
    --img_size 224 \
    --complex_head \
    --unfreeze_blocks 4 \
    --tta \
    --output submission_faces_tta.csv

# Multi-seed ensemble prediction averaging (Seeds 0, 42, 123)
python scripts/submit_ensemble_kaggle.py \
    --dataset faces \
    --checkpoints \
        checkpoints/faces_seed0_best.pt \
        checkpoints/faces_seed42_best.pt \
        checkpoints/faces_seed123_best.pt \
    --tta \
    --output submission_ensemble.csv
```

---

## 📜 Academic Integrity & License
This project was developed for academic research and educational evaluation in Artificial Intelligence at UPB. All source code is released under the [MIT License](LICENSE).
