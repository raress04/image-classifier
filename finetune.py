"""
finetune.py — Fine-tuning ResNet-18 / MobileNet-v2 pre-antrenate pe ImageNet.
Exemplu de utilizare:
  py finetune.py --dataset moles --arch resnet18 --epochs 30 --warmup_epochs 3
  py finetune.py --dataset faces --arch mobilenet --freeze_backbone --no_warmup
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torchvision import transforms

from dataset import CSVImageDataset
from train import run_training, evaluate
from utils import plot_history, plot_confusion_matrix, print_metrics

MOLES_ROOT = "vai-de-pielea-mea"
MOLES_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MOLES_LABEL_MAP = {c: i for i, c in enumerate(MOLES_CLASSES)}
MOLES_NORM = {"mean": [0.7646, 0.5464, 0.5712], "std": [0.1410, 0.1532, 0.1706]}

FACES_ROOT = "you-re-on-candid-camera"
FACES_CLASSES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

# Modele pre-antrenate asteapta 224x224
IMG_SIZE = 224


def build_pretrained_model(arch, num_classes, freeze_backbone=False):
    """Incarca modelul pre-antrenat si inlocuieste capul de clasificare."""
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        head_param_names = ["fc"]

    elif arch == "mobilenet":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        head_param_names = ["classifier"]

    else:
        raise ValueError(f"Arhitectura necunoscuta: {arch}")

    if freeze_backbone:
        # Inghetam toti parametrii, apoi dezghetam capul
        for param in model.parameters():
            param.requires_grad = False
        for name in head_param_names:
            for param in getattr(model, name).parameters():
                param.requires_grad = True

    return model, head_param_names


def build_optimizer(model, head_param_names, lr_head=1e-3, lr_backbone=1e-4, freeze_backbone=False):
    """
    Strategie cu learning rate diferentiat:
    - capul nou → lr mare (lr_head)
    - backbone pre-antrenat → lr mic (lr_backbone)
    """
    if freeze_backbone:
        # Optimizatorul vede DOAR parametrii antrenabili (capul)
        params = filter(lambda p: p.requires_grad, model.parameters())
        return torch.optim.Adam(params, lr=lr_head, weight_decay=1e-4)

    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        is_head = any(name.startswith(h) for h in head_param_names)
        if is_head:
            head_params.append(param)
        else:
            backbone_params.append(param)

    return torch.optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params,     "lr": lr_head},
    ], weight_decay=1e-4)


def build_scheduler(optimizer, warmup_epochs, total_epochs, use_warmup=True):
    """Warmup liniar + cosine decay."""
    if not use_warmup:
        return CosineAnnealingLR(optimizer, T_max=total_epochs)

    warmup = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def load_datasets(args):
    # Normalizare ImageNet standard (obligatorie pentru modele pre-antrenate)
    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        imagenet_norm,
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        imagenet_norm,
    ])

    if args.dataset == "moles":
        train_ds = CSVImageDataset(
            csv_file=os.path.join(MOLES_ROOT, "train.csv"),
            img_dir=os.path.join(MOLES_ROOT, "train"),
            transform=train_tf,
            label_mapping=MOLES_LABEL_MAP,
        )
        val_ds = CSVImageDataset(
            csv_file=os.path.join(MOLES_ROOT, "test.csv"),
            img_dir=os.path.join(MOLES_ROOT, "test"),
            transform=val_tf,
            label_mapping=MOLES_LABEL_MAP,
        )
        class_names = MOLES_CLASSES

    elif args.dataset == "faces":
        train_ds = CSVImageDataset(
            csv_file=os.path.join(FACES_ROOT, "splits", "train.csv"),
            img_dir=FACES_ROOT,
            transform=train_tf,
        )
        val_ds = CSVImageDataset(
            csv_file=os.path.join(FACES_ROOT, "splits", "local_test.csv"),
            img_dir=FACES_ROOT,
            transform=val_tf,
        )
        class_names = FACES_CLASSES
    else:
        raise ValueError(f"Dataset necunoscut: {args.dataset}")

    return train_ds, val_ds, class_names


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning modele pre-antrenate — Tema 2 IA")
    parser.add_argument("--dataset",         choices=["moles", "faces"], required=True)
    parser.add_argument("--arch",            choices=["resnet18", "mobilenet"], default="resnet18")
    parser.add_argument("--epochs",          type=int,   default=30)
    parser.add_argument("--lr_head",         type=float, default=1e-3)
    parser.add_argument("--lr_backbone",     type=float, default=1e-4)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--warmup_epochs",   type=int,   default=3)
    parser.add_argument("--freeze_backbone", action="store_true",
                        help="Ingheata backbone-ul, antreneaza doar capul")
    parser.add_argument("--no_warmup",       action="store_true",
                        help="Dezactiveaza warmup-ul (ablatie)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_ds, val_ds, class_names = load_datasets(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2)

    model, head_param_names = build_pretrained_model(
        args.arch, num_classes=len(class_names), freeze_backbone=args.freeze_backbone
    )
    model = model.to(device)

    optimizer = build_optimizer(
        model, head_param_names,
        lr_head=args.lr_head, lr_backbone=args.lr_backbone,
        freeze_backbone=args.freeze_backbone,
    )
    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        use_warmup=not args.no_warmup,
    )

    frozen_str  = "frozen" if args.freeze_backbone else "full"
    warmup_str  = "no-warmup" if args.no_warmup else f"warmup{args.warmup_epochs}"
    run_name    = f"finetune_{args.dataset}_{args.arch}_{frozen_str}_{warmup_str}_ep{args.epochs}"

    criterion = nn.CrossEntropyLoss()

    history = run_training(
        model, train_loader, val_loader,
        criterion, optimizer,
        num_epochs=args.epochs,
        device=device,
        run_name=run_name,
        scheduler=scheduler,
    )

    _, _, _, _, all_preds, all_labels = evaluate(model, val_loader, criterion, device)
    os.makedirs("graphs", exist_ok=True)
    print_metrics(all_labels, all_preds, class_names)
    plot_history(history, title=run_name, save_path=f"graphs/{run_name}_history.png")
    plot_confusion_matrix(all_labels, all_preds, class_names,
                          title=f"Confusion Matrix — {run_name}",
                          save_path=f"graphs/{run_name}_confusion_matrix.png")


if __name__ == "__main__":
    main()
