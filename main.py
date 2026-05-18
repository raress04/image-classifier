"""
main.py — Entry point principal.
Exemplu de utilizare:
  py main.py --dataset moles --model cnn --epochs 30 --lr 1e-3
  py main.py --dataset faces --model mlp --epochs 20 --augment geometric
"""

import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import CSVImageDataset
from models import SimpleMLP, SimpleCNN
from train import run_training, evaluate
from utils import plot_history, plot_confusion_matrix, print_metrics

# ---------------------------------------------------------------------------
# Configuratii per dataset
# ---------------------------------------------------------------------------
MOLES_ROOT = "vai-de-pielea-mea"
MOLES_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MOLES_LABEL_MAP = {c: i for i, c in enumerate(MOLES_CLASSES)}
MOLES_NORM = {"mean": [0.7646, 0.5464, 0.5712], "std": [0.1410, 0.1532, 0.1706]}

FACES_ROOT = "you-re-on-candid-camera"
FACES_CLASSES = ["Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"]

# ---------------------------------------------------------------------------
# Transformari (augmentari)
# ---------------------------------------------------------------------------
def build_transforms(augment, norm_mean, norm_std, img_size=128):
    """Construieste pipeline-ul de transformari in functie de tipul de augmentare."""
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    if augment == "none":
        train_transform = val_transform

    elif augment == "geometric":
        train_transform = transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            normalize,
        ])

    elif augment == "color":
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    elif augment == "all":
        train_transform = transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        raise ValueError(f"Tip augmentare necunoscut: {augment}")

    return train_transform, val_transform


# ---------------------------------------------------------------------------
# Incarcare date
# ---------------------------------------------------------------------------
def load_datasets(args):
    if args.dataset == "moles":
        norm = MOLES_NORM
        train_tf, val_tf = build_transforms(args.augment, norm["mean"], norm["std"], args.img_size)

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
        stats_path = os.path.join(FACES_ROOT, "splits", "norm_stats.json")
        with open(stats_path) as f:
            stats = json.load(f)
        norm_mean = stats["mean"]
        norm_std = stats["std"]
        train_tf, val_tf = build_transforms(args.augment, norm_mean, norm_std, args.img_size)

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


# ---------------------------------------------------------------------------
# Constructie model
# ---------------------------------------------------------------------------
def build_model(args, num_classes):
    if args.model == "mlp":
        input_size = args.img_size * args.img_size * 3
        return SimpleMLP(num_classes=num_classes, input_size=input_size)

    elif args.model == "cnn":
        # kernel mai mare pentru alunite (features de scara mai mare)
        first_kernel = 5 if args.dataset == "moles" else 3
        return SimpleCNN(num_classes=num_classes, first_kernel_size=first_kernel,
                         use_batchnorm=args.batchnorm)

    else:
        raise ValueError(f"Model necunoscut: {args.model}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tema 2 IA — Clasificare imagini cu PyTorch")
    parser.add_argument("--dataset",    choices=["moles", "faces"], required=True)
    parser.add_argument("--model",      choices=["mlp", "cnn"],     required=True)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--img_size",   type=int,   default=128)
    parser.add_argument("--augment",    choices=["none", "geometric", "color", "all"], default="none")
    parser.add_argument("--batchnorm",  action="store_true",
                        help="Activeaza BatchNorm in CNN (ablatie)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: dataset={args.dataset}, model={args.model}, augment={args.augment}, epochs={args.epochs}")

    # Date
    train_ds, val_ds, class_names = load_datasets(args)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model, loss, optimizer
    model = build_model(args, num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    run_name = (f"{args.dataset}_{args.model}"
                f"_aug-{args.augment}"
                f"_bn{'True' if args.batchnorm else 'False'}"
                f"_lr{args.lr}_ep{args.epochs}")

    # Antrenare
    history = run_training(
        model, train_loader, val_loader,
        criterion, optimizer,
        num_epochs=args.epochs,
        device=device,
        run_name=run_name,
    )

    # Evaluare finala
    _, _, _, _, all_preds, all_labels = evaluate(model, val_loader, criterion, device)

    # Raport
    os.makedirs("graphs", exist_ok=True)
    metrics = print_metrics(all_labels, all_preds, class_names)
    plot_history(history, title=run_name, save_path=f"graphs/{run_name}_history.png")
    plot_confusion_matrix(all_labels, all_preds, class_names,
                          title=f"Confusion Matrix — {run_name}",
                          save_path=f"graphs/{run_name}_confusion_matrix.png")


if __name__ == "__main__":
    main()
