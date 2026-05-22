"""
main.py — Entry point principal.
Exemplu de utilizare:
  py main.py --dataset moles --model cnn --epochs 30 --lr 1e-3
  py main.py --dataset faces --model mlp --epochs 20 --augment geometric
  py main.py --dataset moles --model cnn --epochs 20 --batchnorm --imbalance_strategy weighted_ce
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision.models as tv_models

from dataset import CSVImageDataset
from models import SimpleMLP, SimpleCNN
from train import run_training, evaluate
from utils import plot_history, plot_confusion_matrix, print_metrics

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"  [Reproducibilitate] S-a setat seed-ul: {seed}")

# Statistici ImageNet (obligatorii pentru modelele pre-antrenate)
IMAGENET_NORM = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}

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
# FocalLoss (Sectiunea 4 din Anexa)
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    Reduce ponderea exemplelor clasificate corect cu incredere mare,
    concentrandu-se pe exemplele dificile (adesea clasele minoritare).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha  # tensor [num_classes] sau None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        p_t = torch.exp(-ce)
        loss = (1 - p_t) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss

# ---------------------------------------------------------------------------
# Calcul class weights din train set
# ---------------------------------------------------------------------------
def compute_class_weights(train_ds, num_classes, device):
    """
    Calculeaza greutatile per clasa dupa formula din Anexa:
        w_c = N_total / (K * N_c)
    Returneaza:
        class_weights: tensor [num_classes] pe device
        sample_weights: lista float cu greutatea fiecarui sample
        class_counts: numpy array cu numarul de exemple per clasa
    """
    print("  Calculez distributia claselor pe train set...")
    labels = train_ds.get_labels()
    labels_np = np.array(labels)
    class_counts = np.bincount(labels_np, minlength=num_classes).astype(float)
    total = class_counts.sum()

    print(f"  Total samples train: {int(total)}")
    for i, (count, name) in enumerate(zip(class_counts, range(num_classes))):
        pct = count / total * 100
        print(f"    Clasa {i}: {int(count)} ({pct:.1f}%)")

    # Formula: w_c = N_total / (K * N_c)
    weights = total / (num_classes * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    # Sample weights: fiecare sample primeste greutatea clasei sale
    sample_weights = [weights[l] for l in labels]

    print(f"  Greutati per clasa: {[f'{w:.2f}' for w in weights]}")
    return class_weights, sample_weights, class_counts

# ---------------------------------------------------------------------------
# Constructie criterion (loss function) pe baza strategiei
# ---------------------------------------------------------------------------
def build_criterion(strategy, class_weights, device):
    if strategy == "none":
        return nn.CrossEntropyLoss()
    elif strategy == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    elif strategy == "sampler":
        # Sampler-ul echilibreaza distributia; loss-ul ramane standard
        return nn.CrossEntropyLoss()
    elif strategy == "focal":
        return FocalLoss(alpha=class_weights, gamma=2.0)
    else:
        raise ValueError(f"Strategie necunoscuta: {strategy}")

# ---------------------------------------------------------------------------
# Constructie train loader pe baza strategiei
# ---------------------------------------------------------------------------
def build_train_loader(strategy, train_ds, sample_weights, batch_size, num_workers):
    if strategy == "sampler":
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        # IMPORTANT: shuffle=False cand folosim sampler
        return DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, drop_last=True)
    else:
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, drop_last=True)

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
    is_pretrained = args.model in ("resnet18", "mobilenetv2")

    if args.dataset == "moles":
        # Modelele pretrained cer normalizare ImageNet; celelalte folosesc statistici locale
        norm = IMAGENET_NORM if is_pretrained else MOLES_NORM
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
        if is_pretrained:
            norm_mean = IMAGENET_NORM["mean"]
            norm_std  = IMAGENET_NORM["std"]
        else:
            stats_path = os.path.join(FACES_ROOT, "splits", "norm_stats.json")
            with open(stats_path) as f:
                stats = json.load(f)
            norm_mean = stats["mean"]
            norm_std  = stats["std"]

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
        return SimpleMLP(num_classes=num_classes, input_size=input_size, dropout_prob=args.dropout_prob)

    elif args.model == "cnn":
        first_kernel = 5 if args.dataset == "moles" else 3
        return SimpleCNN(num_classes=num_classes, first_kernel_size=first_kernel,
                         use_batchnorm=args.batchnorm)

    elif args.model == "resnet18":
        model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        # Inghetam complet backbone-ul mai intai
        for param in model.parameters():
            param.requires_grad = False
        
        # Dezghetam selectiv ultimele N blocuri convolutionale
        # 0 = complet inghetat
        # 1 = layer4
        # 2 = layer3 + layer4
        # 3 = layer2 + layer3 + layer4
        # 4 = dezghetat complet (layer1 + layer2 + layer3 + layer4)
        if args.unfreeze_blocks >= 4:
            # Dezghetat complet: toate parametrii backbone-ului
            for name, param in model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = True
        else:
            blocks_to_unfreeze = []
            if args.unfreeze_blocks >= 1:
                blocks_to_unfreeze.append(model.layer4)
            if args.unfreeze_blocks >= 2:
                blocks_to_unfreeze.append(model.layer3)
            if args.unfreeze_blocks >= 3:
                blocks_to_unfreeze.append(model.layer2)
            for block in blocks_to_unfreeze:
                for param in block.parameters():
                    param.requires_grad = True
        
        # Inlocuim capul de clasificare cu unul adaptat la num_classes
        in_features = model.fc.in_features
        if getattr(args, 'complex_head', False):
            # Cap MLP mai adanc: Linear -> BN -> ReLU -> Dropout -> Linear
            model.fc = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes)
            )
            print(f"  ResNet-18: cap COMPLEX (512->256->BN->ReLU->Dropout->{num_classes})")
        else:
            model.fc = nn.Linear(in_features, num_classes)  # requires_grad=True implicit
            print(f"  ResNet-18: cap SIMPLU ({in_features}->{num_classes})")

        if args.unfreeze_blocks == 0:
            frozen_info = "frozen"
        elif args.unfreeze_blocks >= 4:
            frozen_info = "fully_unfrozen"
        else:
            frozen_info = f"unfreeze_last{args.unfreeze_blocks}blocks"
        print(f"  ResNet-18 pretrained: {frozen_info}")
        return model

    elif args.model == "mobilenetv2":
        model = tv_models.mobilenet_v2(weights=tv_models.MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        
        # Dezghetam selectiv
        if args.unfreeze_blocks >= 4:
            for name, param in model.named_parameters():
                if not name.startswith("classifier"):
                    param.requires_grad = True
        elif args.unfreeze_blocks > 0:
            # Dezghetam ultimele straturi ale backbone-ului (features)
            # MobileNet-v2 are 19 module in features (0-18)
            unfreeze_start = max(19 - args.unfreeze_blocks * 2, 0)
            for i in range(unfreeze_start, 19):
                for param in model.features[i].parameters():
                    param.requires_grad = True

        in_features = model.classifier[1].in_features
        if getattr(args, 'complex_head', False):
            # Cap complex MLP adaptat pentru classifier MobileNet
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes)
            )
            print(f"  MobileNet-v2: cap COMPLEX (1280->256->BN->ReLU->Dropout->{num_classes})")
        else:
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(in_features, num_classes)
            )
            print(f"  MobileNet-v2: cap SIMPLU (1280->{num_classes})")

        if args.unfreeze_blocks == 0:
            frozen_info = "frozen"
        elif args.unfreeze_blocks >= 4:
            frozen_info = "fully_unfrozen"
        else:
            frozen_info = f"unfreeze_last{args.unfreeze_blocks}blocks"
        print(f"  MobileNet-v2 pretrained: {frozen_info}")
        return model

    else:
        raise ValueError(f"Model necunoscut: {args.model}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tema 2 IA — Clasificare imagini cu PyTorch")
    parser.add_argument("--dataset",    choices=["moles", "faces"], required=True)
    parser.add_argument("--model",      choices=["mlp", "cnn", "resnet18", "mobilenetv2"], required=True)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-3,
                        help="LR global (folosit daca --lr_backbone/--lr_head nu sunt specificate)")
    parser.add_argument("--lr_backbone", type=float, default=None,
                        help="LR pentru backbone dezghetat (default: lr * 0.1). Suprascrieaza --lr pe backbone.")
    parser.add_argument("--lr_head",     type=float, default=None,
                        help="LR pentru capul de clasificare (default: --lr). Suprascrieaza --lr pe head.")
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--img_size",   type=int,   default=128)
    parser.add_argument("--augment",    choices=["none", "geometric", "color", "all"], default="none")
    parser.add_argument("--dropout_prob", type=float, default=0.3,
                        help="Probabilitatea de dropout pentru MLP (0.0 = fara dropout)")
    parser.add_argument("--batchnorm",  action="store_true",
                        help="Activeaza BatchNorm in CNN (ablatie)")
    parser.add_argument("--complex_head", action="store_true",
                        help="Inlocuieste capul linear simplu cu un MLP (512->256->BN->ReLU->Dropout->7)")
    parser.add_argument("--imbalance_strategy",
                        choices=["none", "weighted_ce", "sampler", "focal"],
                        default="none",
                        help="Strategia de tratare a dezechilibrului de clase (Sectiunea 4)")
    parser.add_argument("--warmup",     action="store_true",
                        help="[DEPRECATED] Foloseste --scheduler warmup_cosine in loc. Pastrat pentru compatibilitate.")
    parser.add_argument("--scheduler",  choices=["none", "cosine", "warmup_cosine"], default="none",
                        help="Tipul de LR scheduler: none (constant), cosine (CosineAnnealing), warmup_cosine (3ep warmup + cosine)")
    parser.add_argument("--unfreeze_blocks", type=int, default=0,
                        help="Numarul de blocuri ResNet finale de dezghetat (0=complet inghetat, 4=complet dezghetat)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed pentru reproducibilitate si ensemble multi-seed")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: dataset={args.dataset}, model={args.model}, augment={args.augment}, "
          f"epochs={args.epochs}, imbalance={args.imbalance_strategy}")

    if args.seed is not None:
        set_seed(args.seed)

    # Date
    train_ds, val_ds, class_names = load_datasets(args)
    num_classes = len(class_names)

    # Calcul class weights (necesar pentru weighted_ce, focal, sampler)
    class_weights, sample_weights, class_counts = compute_class_weights(
        train_ds, num_classes, device
    )

    # Criterion si train loader depind de strategie
    criterion   = build_criterion(args.imbalance_strategy, class_weights, device)
    train_loader = build_train_loader(
        args.imbalance_strategy, train_ds, sample_weights,
        args.batch_size, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model si optimizer
    model = build_model(args, num_classes=num_classes).to(device)

    # Optimizer cu learning rate diferentiat cand backbone e partial/complet dezghetat
    is_pretrained_model = args.model in ("resnet18", "mobilenetv2")
    lr_backbone = getattr(args, 'lr_backbone', None)
    lr_head     = getattr(args, 'lr_head',     None)

    if is_pretrained_model and args.unfreeze_blocks > 0:
        # LR explicit daca e furnizat, altfel fallback la lr*0.1 (backward compat)
        _lr_bb   = lr_backbone if lr_backbone is not None else args.lr * 0.1
        _lr_head = lr_head     if lr_head     is not None else args.lr

        # Selectam capul dinamic pe baza modelului (fc la ResNet, classifier la MobileNet)
        head = model.fc if hasattr(model, 'fc') else model.classifier
        head_params = list(head.parameters())
        head_ids    = {id(p) for p in head_params}
        backbone_params = [p for p in model.parameters()
                           if p.requires_grad and id(p) not in head_ids]
        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": _lr_bb},
            {"params": head_params,     "lr": _lr_head},
        ], weight_decay=1e-4)
        print(f"  Optimizer diferentiat: backbone_lr={_lr_bb:.2e}, head_lr={_lr_head:.2e}")
    else:
        _lr = lr_head if lr_head is not None else args.lr
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(trainable_params, lr=_lr, weight_decay=1e-4)

    # Scheduler
    scheduler_type = getattr(args, 'scheduler', 'none')
    # backward compat: --warmup vechi => warmup_cosine
    if getattr(args, 'warmup', False) and scheduler_type == 'none':
        scheduler_type = 'warmup_cosine'

    scheduler = None
    if scheduler_type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        print(f"  Scheduler: CosineAnnealingLR (T_max={args.epochs})")
    elif scheduler_type == 'warmup_cosine':
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
        warmup_epochs = 3
        warmup  = LambdaLR(optimizer, lr_lambda=lambda e: (e + 1) / warmup_epochs)
        cosine  = CosineAnnealingLR(optimizer, T_max=max(args.epochs - warmup_epochs, 1), eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
        print(f"  Scheduler: warmup({warmup_epochs}ep) + CosineAnnealingLR")
    else:
        print("  Scheduler: none (LR constant)")

    is_pretrained = args.model in ("resnet18", "mobilenetv2")
    # Determinam sufixul scheduler pentru run_name
    _sched_tag = getattr(args, 'scheduler', 'none')
    if getattr(args, 'warmup', False) and _sched_tag == 'none':
        _sched_tag = 'warmup_cosine'
    _head_tag = '_complexhead' if getattr(args, 'complex_head', False) else ''
    _lr_tag   = f"bb{_lr_bb:.0e}" if (is_pretrained_model and args.unfreeze_blocks > 0) else f"lr{args.lr}"
    _seed_tag = f"_seed{args.seed}" if args.seed is not None else ""
    run_name = f"{args.dataset}_{args.model}" + (
        f"_dp{args.dropout_prob}" if args.model == "mlp" else ""
    ) + (
        f"_aug-{args.augment}"
        f"_bn{'True' if args.batchnorm else 'False'}"
        f"_imb-{args.imbalance_strategy}"
    ) + (
        f"_sched-{_sched_tag}" if is_pretrained else ""
    ) + (
        f"_ub{args.unfreeze_blocks}" if is_pretrained and args.unfreeze_blocks > 0 else ""
    ) + _head_tag + f"_{_lr_tag}_ep{args.epochs}" + _seed_tag

    # Antrenare
    history = run_training(
        model, train_loader, val_loader,
        criterion, optimizer,
        num_epochs=args.epochs,
        device=device,
        run_name=run_name,
        scheduler=scheduler,
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
