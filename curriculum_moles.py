"""
curriculum_moles.py — Curriculum Learning pe setul de date Moles (vai-de-pielea-mea).

Ruleaza 3 experimente secvential si le compara:
  1. Baseline          — antrenare normala fara CL
  2. Strategia 1       — Pacing static dupa dificultatea pre-calculata (varianta imaginii)
  3. Strategia 2       — Self-Paced Learning dupa loss-ul per-exemplu al modelului curent

Utilizare:
  python curriculum_moles.py
  python curriculum_moles.py --epochs 40 --device cuda
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tv_models
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from sklearn.metrics import f1_score, classification_report

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Constante
# ---------------------------------------------------------------------------
MOLES_ROOT   = "vai-de-pielea-mea"
MOLES_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
MOLES_LABEL_MAP = {c: i for i, c in enumerate(MOLES_CLASSES)}
NUM_CLASSES  = len(MOLES_CLASSES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE = 224

# ---------------------------------------------------------------------------
# Dataset cu dificultate (Sectiunea 7.1 din enunt)
# ---------------------------------------------------------------------------
class DatasetWithDifficulty(Dataset):
    """
    Dataset care intoarce (imagine, eticheta, scor_dificultate) per sample.
    samples      — lista de (cale_imagine, eticheta_int)
    difficulties — lista de float-uri in [0, 1], 0=usor, 1=greu
    transform    — pipeline de transformari torchvision
    """
    def __init__(self, samples, difficulties, transform=None):
        assert len(samples) == len(difficulties), \
            "Numarul de samples si de scoruri de dificultate trebuie sa fie egal."
        self.samples      = samples
        self.difficulties = difficulties
        self.transform    = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label, self.difficulties[idx]

    def get_labels(self):
        return [label for _, label in self.samples]


# ---------------------------------------------------------------------------
# Calculul scorului de dificultate per imagine (contrast / varianta)
# ---------------------------------------------------------------------------
def compute_image_difficulty(samples):
    """
    Calculeaza un scor de dificultate in [0,1] pentru fiecare imagine din lista.

    Proxy folosit: 1 - varianta_normalizata_a_pixelilor.
      - Imagine cu contrast mare (leziune bine conturata)  => varianta mare => dificultate MICA (usoare)
      - Imagine uniforma / intunecata / neclara             => varianta mica => dificultate MARE (grele)

    Aceasta metrica nu necesita adnotari manuale si se calculeaza o singura data.
    """
    print("  Calculez scorurile de dificultate pe baza variantei pixelilor...")
    variances = []
    for path, _ in samples:
        img = Image.open(path).convert("L")   # grayscale pentru eficienta
        arr = np.array(img, dtype=np.float32)
        variances.append(float(arr.var()))

    v = np.array(variances, dtype=np.float32)
    # Normalizam varianta in [0, 1]
    v_norm = (v - v.min()) / (v.max() - v.min() + 1e-8)
    # Dificultate = inversul usurintei: varianta mare = usor = dificultate mica
    difficulties = 1.0 - v_norm
    print(f"    Dificultate medie: {difficulties.mean():.3f} | "
          f"min: {difficulties.min():.3f} | max: {difficulties.max():.3f}")
    return difficulties.tolist()


# ---------------------------------------------------------------------------
# Functia de pacing (Strategia 1 — Sectiunea 7.2 din enunt)
# ---------------------------------------------------------------------------
def pacing(epoch, total_epochs, start_frac=0.3):
    """
    Procentul din date (de la cele mai usoare la cele mai grele)
    expus modelului la epoca curenta.

    Crestere liniara de la start_frac la 1.0:
      - La epoca 0          => start_frac * 100% din date
      - La epoca total/2    => 100% din date
      - Dupa total/2        => ramane la 100%
    """
    return min(1.0, start_frac + (1.0 - start_frac) * (epoch / (total_epochs * 0.5)))


# ---------------------------------------------------------------------------
# Loss per exemplu (Strategia 2 — Sectiunea 7.3 din enunt)
# ---------------------------------------------------------------------------
@torch.no_grad()
def per_sample_losses(model, dataset, criterion_no_red, device, batch_size=64):
    """
    Evalueaza modelul pe tot dataset-ul si intoarce loss-ul per exemplu.
    Foloseste reduction='none' => un scalar de loss pentru fiecare imagine.
    """
    model.eval()
    losses = []
    # DataLoader simplu, fara shuffle, pentru indexare consistenta
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for batch in loader:
        # DatasetWithDifficulty intoarce (img, label, difficulty)
        images, labels = batch[0], batch[1]
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        l = criterion_no_red(logits, labels)
        losses.append(l.cpu())
    return torch.cat(losses)


# ---------------------------------------------------------------------------
# Un pas de antrenare
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Ruleaza o singura epoca de antrenare pe loader-ul dat."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        images, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return running_loss / total, correct / total


# ---------------------------------------------------------------------------
# Evaluare pe setul de validare complet
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """Evalueaza modelul. Returneaza loss, acc, f1_macro, preds, labels."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)
        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    avg_loss  = running_loss / total
    acc       = correct / total
    f1_macro  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1_macro, all_preds, all_labels


# ---------------------------------------------------------------------------
# Constructie model ResNet-18
# ---------------------------------------------------------------------------
def build_resnet18(num_classes, unfreeze_blocks=2):
    """
    ResNet-18 pre-antrenat pe ImageNet cu cap complex si dezghetare selectiva.
    unfreeze_blocks=2 => dezghetam layer3 + layer4 (ultimele 2 blocuri reziduale).
    """
    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    # Inghetam tot backbone-ul
    for param in model.parameters():
        param.requires_grad = False

    # Dezghetam selectiv ultimele N blocuri
    blocks = []
    if unfreeze_blocks >= 1:
        blocks.append(model.layer4)
    if unfreeze_blocks >= 2:
        blocks.append(model.layer3)
    if unfreeze_blocks >= 3:
        blocks.append(model.layer2)
    if unfreeze_blocks >= 4:
        blocks.append(model.layer1)
    for block in blocks:
        for param in block.parameters():
            param.requires_grad = True

    # Cap de clasificare complex (ca la experimentele finale)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    print(f"  ResNet-18: {unfreeze_blocks} blocuri dezghetate, cap complex (512->256->BN->ReLU->Dropout->{num_classes})")
    return model


# ---------------------------------------------------------------------------
# Incarcare date din CSV
# ---------------------------------------------------------------------------
def load_moles_samples(split="train"):
    """Citeste CSV-ul si intoarce lista de (cale_absoluta, eticheta_int)."""
    import pandas as pd
    csv_path = os.path.join(MOLES_ROOT, f"{split}.csv")
    img_dir  = os.path.join(MOLES_ROOT, split)
    df = pd.read_csv(csv_path)

    # Detectam coloana cu imaginile
    if "imagine" in df.columns:
        img_col = "imagine"
    elif "path" in df.columns:
        img_col = "path"
    else:
        raise KeyError("Nu gasesc coloana cu caile imaginilor in CSV.")

    label_col = "diagnostic" if "diagnostic" in df.columns else "label"
    samples = []
    for _, row in df.iterrows():
        path  = os.path.join(img_dir, str(row[img_col]))
        label = MOLES_LABEL_MAP[row[label_col]]
        samples.append((path, label))
    return samples


# ---------------------------------------------------------------------------
# Grafice
# ---------------------------------------------------------------------------
def plot_training_history(history, title, save_path):
    """Salveaza graficul de loss + accuracy + F1-macro."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(title, fontsize=12)

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()

    axes[2].plot(history["val_f1_macro"], label="F1-macro val", color="green")
    axes[2].set_title("F1-macro (val)")
    axes[2].set_xlabel("Epoca")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Grafic salvat: {save_path}")


def plot_comparison(results, save_path):
    """Grafic comparativ F1-macro pe val pentru toate cele 3 rulari."""
    plt.figure(figsize=(10, 5))
    colors = {"Baseline": "steelblue", "Strategia 1 (Pacing Static)": "darkorange",
              "Strategia 2 (Self-Paced)": "forestgreen"}
    for name, history in results.items():
        plt.plot(history["val_f1_macro"], label=name, color=colors.get(name, "gray"), linewidth=2)
    plt.title("Curriculum Learning — Comparatie F1-macro pe Validare (Moles)", fontsize=13)
    plt.xlabel("Epoca")
    plt.ylabel("F1-macro")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Grafic comparativ salvat: {save_path}")


def plot_difficulty_distribution(difficulties, save_path):
    """Histograma distributiei scorurilor de dificultate."""
    plt.figure(figsize=(8, 4))
    plt.hist(difficulties, bins=50, color="coral", edgecolor="darkred", alpha=0.8)
    plt.title("Distributia Scorurilor de Dificultate — Moles Train Set", fontsize=12)
    plt.xlabel("Scor dificultate (0=usor, 1=greu)")
    plt.ylabel("Numar imagini")
    plt.axvline(np.percentile(difficulties, 30), color="navy",
                linestyle="--", label="Percentila 30% (start pacing)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Histograma dificultate salvata: {save_path}")


def plot_pacing_curve(total_epochs, save_path, start_frac=0.3):
    """Grafic al functiei de pacing: ce % din date e folosit la fiecare epoca."""
    epochs = list(range(total_epochs))
    fracs  = [pacing(e, total_epochs, start_frac) * 100 for e in epochs]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, fracs, color="darkorange", linewidth=2)
    plt.title("Functia de Pacing (Strategia 1) — % Date Folosite per Epoca", fontsize=12)
    plt.xlabel("Epoca")
    plt.ylabel("% din date de antrenare folosite")
    plt.ylim(0, 105)
    plt.axhline(100, color="gray", linestyle="--", alpha=0.5)
    plt.fill_between(epochs, fracs, alpha=0.2, color="darkorange")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Grafic pacing salvat: {save_path}")


# ---------------------------------------------------------------------------
# Rulare experiment
# ---------------------------------------------------------------------------
def run_experiment(name, model, train_dataset, val_loader, total_epochs,
                   device, strategy, difficulties=None, criterion_no_red=None):
    """
    Ruleaza un experiment complet (baseline / strategia1 / strategia2).

    strategy: "baseline" | "pacing" | "self_paced"
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(train_dataset, NUM_CLASSES, device)
    )

    # Optimizer cu LR diferentiat
    head_params     = list(model.fc.parameters())
    head_ids        = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": 1e-4},
        {"params": head_params,     "lr": 1e-3},
    ], weight_decay=1e-4)

    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # Sortam indicii dupa dificultate (pentru Strategia 1)
    if strategy == "pacing" and difficulties is not None:
        sorted_indices = sorted(range(len(train_dataset)),
                                key=lambda i: train_dataset.difficulties[i])

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [],   "val_acc": [],
        "val_f1_macro": [],
        "n_samples_used": [],  # cate sample-uri am folosit per epoca (pentru CL)
    }
    best_f1_macro = 0.0
    best_preds, best_labels = [], []

    for epoch in range(total_epochs):
        # ----------------------------------------------------------------
        # Construim loader-ul de antrenare in functie de strategie
        # ----------------------------------------------------------------
        if strategy == "baseline":
            loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                                num_workers=0, drop_last=True)
            n_used = len(train_dataset)

        elif strategy == "pacing":
            frac   = pacing(epoch, total_epochs, start_frac=0.3)
            n_used = max(64, int(frac * len(sorted_indices)))  # minim un batch
            current_indices = sorted_indices[:n_used]
            subset = Subset(train_dataset, current_indices)
            loader = DataLoader(subset, batch_size=64, shuffle=True,
                                num_workers=0, drop_last=True)

        elif strategy == "self_paced":
            if epoch == 0:
                # Prima epoca: folosim primele 30% cu loss cel mai mic (estimate de la zero)
                # In practica: luam un subset aleator de 30% pentru prima epoca
                n_used = max(128, int(0.3 * len(train_dataset)))
                indices = torch.randperm(len(train_dataset))[:n_used].tolist()
            else:
                losses    = per_sample_losses(model, train_dataset,
                                              criterion_no_red, device, batch_size=64)
                q         = min(1.0, 0.3 + 0.05 * epoch)
                threshold = torch.quantile(losses, q=q)
                indices   = (losses <= threshold).nonzero(as_tuple=True)[0].tolist()
                # Garantam minim 2 batch-uri ca sa nu cada BatchNorm
                n_used    = max(128, len(indices))
                indices   = indices[:n_used]
            subset = Subset(train_dataset, indices)
            loader = DataLoader(subset, batch_size=64, shuffle=True,
                                num_workers=0, drop_last=True)

        # ----------------------------------------------------------------
        # Antrenare + evaluare
        # ----------------------------------------------------------------
        tr_loss, tr_acc = train_one_epoch(model, loader, criterion, optimizer, device)
        # IMPORTANT: evaluarea se face MEREU pe setul de validare COMPLET
        va_loss, va_acc, va_f1_macro, preds, labels = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_f1_macro"].append(va_f1_macro)
        history["n_samples_used"].append(n_used)

        if va_f1_macro > best_f1_macro:
            best_f1_macro = va_f1_macro
            best_preds, best_labels = preds, labels
            ckpt_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(),
                       f"checkpoints/moles_cl_{ckpt_name}_best.pt")

        pct = (n_used / len(train_dataset)) * 100
        print(f"  Ep {epoch+1:02d}/{total_epochs} | "
              f"date folosite: {n_used}/{len(train_dataset)} ({pct:.0f}%) | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.3f} f1={va_f1_macro:.4f} "
              f"[best={best_f1_macro:.4f}]")

    return history, best_f1_macro, best_preds, best_labels


# ---------------------------------------------------------------------------
# Calcul class weights (pentru criterion cu weighted_ce)
# ---------------------------------------------------------------------------
def compute_class_weights(dataset, num_classes, device):
    labels_list = dataset.get_labels() if hasattr(dataset, "get_labels") else \
                  [dataset.dataset.get_labels()[i] for i in dataset.indices] \
                  if isinstance(dataset, Subset) else []
    counts = np.bincount(labels_list, minlength=num_classes).astype(float)
    total  = counts.sum()
    weights = total / (num_classes * counts + 1e-8)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Curriculum Learning pe Moles — Tema 2 IA")
    parser.add_argument("--epochs",          type=int, default=40,
                        help="Numarul de epoci de antrenare per experiment (default: 40)")
    parser.add_argument("--device",          type=str, default=None,
                        help="Dispozitiv: 'cuda' sau 'cpu' (auto-detect daca nu e specificat)")
    parser.add_argument("--unfreeze_blocks", type=int, default=2,
                        help="Cate blocuri ResNet-18 sa dezghetam (default: 2)")
    parser.add_argument("--start_frac",      type=float, default=0.3,
                        help="Fractia initiala de date in pacing (default: 0.3 = 30%%)")
    parser.add_argument("--start_from",      type=int, default=1, choices=[1, 2, 3],
                        help="Incepe de la experimentul N: 1=Baseline, 2=Pacing, 3=SelfPaced (default: 1)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Epoci per experiment: {args.epochs}")
    print(f"Unfreeze blocks: {args.unfreeze_blocks}")

    os.makedirs("graphs", exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Incarcare date
    # ----------------------------------------------------------------
    print("\n[1/6] Incarcare date Moles...")
    train_samples = load_moles_samples("train")
    val_samples   = load_moles_samples("test")
    print(f"  Train: {len(train_samples)} imagini | Val: {len(val_samples)} imagini")

    # ----------------------------------------------------------------
    # 2. Calculul scorurilor de dificultate (o singura data)
    # ----------------------------------------------------------------
    print("\n[2/6] Calculez scorurile de dificultate...")
    difficulties = compute_image_difficulty(train_samples)

    # Grafic distributie
    plot_difficulty_distribution(difficulties, "graphs/moles_cl_difficulty_distribution.png")
    plot_pacing_curve(args.epochs, "graphs/moles_cl_pacing_curve.png", args.start_frac)

    # ----------------------------------------------------------------
    # 3. Transformari
    # ----------------------------------------------------------------
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize,
    ])

    # DatasetWithDifficulty pentru toate cele 3 rulari
    train_dataset = DatasetWithDifficulty(train_samples, difficulties, transform=train_transform)

    # Val dataset simplu (nu are nevoie de dificultate)
    from dataset import CSVImageDataset
    val_dataset = CSVImageDataset(
        csv_file=os.path.join(MOLES_ROOT, "test.csv"),
        img_dir=os.path.join(MOLES_ROOT, "test"),
        transform=val_transform,
        label_mapping=MOLES_LABEL_MAP,
    )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Criterion fara reducere (pentru self-paced)
    criterion_no_red = nn.CrossEntropyLoss(reduction="none")

    # ----------------------------------------------------------------
    # 4. Rulam cele 3 experimente
    # ----------------------------------------------------------------
    all_results = {}
    # Valori placeholder pentru experimentele sarite (pentru graficul comparativ)
    hist_base   = {"val_f1_macro": [0.0] * args.epochs}
    best_f1_base, preds_base, labels_base = 0.0, [], []
    hist_s1     = {"val_f1_macro": [0.0] * args.epochs}
    best_f1_s1, preds_s1, labels_s1 = 0.0, [], []

    # --- Experiment 1: Baseline ---
    if args.start_from <= 1:
        print("\n[3/6] BASELINE — fara Curriculum Learning")
        model_baseline = build_resnet18(NUM_CLASSES, args.unfreeze_blocks).to(device)
        hist_base, best_f1_base, preds_base, labels_base = run_experiment(
        name="baseline",
        model=model_baseline,
        train_dataset=train_dataset,
        val_loader=val_loader,
        total_epochs=args.epochs,
        device=device,
        strategy="baseline",
        )
        all_results["Baseline"] = hist_base
        plot_training_history(hist_base, "Baseline (fara CL) — Moles",
                              "graphs/moles_cl_baseline_history.png")
    else:
        print("\n[3/6] BASELINE — SARIT (start_from=%d)" % args.start_from)

    # --- Experiment 2: Strategia 1 — Pacing static ---
    if args.start_from <= 2:
        print("\n[4/6] STRATEGIA 1 — Pacing Static (dificultate = varianta imagine)")
        model_s1 = build_resnet18(NUM_CLASSES, args.unfreeze_blocks).to(device)
        hist_s1, best_f1_s1, preds_s1, labels_s1 = run_experiment(
        name="strategia_1_pacing",
        model=model_s1,
        train_dataset=train_dataset,
        val_loader=val_loader,
        total_epochs=args.epochs,
        device=device,
        strategy="pacing",
        difficulties=difficulties,
        )
        all_results["Strategia 1 (Pacing Static)"] = hist_s1
        plot_training_history(hist_s1, "Strategia 1: Pacing Static — Moles",
                              "graphs/moles_cl_strategy1_history.png")
    else:
        print("\n[4/6] STRATEGIA 1 — SARITA (start_from=%d)" % args.start_from)

    # --- Experiment 3: Strategia 2 — Self-Paced ---
    print("\n[5/6] STRATEGIA 2 — Self-Paced Learning (dificultate = loss per exemplu)")
    model_s2 = build_resnet18(NUM_CLASSES, args.unfreeze_blocks).to(device)
    hist_s2, best_f1_s2, preds_s2, labels_s2 = run_experiment(
        name="strategia_2_selfpaced",
        model=model_s2,
        train_dataset=train_dataset,
        val_loader=val_loader,
        total_epochs=args.epochs,
        device=device,
        strategy="self_paced",
        criterion_no_red=criterion_no_red,
    )
    all_results["Strategia 2 (Self-Paced)"] = hist_s2
    plot_training_history(hist_s2, "Strategia 2: Self-Paced Learning — Moles",
                          "graphs/moles_cl_strategy2_history.png")

    # ----------------------------------------------------------------
    # 5. Grafic comparativ si raport final
    # ----------------------------------------------------------------
    print("\n[6/6] Generez grafice comparative si raport final...")
    plot_comparison(all_results, "graphs/moles_cl_comparison.png")

    # Tabel sumar
    print("\n" + "="*65)
    print("  SUMAR FINAL — Curriculum Learning pe Moles")
    print("="*65)
    print(f"  {'Experiment':<35} {'Best F1-macro':>13} {'Final F1-macro':>14}")
    print(f"  {'-'*35} {'-'*13} {'-'*14}")

    experiments = [
        ("Baseline (fara CL)",                   hist_base, best_f1_base),
        ("Strategia 1 (Pacing Static)",           hist_s1,   best_f1_s1),
        ("Strategia 2 (Self-Paced)",              hist_s2,   best_f1_s2),
    ]
    for exp_name, hist, best_f1 in experiments:
        final_f1 = hist["val_f1_macro"][-1]
        print(f"  {exp_name:<35} {best_f1:>13.4f} {final_f1:>14.4f}")

    print("="*65)

    # Raport detaliat per clasa pentru cel mai bun model CL
    best_cl_f1   = max(best_f1_s1, best_f1_s2)
    best_cl_name = "Strategia 1" if best_f1_s1 >= best_f1_s2 else "Strategia 2"
    best_preds_cl   = preds_s1 if best_f1_s1 >= best_f1_s2 else preds_s2
    best_labels_cl  = labels_s1 if best_f1_s1 >= best_f1_s2 else labels_s2

    print(f"\n  Cel mai bun model CL: {best_cl_name} (F1-macro={best_cl_f1:.4f})")
    print(f"  Raport per clasa ({best_cl_name}):")
    print(classification_report(best_labels_cl, best_preds_cl,
                                target_names=MOLES_CLASSES, zero_division=0))

    print("\n  Grafice generate in graphs/:")
    for fname in ["moles_cl_difficulty_distribution.png", "moles_cl_pacing_curve.png",
                  "moles_cl_baseline_history.png", "moles_cl_strategy1_history.png",
                  "moles_cl_strategy2_history.png", "moles_cl_comparison.png"]:
        print(f"    - graphs/{fname}")

    print("\n  Checkpointuri salvate in checkpoints/:")
    for fname in ["moles_cl_baseline_best.pt", "moles_cl_strategia_1_pacing_best.pt",
                  "moles_cl_strategia_2_selfpaced_best.pt"]:
        print(f"    - checkpoints/{fname}")

    print("\nFinalizat!")


if __name__ == "__main__":
    main()
