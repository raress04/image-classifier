"""
visualize_aug.py
Genereaza grafice de vizualizare a augmentarilor pentru AMBELE seturi de date.
Pentru fiecare dataset: cate un exemplu din fiecare clasa x cele 4 tipuri de augmentare.
Cerinta 3.1: cel putin un exemplu din fiecare clasa pentru fiecare dataset.
"""
import os
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from dataset import CSVImageDataset
from main import MOLES_LABEL_MAP, MOLES_CLASSES, MOLES_ROOT, MOLES_NORM
from main import FACES_ROOT, FACES_CLASSES, build_transforms

os.makedirs("graphs", exist_ok=True)

# ---------------------------------------------------------------------------
# Utilitare
# ---------------------------------------------------------------------------
def unnormalize(tensor, mean, std):
    m = torch.tensor(mean).view(3, 1, 1)
    s = torch.tensor(std).view(3, 1, 1)
    return torch.clamp(tensor * s + m, 0, 1)

def find_first_per_class(df, label_col, classes):
    """Returneaza indexul primului rand din fiecare clasa."""
    indices = {}
    for i, row in df.iterrows():
        lbl = row[label_col]
        if lbl not in indices:
            indices[lbl] = i
        if len(indices) == len(classes):
            break
    return indices

def plot_augmentations_grid(ds_none, ds_geom, ds_color, ds_all,
                            class_indices, class_names,
                            mean, std, save_path, title):
    """
    Genereaza un grid: fiecare rand = o clasa, fiecare coloana = un tip de augmentare.
    """
    aug_names = ["Fara augmentare", "Geometric", "Culoare", "Toate"]
    datasets   = [ds_none, ds_geom, ds_color, ds_all]

    n_classes = len(class_indices)
    fig, axes = plt.subplots(n_classes, 4, figsize=(14, 2.5 * n_classes))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for row_idx, (cls_key, sample_idx) in enumerate(class_indices.items()):
        for col_idx, (ds, aug_name) in enumerate(zip(datasets, aug_names)):
            img, _ = ds[sample_idx]
            img_np = unnormalize(img, mean, std).permute(1, 2, 0).numpy()
            ax = axes[row_idx][col_idx]
            ax.imshow(img_np)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(aug_name, fontsize=9, fontweight="bold")
            if col_idx == 0:
                label = class_names[row_idx] if row_idx < len(class_names) else str(cls_key)
                ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=50, va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Salvat: {save_path}")

# ===========================================================================
# 1. MOLES — "Vai de pielea mea"
# ===========================================================================
print("Generez vizualizari augmentari pentru MOLES...")
mean_m, std_m = MOLES_NORM["mean"], MOLES_NORM["std"]

tf_none_m,  _ = build_transforms("none",      mean_m, std_m, 128)
tf_geom_m,  _ = build_transforms("geometric", mean_m, std_m, 128)
tf_color_m, _ = build_transforms("color",     mean_m, std_m, 128)
tf_all_m,   _ = build_transforms("all",       mean_m, std_m, 128)

csv_m = os.path.join(MOLES_ROOT, "train.csv")
dir_m = os.path.join(MOLES_ROOT, "train")

ds_none_m  = CSVImageDataset(csv_m, dir_m, transform=tf_none_m,  label_mapping=MOLES_LABEL_MAP)
ds_geom_m  = CSVImageDataset(csv_m, dir_m, transform=tf_geom_m,  label_mapping=MOLES_LABEL_MAP)
ds_color_m = CSVImageDataset(csv_m, dir_m, transform=tf_color_m, label_mapping=MOLES_LABEL_MAP)
ds_all_m   = CSVImageDataset(csv_m, dir_m, transform=tf_all_m,   label_mapping=MOLES_LABEL_MAP)

# Gasim primul exemplu din fiecare clasa
df_m = pd.read_csv(csv_m)
class_indices_m = {}
for idx, row in df_m.iterrows():
    lbl = row["diagnostic"]
    if lbl not in class_indices_m:
        class_indices_m[lbl] = idx
    if len(class_indices_m) == len(MOLES_CLASSES):
        break

plot_augmentations_grid(
    ds_none_m, ds_geom_m, ds_color_m, ds_all_m,
    class_indices_m, MOLES_CLASSES,
    mean_m, std_m,
    save_path="graphs/augmentations_moles_per_class.png",
    title="Augmentari pe setul 'Vai de pielea mea' — cate o clasa pe rand"
)

# ===========================================================================
# 2. FACES — "You're on Candid Camera"
# ===========================================================================
print("Generez vizualizari augmentari pentru FACES...")
stats_path = os.path.join(FACES_ROOT, "splits", "norm_stats.json")
with open(stats_path) as f:
    stats = json.load(f)
mean_f, std_f = stats["mean"], stats["std"]

tf_none_f,  _ = build_transforms("none",      mean_f, std_f, 128)
tf_geom_f,  _ = build_transforms("geometric", mean_f, std_f, 128)
tf_color_f, _ = build_transforms("color",     mean_f, std_f, 128)
tf_all_f,   _ = build_transforms("all",       mean_f, std_f, 128)

csv_f = os.path.join(FACES_ROOT, "splits", "train.csv")
dir_f = FACES_ROOT

ds_none_f  = CSVImageDataset(csv_f, dir_f, transform=tf_none_f)
ds_geom_f  = CSVImageDataset(csv_f, dir_f, transform=tf_geom_f)
ds_color_f = CSVImageDataset(csv_f, dir_f, transform=tf_color_f)
ds_all_f   = CSVImageDataset(csv_f, dir_f, transform=tf_all_f)

# Gasim primul exemplu din fiecare clasa (labels 1-7)
df_f = pd.read_csv(csv_f)
class_indices_f = {}
for idx, row in df_f.iterrows():
    lbl = int(row["label"])          # 1-7
    if lbl not in class_indices_f:
        class_indices_f[lbl] = idx
    if len(class_indices_f) == len(FACES_CLASSES):
        break

plot_augmentations_grid(
    ds_none_f, ds_geom_f, ds_color_f, ds_all_f,
    class_indices_f, FACES_CLASSES,
    mean_f, std_f,
    save_path="graphs/augmentations_faces_per_class.png",
    title="Augmentari pe setul 'You're on Candid Camera' — cate o clasa pe rand"
)

print("Done! Graficele sunt in folderul graphs/")
