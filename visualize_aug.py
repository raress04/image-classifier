import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from dataset import CSVImageDataset
from main import MOLES_LABEL_MAP, build_transforms, MOLES_ROOT

def main():
    os.makedirs("graphs", exist_ok=True)
    
    # Val transform is "none", others are augmentations
    tf_none, _ = build_transforms("none", [0.7646, 0.5464, 0.5712], [0.1410, 0.1532, 0.1706], 128)
    tf_geom, _ = build_transforms("geometric", [0.7646, 0.5464, 0.5712], [0.1410, 0.1532, 0.1706], 128)
    tf_color, _ = build_transforms("color", [0.7646, 0.5464, 0.5712], [0.1410, 0.1532, 0.1706], 128)
    tf_all, _ = build_transforms("all", [0.7646, 0.5464, 0.5712], [0.1410, 0.1532, 0.1706], 128)
    
    # Incarcam o poza originala dintr-o clasa minora (ex: mel)
    # Cautam manual o poza cu melanom in train.csv sau luam pur si simplu primul element din dataset
    # Pentru un display curat, refacem imaginea originala prin demaparea normalizarii
    
    def unnormalize(tensor):
        mean = torch.tensor([0.7646, 0.5464, 0.5712]).view(3, 1, 1)
        std = torch.tensor([0.1410, 0.1532, 0.1706]).view(3, 1, 1)
        tensor = tensor * std + mean
        return torch.clamp(tensor, 0, 1)
        
    ds_none = CSVImageDataset(os.path.join(MOLES_ROOT, "train.csv"), os.path.join(MOLES_ROOT, "train"), transform=tf_none, label_mapping=MOLES_LABEL_MAP)
    ds_geom = CSVImageDataset(os.path.join(MOLES_ROOT, "train.csv"), os.path.join(MOLES_ROOT, "train"), transform=tf_geom, label_mapping=MOLES_LABEL_MAP)
    ds_color = CSVImageDataset(os.path.join(MOLES_ROOT, "train.csv"), os.path.join(MOLES_ROOT, "train"), transform=tf_color, label_mapping=MOLES_LABEL_MAP)
    ds_all = CSVImageDataset(os.path.join(MOLES_ROOT, "train.csv"), os.path.join(MOLES_ROOT, "train"), transform=tf_all, label_mapping=MOLES_LABEL_MAP)
    
    idx = 10 # alegem un index arbitrar (o imagine cu o alunita)
    
    img_none, _ = ds_none[idx]
    img_geom, _ = ds_geom[idx]
    img_color, _ = ds_color[idx]
    img_all, _ = ds_all[idx]
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(unnormalize(img_none).permute(1, 2, 0).numpy())
    axes[0].set_title("Original (Fara Augmentare)")
    axes[0].axis('off')
    
    axes[1].imshow(unnormalize(img_geom).permute(1, 2, 0).numpy())
    axes[1].set_title("Geometric (Rotire, Crop, Flip)")
    axes[1].axis('off')
    
    axes[2].imshow(unnormalize(img_color).permute(1, 2, 0).numpy())
    axes[2].set_title("Color (Brightness, Contrast)")
    axes[2].axis('off')
    
    axes[3].imshow(unnormalize(img_all).permute(1, 2, 0).numpy())
    axes[3].set_title("Toate (Geometric + Color)")
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig("graphs/augmentations_demo.png", dpi=150)
    plt.close()
    print("Poza cu augmentarile a fost generata in graphs/augmentations_demo.png")

if __name__ == "__main__":
    main()
