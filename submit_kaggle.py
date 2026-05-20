import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable


from dataset import CSVImageDataset
from main import (
    build_transforms,
    build_model,
    MOLES_CLASSES,
    MOLES_LABEL_MAP,
    MOLES_NORM,
    FACES_CLASSES,
    IMAGENET_NORM,
    FACES_ROOT,
    MOLES_ROOT
)

def main():
    parser = argparse.ArgumentParser(description="Genereaza submission.csv pentru Kaggle")
    parser.add_argument("--dataset", choices=["moles", "faces"], required=True)
    parser.add_argument("--model", choices=["mlp", "cnn", "resnet18", "mobilenetv2"], required=True)
    parser.add_argument("--checkpoint", type=str, required=True, help="Calea catre fisierul checkpoint .pt")
    parser.add_argument("--img_size", type=int, default=128, help="Dimensiunea imaginii la intrarea in model")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--batchnorm", action="store_true", help="Activeaza BatchNorm in CNN (daca modelul a fost antrenat asa)")
    parser.add_argument("--dropout_prob", type=float, default=0.3, help="Probabilitatea de dropout pentru MLP")
    parser.add_argument("--output", type=str, default="submission.csv", help="Calea catre fisierul de iesire CSV")
    parser.add_argument("--moles_format", choices=["index0", "index1", "string"], default="index0",
                        help="Formatul etichetelor pentru Moles: index0 (0-6), index1 (1-7), string (nv, mel...)")
    parser.add_argument("--unfreeze_blocks", type=int, default=0,
                        help="Numarul de blocuri ResNet finale de dezghetat")
    parser.add_argument("--complex_head", action="store_true",
                        help="Foloseste capul complex MLP (512->256->BN->ReLU->Dropout->7). Trebuie sa fie acelasi ca la antrenare.")
    parser.add_argument("--tta", action="store_true",
                        help="Test-Time Augmentation: media logits pe imaginea originala + imaginea flip orizontal")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference pe {device}...")
    
    # 1. Stabilim configuratia si clasele
    is_pretrained = args.model in ("resnet18", "mobilenetv2")
    
    if args.dataset == "moles":
        num_classes = len(MOLES_CLASSES)
        norm = IMAGENET_NORM if is_pretrained else MOLES_NORM
        csv_file = os.path.join(MOLES_ROOT, "kaggle_test.csv")
        img_dir = os.path.join(MOLES_ROOT, "kaggle_test")
        
    else:  # faces
        num_classes = len(FACES_CLASSES)
        if is_pretrained:
            norm_mean = IMAGENET_NORM["mean"]
            norm_std = IMAGENET_NORM["std"]
        else:
            import json
            stats_path = os.path.join(FACES_ROOT, "splits", "norm_stats.json")
            with open(stats_path) as f:
                stats = json.load(f)
            norm_mean = stats["mean"]
            norm_std = stats["std"]
        norm = {"mean": norm_mean, "std": norm_std}
        csv_file = os.path.join(FACES_ROOT, "splits", "remote_test.csv")
        # Imaginile de test sunt in subdirectoare pe clasa (1/, 2/, ..., 7/)
        # Construim un index: filename -> cale completa
        test_root = os.path.join(FACES_ROOT, "DATASET", "test")
        fname_to_path = {}
        for cls_dir in os.listdir(test_root):
            cls_path = os.path.join(test_root, cls_dir)
            if os.path.isdir(cls_path):
                for fname in os.listdir(cls_path):
                    fname_to_path[fname] = os.path.join(cls_path, fname)
        img_dir = None  # semnalizam ca folosim index
        
    print(f"Incarcare CSV din: {csv_file}")
    
    # 2. Definim transformarile (identice cu cele de validare)
    _, val_tf = build_transforms(augment="none", norm_mean=norm["mean"], norm_std=norm["std"], img_size=args.img_size)
    
    # 3. Initializam dataset-ul de test (is_test=True)
    if args.dataset == "faces" and img_dir is None:
        # Folosim indexul construit mai sus; suprascriem img_names cu cai complete
        test_ds = CSVImageDataset(
            csv_file=csv_file,
            img_dir=test_root,  # nu conteaza direct, suprascris mai jos
            transform=val_tf,
            is_test=True
        )
        # Salvam filename-urile originale (vor fi Id-urile din submission)
        original_ids = list(test_ds.img_names)
        # Suprascriem img_names cu caile COMPLETE pentru incarcarea imaginilor
        test_ds.img_names = [fname_to_path.get(fn, os.path.join(test_root, fn))
                             for fn in original_ids]
        # img_dir devine "" ca os.path.join("", cale_completa) = cale_completa
        test_ds.img_dir = ""
    else:
        test_ds = CSVImageDataset(
            csv_file=csv_file,
            img_dir=img_dir,
            transform=val_tf,
            is_test=True
        )
        original_ids = None

    
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    
    # 4. Construim modelul
    model = build_model(args, num_classes)
    
    # Incarcam checkpoint-ul
    print(f"Incarcare checkpoint din {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device)
    
    # Daca checkpoint-ul a fost salvat ca model intreg sau state_dict
    if "model_state_dict" in state_dict:
        model.load_state_dict(state_dict["model_state_dict"])
    else:
        model.load_state_dict(state_dict)
        
    model = model.to(device)
    model.eval()
    
    # 5. Rulam predictiile
    ids = []
    preds = []
    
    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Predictie"):
            images = images.to(device)
            logits = model(images)

            # TTA: average logits pe original + flip orizontal
            if getattr(args, 'tta', False):
                images_flip = torch.flip(images, dims=[3])  # flip pe axa W
                logits_flip = model(images_flip)
                logits = (logits + logits_flip) / 2.0

            batch_preds = logits.argmax(dim=1).cpu().numpy()

            ids.extend(img_names)
            preds.extend(batch_preds)
            
    # 6. Salvam formatul Kaggle
    # Pentru Faces: ids contine cai complete (din index); inlocuim cu filename-urile originale
    if args.dataset == "faces" and original_ids is not None:
        ids = original_ids  # filename-urile originale din remote_test.csv
    
    if args.dataset == "moles":
        # Moles: diagnostic ca string (nv, mel...) sau ca integer (0-6 / 1-7)
        if args.moles_format == "string":
            diagnostic_labels = [MOLES_CLASSES[p] for p in preds]
        elif args.moles_format == "index1":
            diagnostic_labels = [int(p) + 1 for p in preds]
        else: # index0
            diagnostic_labels = [int(p) for p in preds]
            
        df_sub = pd.DataFrame({
            "imagine": ids,
            "diagnostic": diagnostic_labels
        })
        df_orig = pd.read_csv(csv_file)
        df_sub = df_sub.set_index("imagine").reindex(df_orig["imagine"]).reset_index()
        df_sub = df_sub.rename(columns={"imagine": "Id", "diagnostic": "Prediction"})
    else:
        # Pentru faces: label ca intreg 1-indexed (1-7)
        int_labels = [int(p) + 1 for p in preds]
        df_sub = pd.DataFrame({
            "id": ids,
            "label": int_labels
        })
        df_orig = pd.read_csv(csv_file)
        df_sub = df_sub.set_index("id").reindex(df_orig["id"]).reset_index()
        # Kaggle Faces asteapta coloanele "id" si "label" (nu Id/Prediction)


    
    # Salvare CSV
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    df_sub.to_csv(args.output, index=False)
    print(f"Submission salvat cu succes in: {args.output}")
    print(f"Numar intrari: {len(df_sub)}")
    print(df_sub.head(10))

if __name__ == "__main__":
    main()
