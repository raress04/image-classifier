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

class DummyArgs:
    def __init__(self, model, unfreeze_blocks, complex_head, img_size):
        self.model = model
        self.unfreeze_blocks = unfreeze_blocks
        self.complex_head = complex_head
        self.img_size = img_size
        self.dataset = "faces"  # default
        self.batchnorm = False
        self.dropout_prob = 0.3

def main():
    parser = argparse.ArgumentParser(description="Genereaza submission.csv prin ensemble de multiple modele (soft-voting)")
    parser.add_argument("--dataset", choices=["moles", "faces"], required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Lista de cai catre checkpoint-uri (.pt)")
    parser.add_argument("--models", nargs="+", required=True, help="Lista de tipuri de modele (ex: resnet18 mobilenetv2)")
    parser.add_argument("--unfreeze_blocks", type=int, nargs="+", required=True, help="Lista de unfreeze_blocks (ex: 4 4)")
    parser.add_argument("--complex_heads", type=int, nargs="+", required=True, help="Lista de 0/1 pt complex head (ex: 1 1)")
    parser.add_argument("--img_size", type=int, default=224, help="Rezolutia comuna pentru transformari")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--tta", action="store_true", help="Activeaza TTA (horizontal flip) pe toate modelele")
    parser.add_argument("--output", type=str, default="submissions/ensemble_submission.csv", help="Calea catre fisierul de iesire")
    
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Ensemble Inference pe {device} cu {len(args.checkpoints)} modele...")

    assert len(args.checkpoints) == len(args.models) == len(args.unfreeze_blocks) == len(args.complex_heads), \
        "Lungimea listelor de checkpoint-uri, modele, unfreeze_blocks si complex_heads trebuie sa fie egala!"

    # 1. Incarcam datele de test
    is_pretrained = True # presupunem true pentru ensemble de transfer learning
    if args.dataset == "moles":
        num_classes = len(MOLES_CLASSES)
        norm = IMAGENET_NORM
        csv_file = os.path.join(MOLES_ROOT, "kaggle_test.csv")
        img_dir = os.path.join(MOLES_ROOT, "kaggle_test")
    else:
        num_classes = len(FACES_CLASSES)
        norm = IMAGENET_NORM
        csv_file = os.path.join(FACES_ROOT, "splits", "remote_test.csv")
        test_root = os.path.join(FACES_ROOT, "DATASET", "test")
        fname_to_path = {}
        for cls_dir in os.listdir(test_root):
            cls_path = os.path.join(test_root, cls_dir)
            if os.path.isdir(cls_path):
                for fname in os.listdir(cls_path):
                    fname_to_path[fname] = os.path.join(cls_path, fname)
        img_dir = None

    # Transformari comune
    _, val_tf = build_transforms(augment="none", norm_mean=norm["mean"], norm_std=norm["std"], img_size=args.img_size)

    if args.dataset == "faces" and img_dir is None:
        test_ds = CSVImageDataset(csv_file=csv_file, img_dir=test_root, transform=val_tf, is_test=True)
        original_ids = list(test_ds.img_names)
        test_ds.img_names = [fname_to_path.get(fn, os.path.join(test_root, fn)) for fn in original_ids]
        test_ds.img_dir = ""
    else:
        test_ds = CSVImageDataset(csv_file=csv_file, img_dir=img_dir, transform=val_tf, is_test=True)
        original_ids = None

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 2. Initializam si incarcam toate modelele
    loaded_models = []
    for ckpt, model_name, uf, ch in zip(args.checkpoints, args.models, args.unfreeze_blocks, args.complex_heads):
        print(f"Incarcare model: {model_name} | complex_head={ch} | unfreeze={uf} | ckpt={ckpt}")
        dummy_args = DummyArgs(model=model_name, unfreeze_blocks=uf, complex_head=bool(ch), img_size=args.img_size)
        model = build_model(dummy_args, num_classes)
        
        state_dict = torch.load(ckpt, map_location=device)
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        else:
            model.load_state_dict(state_dict)
            
        model = model.to(device)
        model.eval()
        loaded_models.append(model)

    # 3. Predictie (soft voting)
    ids = []
    preds = []

    with torch.no_grad():
        for images, img_names in tqdm(test_loader, desc="Inference Ensemble"):
            images = images.to(device)
            
            # Acumulam logits de la toate modelele
            batch_logits = torch.zeros((images.size(0), num_classes), device=device)
            
            for model in loaded_models:
                logits = model(images)
                
                if args.tta:
                    images_flip = torch.flip(images, dims=[3])
                    logits_flip = model(images_flip)
                    logits = (logits + logits_flip) / 2.0
                
                # Adunam logits (soft-voting la nivel de logit)
                batch_logits += logits
            
            # Facem media
            batch_logits /= len(loaded_models)
            batch_preds = batch_logits.argmax(dim=1).cpu().numpy()
            
            ids.extend(img_names)
            preds.extend(batch_preds)

    # 4. Salvare submission format
    if args.dataset == "faces" and original_ids is not None:
        ids = original_ids

    if args.dataset == "moles":
        diagnostic_labels = [int(p) for p in preds]
        df_sub = pd.DataFrame({"imagine": ids, "diagnostic": diagnostic_labels})
        df_orig = pd.read_csv(csv_file)
        df_sub = df_sub.set_index("imagine").reindex(df_orig["imagine"]).reset_index()
        df_sub = df_sub.rename(columns={"imagine": "Id", "diagnostic": "Prediction"})
    else:
        int_labels = [int(p) + 1 for p in preds]
        df_sub = pd.DataFrame({"id": ids, "label": int_labels})
        df_orig = pd.read_csv(csv_file)
        df_sub = df_sub.set_index("id").reindex(df_orig["id"]).reset_index()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    df_sub.to_csv(args.output, index=False)
    print(f"Ensemble submission salvat cu succes in: {args.output}")
    print(df_sub.head(10))

if __name__ == "__main__":
    main()
