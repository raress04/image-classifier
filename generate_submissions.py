import os
import subprocess
import sys

def main():
    # 1. Definirea config-urilor pentru fiecare model/dataset
    submissions = [
        # --- MOLES (Vai de pielea mea) ---
        {
            "dataset": "moles",
            "model": "mlp",
            "checkpoint": "checkpoints/moles_mlp_dp0.0_aug-none_bnFalse_imb-none_lr0.001_ep20_best.pt",
            "img_size": 128,
            "output": "submissions/submission_moles_mlp.csv",
            "extra_args": ["--dropout_prob", "0.0"]
        },
        {
            "dataset": "moles",
            "model": "cnn",
            "checkpoint": "checkpoints/moles_cnn_aug-none_bnTrue_imb-sampler_lr0.001_ep20_best.pt",
            "img_size": 128,
            "output": "submissions/submission_moles_cnn.csv",
            "extra_args": ["--batchnorm"]
        },
        {
            "dataset": "moles",
            "model": "resnet18",
            "checkpoint": "checkpoints/moles_resnet18_aug-geometric_bnFalse_imb-none_nowarmup_lr0.001_ep30_best.pt",
            "img_size": 224,
            "output": "submissions/submission_moles_resnet18_nowarmup.csv",
            "extra_args": []
        },
        {
            "dataset": "moles",
            "model": "resnet18",
            "checkpoint": "checkpoints/moles_resnet18_aug-geometric_bnFalse_imb-none_warmup_lr0.001_ep30_best.pt",
            "img_size": 224,
            "output": "submissions/submission_moles_resnet18_warmup.csv",
            "extra_args": []
        },
        
        # --- FACES (You're on candid camera) ---
        {
            "dataset": "faces",
            "model": "mlp",
            "checkpoint": "checkpoints/faces_mlp_dp0.0_aug-none_bnFalse_imb-none_lr0.001_ep20_best.pt",
            "img_size": 128,
            "output": "submissions/submission_faces_mlp.csv",
            "extra_args": ["--dropout_prob", "0.0"]
        },
        {
            "dataset": "faces",
            "model": "cnn",
            "checkpoint": "checkpoints/faces_cnn_aug-none_bnTrue_imb-sampler_lr0.001_ep20_best.pt",
            "img_size": 128,
            "output": "submissions/submission_faces_cnn.csv",
            "extra_args": ["--batchnorm"]
        },
        {
            "dataset": "faces",
            "model": "resnet18",
            "checkpoint": "checkpoints/faces_resnet18_aug-geometric_bnFalse_imb-none_nowarmup_lr0.001_ep30_best.pt",
            "img_size": 224,
            "output": "submissions/submission_faces_resnet18_nowarmup.csv",
            "extra_args": []
        },
        {
            "dataset": "faces",
            "model": "resnet18",
            "checkpoint": "checkpoints/faces_resnet18_aug-geometric_bnFalse_imb-none_warmup_lr0.001_ep30_best.pt",
            "img_size": 224,
            "output": "submissions/submission_faces_resnet18_warmup.csv",
            "extra_args": []
        },
        
        # --- IMPROVED FINE-TUNED MODELS (selective unfreeze + sampler + custom augmentations) ---
        {
            "dataset": "moles",
            "model": "resnet18",
            "checkpoint": "checkpoints/moles_resnet18_aug-geometric_bnFalse_imb-sampler_nowarmup_ub2_lr0.001_ep50_best.pt",
            "img_size": 224,
            "output": "submissions/submission_moles_resnet18_nowarmup_improved.csv",
            "extra_args": ["--unfreeze_blocks", "2"]
        },
        {
            "dataset": "moles",
            "model": "resnet18",
            "checkpoint": "checkpoints/moles_resnet18_aug-geometric_bnFalse_imb-sampler_warmup_ub2_lr0.001_ep50_best.pt",
            "img_size": 224,
            "output": "submissions/submission_moles_resnet18_warmup_improved.csv",
            "extra_args": ["--unfreeze_blocks", "2"]
        },
        {
            "dataset": "faces",
            "model": "resnet18",
            "checkpoint": "checkpoints/faces_resnet18_aug-color_bnFalse_imb-sampler_nowarmup_ub2_lr0.001_ep50_best.pt",
            "img_size": 224,
            "output": "submissions/submission_faces_resnet18_nowarmup_improved.csv",
            "extra_args": ["--unfreeze_blocks", "2"]
        },
        {
            "dataset": "faces",
            "model": "resnet18",
            "checkpoint": "checkpoints/faces_resnet18_aug-color_bnFalse_imb-sampler_warmup_ub2_lr0.001_ep50_best.pt",
            "img_size": 224,
            "output": "submissions/submission_faces_resnet18_warmup_improved.csv",
            "extra_args": ["--unfreeze_blocks", "2"]
        }
    ]
    
    python_exec = sys.executable
    print(f"Folosesc Python: {python_exec}")
    os.makedirs("submissions", exist_ok=True)
    
    completed = 0
    skipped = 0
    
    for sub in submissions:
        ckpt = sub["checkpoint"]
        out = sub["output"]
        
        print("\n" + "=" * 80)
        print(f"Procesare: {sub['dataset'].upper()} | {sub['model'].upper()}")
        print(f"Checkpoint: {ckpt}")
        print("=" * 80)
        
        if not os.path.exists(ckpt):
            print(f"[-] Checkpoint-ul nu exista pe disc! Se sare peste el. (Inca nu s-a antrenat)")
            skipped += 1
            continue
            
        # Adaugam explicit --moles_format index0 pentru siguranta daca e moles
        fmt_args = ["--moles_format", "index0"] if sub["dataset"] == "moles" else []
        
        cmd = [
            python_exec, "submit_kaggle.py",
            "--dataset", sub["dataset"],
            "--model", sub["model"],
            "--checkpoint", ckpt,
            "--img_size", str(sub["img_size"]),
            "--output", out
        ] + sub["extra_args"] + fmt_args
        
        print(f"Rulam comanda: {' '.join(cmd)}")
        try:
            res = subprocess.run(cmd, check=True)
            print(f"[+] Succes! Fisierul a fost generat.")
            completed += 1
        except subprocess.CalledProcessError as e:
            print(f"[x] Eroare la rularea scriptului pentru {sub['dataset']} {sub['model']}: {e}")
            
    print("\n" + "=" * 80)
    print(f"Sumar finalizare: {completed} realizate, {skipped} amanate (nu exista inca checkpoint-ul).")
    print(f"Fisierele CSV sunt in folderul: {os.path.abspath('submissions')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
