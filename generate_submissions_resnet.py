import os
import subprocess
import sys

def main():
    # Definirea config-urilor doar pentru modelele ResNet18 (baseline, anterior-imbunatatite si cele agresive noi)
    submissions = [
        # --- BASELINE RESNET18 ---
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
        
        # --- IMPROVED FINE-TUNED RESNET18 (Round 1) ---
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
        },

        # --- AGGRESSIVE FACES RESNET18 (Round 2) ---
        {
            "dataset": "faces",
            "model": "resnet18",
            "checkpoint": "checkpoints/faces_resnet18_aug-color_bnFalse_imb-none_nowarmup_ub4_lr0.001_ep50_best.pt",
            "img_size": 224,
            "output": "submissions/submission_faces_resnet18_ub4_none.csv",
            "extra_args": ["--unfreeze_blocks", "4"]
        },
        {
            "dataset": "faces",
            "model": "resnet18",
            "checkpoint": "checkpoints/faces_resnet18_aug-color_bnFalse_imb-none_nowarmup_ub2_lr0.001_ep50_best.pt",
            "img_size": 224,
            "output": "submissions/submission_faces_resnet18_ub2_none.csv",
            "extra_args": ["--unfreeze_blocks", "2"]
        },
        {
            "dataset": "faces",
            "model": "resnet18",
            "checkpoint": "checkpoints/faces_resnet18_aug-color_bnFalse_imb-weighted_ce_nowarmup_ub4_lr0.001_ep50_best.pt",
            "img_size": 224,
            "output": "submissions/submission_faces_resnet18_ub4_weighted_ce.csv",
            "extra_args": ["--unfreeze_blocks", "4"]
        },
        {
            "dataset": "faces",
            "model": "resnet18",
            "checkpoint": "checkpoints/faces_resnet18_aug-all_bnFalse_imb-none_nowarmup_ub4_lr0.001_ep50_best.pt",
            "img_size": 224,
            "output": "submissions/submission_faces_resnet18_ub4_none_augall.csv",
            "extra_args": ["--unfreeze_blocks", "4"]
        },

        # --- ROUND 3: ub4 + weighted_ce + augall/color, 50 & 75 epoci ---
        {
            "dataset": "faces",
            "model": "resnet18",
            "checkpoint": "checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_nowarmup_ub4_lr0.001_ep75_best.pt",
            "img_size": 224,
            "output": "submissions/submission_faces_resnet18_ub4_weighted_ce_augall_ep75.csv",
            "extra_args": ["--unfreeze_blocks", "4"]
        },
        {
            "dataset": "faces",
            "model": "resnet18",
            "checkpoint": "checkpoints/faces_resnet18_aug-color_bnFalse_imb-weighted_ce_nowarmup_ub4_lr0.001_ep75_best.pt",
            "img_size": 224,
            "output": "submissions/submission_faces_resnet18_ub4_weighted_ce_color_ep75.csv",
            "extra_args": ["--unfreeze_blocks", "4"]
        },
        {
            "dataset": "faces",
            "model": "resnet18",
            "checkpoint": "checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_nowarmup_ub4_lr0.001_ep50_best.pt",
            "img_size": 224,
            "output": "submissions/submission_faces_resnet18_ub4_weighted_ce_augall_ep50.csv",
            "extra_args": ["--unfreeze_blocks", "4"]
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
    print(f"Sumar finalizare: {completed} realizate, {skipped} amanate.")
    print(f"Fisierele CSV sunt in folderul: {os.path.abspath('submissions')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
