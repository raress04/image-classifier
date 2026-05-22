# PowerShell script pentru generare submissions pentru modelele multi-seed

$PYTHON_BIN = ".venv-ia2\Scripts\python.exe"
if (-not (Test-Path $PYTHON_BIN)) {
    $PYTHON_BIN = ".venv-ia2\bin\python.exe"
}
if (-not (Test-Path $PYTHON_BIN)) {
    $PYTHON_BIN = "python"
}

Write-Host "Folosim interpretorul Python: $PYTHON_BIN" -ForegroundColor Green
Write-Host "=========================================="
Write-Host " GENERARE SUBMISSIONS PENTRU MODELELE MULTI-SEED"
Write-Host "=========================================="

# 1. Generare submisie model Base (ep120) cu TTA
Write-Host "`n[1/4] Generare submisie model Base (implicit)..." -ForegroundColor Yellow
& $PYTHON_BIN submit_kaggle.py `
    --dataset faces `
    --model resnet18 `
    --img_size 224 `
    --unfreeze_blocks 4 `
    --complex_head `
    --tta `
    --checkpoint checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_best.pt `
    --output submissions/submission_faces_resnet18_ep120_best_tta.csv

# 2. Generare submisie model Seed 42 cu TTA
Write-Host "`n[2/4] Generare submisie model Seed 42..." -ForegroundColor Yellow
& $PYTHON_BIN submit_kaggle.py `
    --dataset faces `
    --model resnet18 `
    --img_size 224 `
    --unfreeze_blocks 4 `
    --complex_head `
    --tta `
    --checkpoint checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_seed42_best.pt `
    --output submissions/submission_faces_resnet18_ep120_seed42_tta.csv

# 3. Generare submisie model Seed 123 cu TTA
Write-Host "`n[3/4] Generare submisie model Seed 123..." -ForegroundColor Yellow
& $PYTHON_BIN submit_kaggle.py `
    --dataset faces `
    --model resnet18 `
    --img_size 224 `
    --unfreeze_blocks 4 `
    --complex_head `
    --tta `
    --checkpoint checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_seed123_best.pt `
    --output submissions/submission_faces_resnet18_ep120_seed123_tta.csv

# 4. Generare Ensemble (Soft-voting pe cele 3 modele) cu TTA
Write-Host "`n[4/4] Generare submisie ENSEMBLE (Base + Seed 42 + Seed 123)..." -ForegroundColor Yellow
& $PYTHON_BIN submit_ensemble_kaggle.py `
    --dataset faces `
    --models resnet18 resnet18 resnet18 `
    --img_size 224 `
    --unfreeze_blocks 4 4 4 `
    --complex_heads 1 1 1 `
    --tta `
    --checkpoints `
        checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_best.pt `
        checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_seed42_best.pt `
        checkpoints/faces_resnet18_aug-all_bnFalse_imb-weighted_ce_sched-cosine_ub4_complexhead_bb1e-04_ep120_seed123_best.pt `
    --output submissions/submission_faces_resnet18_ensemble_seeds_tta.csv

Write-Host "`n==========================================" -ForegroundColor Green
Write-Host " TOATE SUBMISIILE AU FOST GENERATE CU SUCCES!" -ForegroundColor Green
Write-Host " Gasiti fisierele CSV in directorul 'submissions/':"
Write-Host "   - submission_faces_resnet18_ep120_best_tta.csv"
Write-Host "   - submission_faces_resnet18_ep120_seed42_tta.csv"
Write-Host "   - submission_faces_resnet18_ep120_seed123_tta.csv"
Write-Host "   - submission_faces_resnet18_ensemble_seeds_tta.csv (Ensemble)"
Write-Host "==========================================" -ForegroundColor Green
