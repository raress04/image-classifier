import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def get_loss_from_tb(log_dir):
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()
    train_loss = [x.value for x in ea.Scalars('Loss/train')]
    val_loss = [x.value for x in ea.Scalars('Loss/val')]
    return train_loss, val_loss

def plot_combined_dropout(dataset):
    runs_dir = "runs"
    dirs = os.listdir(runs_dir)
    
    dp03_dir = None
    dp00_dir = None
    
    for d in dirs:
        if dataset in d and "mlp" in d and "dp0.3" in d:
            dp03_dir = os.path.join(runs_dir, d)
        elif dataset in d and "mlp" in d and "dp0.0" in d:
            dp00_dir = os.path.join(runs_dir, d)
            
    if not dp03_dir or not dp00_dir:
        print(f"Eroare: Nu am gasit logurile pentru {dataset}.")
        return

    train_03, val_03 = get_loss_from_tb(dp03_dir)
    train_00, val_00 = get_loss_from_tb(dp00_dir)

    plt.figure(figsize=(8, 6))
    epochs = range(1, len(train_03) + 1)
    
    # Plot dp=0.3 (Solid lines)
    plt.plot(epochs, train_03, 'b-', label='Train Loss (DP=0.3)', linewidth=2)
    plt.plot(epochs, val_03, 'r-', label='Val Loss (DP=0.3)', linewidth=2)
    
    # Plot dp=0.0 (Dashed lines)
    plt.plot(epochs, train_00, 'b--', label='Train Loss (DP=0.0)', alpha=0.7)
    plt.plot(epochs, val_00, 'r--', label='Val Loss (DP=0.0)', alpha=0.7)

    plt.title(f'Dropout Ablation (MLP) - {dataset.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    out_dir = "graphs/mlp_ablation"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"{dataset}_dropout_ablation.png"))
    plt.close()
    print(f"Salvat grafic pentru {dataset}.")

if __name__ == "__main__":
    os.makedirs("graphs", exist_ok=True)
    plot_combined_dropout("moles")
    plot_combined_dropout("faces")
