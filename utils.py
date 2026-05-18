import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             f1_score, classification_report)


def plot_history(history, title="Training", save_path=None):
    """Afiseaza curbele de loss si acuratete."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoca")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoca")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_confusion_matrix(all_labels, all_preds, class_names, title="Confusion Matrix", save_path=None):
    """Afiseaza matricea de confuzie."""
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def print_metrics(all_labels, all_preds, class_names):
    """Afiseaza accuracy, F1-macro, F1-micro si raport complet per clasa."""
    acc = np.mean(np.array(all_labels) == np.array(all_preds))
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"F1-macro : {f1_macro:.4f}")
    print(f"F1-micro : {f1_micro:.4f}")
    print("\nRaport per clasa:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    return {"accuracy": acc, "f1_macro": f1_macro, "f1_micro": f1_micro}
